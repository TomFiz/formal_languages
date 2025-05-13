import os
import re
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json
from pathlib import Path

# Get report files
report_dir = os.environ['RESULTS_DIR'] + '/reports'
reports = glob.glob(f"{report_dir}/report_step_*.txt")

# Function to extract metrics from report files
def extract_metrics(report_file):
    metrics = {}
    
    # Extract step, temperature, and top-k from filename
    filename = os.path.basename(report_file)
    match = re.match(r'report_step_(\d+)_temp([\d\.]+)_topk(\d+)\.txt', filename)
    if match:
        step = int(match.group(1))
        temp = float(match.group(2))
        top_k = int(match.group(3))
        metrics['step'] = step
        metrics['temperature'] = temp
        metrics['top_k'] = top_k
    
    # Parse report file
    with open(report_file, 'r') as f:
        content = f.read()
        
        # Extract basic metrics
        valid_ratio_match = re.search(r'Valid sequences: \d+ \(([\d\.]+)%\)', content)
        if valid_ratio_match:
            metrics['valid_ratio'] = float(valid_ratio_match.group(1)) / 100
        
        support_size_match = re.search(r'Support size: (\d+)', content)
        if support_size_match:
            metrics['support_size'] = int(support_size_match.group(1))
            
        avg_length_match = re.search(r'Average length: ([\d\.]+)', content)
        if avg_length_match:
            metrics['avg_length'] = float(avg_length_match.group(1))
            
        avg_depth_match = re.search(r'Average depth: ([\d\.]+)', content)
        if avg_depth_match:
            metrics['avg_depth'] = float(avg_depth_match.group(1))
            
        # Extract distributions
        length_dist_match = re.search(r'Length distribution: ({.*?})', content)
        if length_dist_match:
            try:
                metrics['length_distribution'] = eval(length_dist_match.group(1))
            except:
                metrics['length_distribution'] = {}
                
        depth_dist_match = re.search(r'Depth distribution: ({.*?})', content)
        if depth_dist_match:
            try:
                metrics['depth_distribution'] = eval(depth_dist_match.group(1))
            except:
                metrics['depth_distribution'] = {}
    
    return metrics

# Extract metrics from all reports
all_metrics = []
for report in reports:
    metrics = extract_metrics(report)
    all_metrics.append(metrics)

# Create DataFrame
df = pd.DataFrame(all_metrics)
df = df.sort_values(['step', 'temperature', 'top_k'])

# Save metrics to CSV
df[['step', 'temperature', 'top_k', 'valid_ratio', 'support_size', 'avg_length', 'avg_depth']].to_csv(
    os.environ['RESULTS_DIR'] + '/metrics_summary.csv', index=False
)

# Create plots directory
plots_dir = os.environ['RESULTS_DIR'] + '/plots'
os.makedirs(plots_dir, exist_ok=True)

# Group by temperature and top_k
grouped = df.groupby(['temperature', 'top_k'])

# Create plots for each metric
metrics_to_plot = ['valid_ratio', 'support_size', 'avg_length', 'avg_depth']
for metric in metrics_to_plot:
    plt.figure(figsize=(10, 6))
    
    for (temp, top_k), group in grouped:
        label = f"temp={temp}, top_k={top_k}"
        plt.plot(group['step'], group[metric], marker='o', label=label)
    
    plt.xlabel('Training Steps')
    plt.ylabel(metric.replace('_', ' ').title())
    plt.title(f'Evolution of {metric.replace("_", " ").title()} Across Training Steps')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"{plots_dir}/evolution_{metric}.png")
    plt.close()

# Create heatmap of depth distribution evolution
# For each temperature/top_k combination
for (temp, top_k), group in grouped:
    # Sort by step
    group = group.sort_values('step')
    
    # Extract depth distributions
    max_depth = 0
    step_depths = []
    steps = []
    
    for _, row in group.iterrows():
        if 'depth_distribution' in row and row['depth_distribution']:
            depth_dist = row['depth_distribution']
            max_depth = max(max_depth, max(int(k) for k in depth_dist.keys()))
            step_depths.append(depth_dist)
            steps.append(row['step'])
    
    if not step_depths:
        continue
        
    # Create matrix for heatmap
    depth_matrix = np.zeros((len(steps), max_depth + 1))
    
    for i, depth_dist in enumerate(step_depths):
        for depth, count in depth_dist.items():
            depth_int = int(depth)
            if depth_int <= max_depth:
                depth_matrix[i, depth_int] = count
    
    # Normalize by row to show distribution
    row_sums = depth_matrix.sum(axis=1, keepdims=True)
    depth_matrix_norm = np.divide(depth_matrix, row_sums, out=np.zeros_like(depth_matrix), where=row_sums!=0)
    
    # Plot heatmap
    plt.figure(figsize=(12, 8))
    plt.imshow(depth_matrix_norm, aspect='auto', cmap='viridis')
    plt.colorbar(label='Probability')
    plt.xlabel('Depth')
    plt.ylabel('Training Step')
    plt.title(f'Evolution of Depth Distribution (temp={temp}, top_k={top_k})')
    plt.yticks(range(len(steps)), [str(s) for s in steps])
    plt.xticks(range(max_depth + 1))
    plt.tight_layout()
    plt.savefig(f"{plots_dir}/depth_heatmap_temp{temp}_topk{top_k}.png")
    plt.close()

# Create dashboard HTML with all plots
html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>Formal Language Evaluation Results</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .container { max-width: 1200px; margin: 0 auto; }
        .metric-section { margin-bottom: 40px; }
        h1, h2 { color: #333; }
        .plot-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
        img { max-width: 100%; border: 1px solid #ddd; }
        table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
        tr:nth-child(even) { background-color: #f9f9f9; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Formal Language Model Evaluation Dashboard</h1>
        
        <div class="metric-section">
            <h2>Evolution of Metrics</h2>
            <div class="plot-grid">
"""

# Add evolution plots
for metric in metrics_to_plot:
    html_content += f"""
                <div>
                    <h3>{metric.replace('_', ' ').title()}</h3>
                    <img src="plots/evolution_{metric}.png" alt="{metric} evolution">
                </div>
    """

html_content += """
            </div>
        </div>
        
        <div class="metric-section">
            <h2>Depth Distribution Evolution</h2>
            <div class="plot-grid">
"""

# Add depth heatmaps
for temp in df['temperature'].unique():
    for top_k in df['top_k'].unique():
        heatmap_file = f"plots/depth_heatmap_temp{temp}_topk{top_k}.png"
        if os.path.exists(os.environ['RESULTS_DIR'] + '/' + heatmap_file):
            html_content += f"""
                <div>
                    <h3>Temperature {temp}, Top-K {top_k}</h3>
                    <img src="{heatmap_file}" alt="Depth distribution heatmap">
                </div>
            """

html_content += """
            </div>
        </div>
        
        <div class="metric-section">
            <h2>Metrics Summary</h2>
            <table>
                <tr>
                    <th>Step</th>
                    <th>Temperature</th>
                    <th>Top-K</th>
                    <th>Valid Ratio</th>
                    <th>Support Size</th>
                    <th>Avg Length</th>
                    <th>Avg Depth</th>
                </tr>
"""

# Add table rows
df_sorted = df.sort_values(['step', 'temperature', 'top_k'])
for _, row in df_sorted.iterrows():
    if all(metric in row for metric in ['step', 'temperature', 'top_k', 'valid_ratio', 'support_size', 'avg_length', 'avg_depth']):
        html_content += f"""
                <tr>
                    <td>{row['step']}</td>
                    <td>{row['temperature']}</td>
                    <td>{row['top_k']}</td>
                    <td>{row['valid_ratio']:.4f}</td>
                    <td>{row['support_size']}</td>
                    <td>{row['avg_length']:.2f}</td>
                    <td>{row['avg_depth']:.2f}</td>
                </tr>
        """

html_content += """
            </table>
        </div>
    </div>
</body>
</html>
"""

# Write HTML dashboard
with open(os.environ['RESULTS_DIR'] + '/dashboard.html', 'w') as f:
    f.write(html_content)

print("Analysis complete. Results saved to:", os.environ['RESULTS_DIR'])
print("Dashboard created at:", os.environ['RESULTS_DIR'] + '/dashboard.html')
