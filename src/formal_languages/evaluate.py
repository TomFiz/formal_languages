import argparse
import json
import numpy as np
import pandas as pd # type: ignore
import matplotlib.pyplot as plt # type: ignore
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional

from .languages import Dyck, ShuffleDyck, Tokenizer


def load_metadata(metadata_file: str) -> Dict[str, Any]:
    """Load metadata from a JSON file."""
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    return metadata


def reconstruct_language(metadata: Dict[str, Any]) -> Tuple[Any, Tokenizer]:
    """Reconstruct the language class from metadata."""
    language_name = metadata["language"]
    opening = metadata["opening"]
    closing = metadata["closing"]
    max_depth = metadata["max_depth"]
    
    if language_name.lower() == "dyck":
        language = Dyck(opening, closing, max_depth)
    elif language_name.lower() == "shuffledyck":
        language = ShuffleDyck(opening, closing, max_depth)
    else:
        raise ValueError(f"Unsupported language: {language_name}")
    
    # Reconstruct tokenizer from metadata
    char_to_int = metadata["char_to_int"]
    bos_token = metadata["bos_token"]
    eos_token = metadata["eos_token"]
    tokenizer = Tokenizer(char_to_int, bos_token, eos_token)
    
    return language, tokenizer


def load_sequences(input_file: str, tokenizer: Tokenizer) -> List[str]:
    """Load and decode sequences from an input file."""
    sequences = []
    
    # Determine file format based on extension
    ext = Path(input_file).suffix.lower()
    
    if ext == '.jsonl':
        # Load from JSONL format
        with open(input_file, 'r') as f:
            for line in f:
                data = json.loads(line)
                if 'tokens' in data:
                    # If tokens are provided directly
                    tokens = data['tokens']
                    support_size = pd.Series(tokens).nunique()
                    # Split into individual sequences if they're batched
                    i = 0
                    while i < len(tokens):
                        # Find the next EOS token
                        j = i
                        while j < len(tokens) and tokens[j] != tokenizer.eos_token:
                            j += 1
                        
                        if j < len(tokens):  # Found an EOS token
                            seq_tokens = tokens[i:j+1]
                            # Only process complete sequences with BOS and EOS
                            if seq_tokens[0] == tokenizer.bos_token:
                                sequences.append(tokenizer.decode(seq_tokens))
                        i = j + 1
                elif 'sequence' in data:
                    # If sequence is provided as a string
                    sequences.append(data['sequence'])
    elif ext == '.txt':
        # Load plain text, one sequence per line
        with open(input_file, 'r') as f:
            sequences = [line.strip() for line in f if line.strip()]
    else:
        raise ValueError(f"Unsupported file format: {ext}")
    
    return sequences, support_size


def calculate_depth(sequence: str, opening: str, closing: str) -> int:
    """Calculate the maximum nesting depth of a sequence."""
    max_depth = 0
    current_depth = 0
    
    for char in sequence:
        if char in opening:
            current_depth += 1
            max_depth = max(max_depth, current_depth)
        elif char in closing:
            current_depth -= 1
    
    return max_depth


def validate_sequences(sequences: List[str], language, metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Validate sequences and compute statistics."""
    opening = metadata["opening"]
    closing = metadata["closing"]
    expected_length = metadata["sequence_length"]
    max_depth = metadata["max_depth"]
    
    # Initialize statistics
    stats = {
        "total_sequences": len(sequences),
        "valid_sequences": 0,
        "invalid_sequences": 0,
        "lengths": [],
        "depths": [],
        "valid_ratio": 0.0,
        "avg_length": 0.0,
        "avg_depth": 0.0,
        "max_observed_depth": 0,
        "expected_length": expected_length,
        "max_allowed_depth": max_depth,
    }
    
    # Process each sequence
    detailed_results = []
    
    for seq in sequences:
        is_valid = language.is_valid(seq)
        seq_length = len(seq)
        seq_depth = calculate_depth(seq, opening, closing)
        
        if is_valid:
            stats["valid_sequences"] += 1
        else:
            stats["invalid_sequences"] += 1
        
        stats["lengths"].append(seq_length)
        stats["depths"].append(seq_depth)
        stats["max_observed_depth"] = max(stats["max_observed_depth"], seq_depth)
        
        detailed_results.append({
            "sequence": seq,
            "valid": is_valid,
            "length": seq_length,
            "depth": seq_depth
        })
    
    # Calculate aggregated statistics
    if sequences:
        stats["valid_ratio"] = stats["valid_sequences"] / stats["total_sequences"]
        stats["avg_length"] = sum(stats["lengths"]) / len(stats["lengths"])
        stats["avg_depth"] = sum(stats["depths"]) / len(stats["depths"])
    
    # Add distribution of lengths and depths
    length_bins = np.bincount(stats["lengths"])
    depth_bins = np.bincount(stats["depths"])
    
    stats["length_distribution"] = {
        i: int(count) for i, count in enumerate(length_bins) if count > 0
    }
    stats["depth_distribution"] = {
        i: int(count) for i, count in enumerate(depth_bins) if count > 0
    }
    
    return {
        "summary": stats,
        "detailed_results": detailed_results
    }


def generate_report(validation_results: Dict[str, Any], output_file: Optional[str] = None):
    """Generate a report of validation results."""
    stats = validation_results["summary"]
    
    # Create a formatted report
    report = [
        "Validation Report",
        "================",
        f"Total sequences: {stats['total_sequences']}",
        f"Valid sequences: {stats['valid_sequences']} ({stats['valid_ratio']:.2%})",
        f"Invalid sequences: {stats['invalid_sequences']} ({1 - stats['valid_ratio']:.2%})",
        "",
        f"Support size: {stats['support_size']}",
        "",
        "Length Statistics:",
        f"  Expected length: {stats['expected_length']}",
        f"  Average length: {stats['avg_length']:.2f}",
        f"  Length distribution: {dict(sorted(stats['length_distribution'].items()))}",
        "",
        "Depth Statistics:",
        f"  Maximum allowed depth: {stats['max_allowed_depth']}",
        f"  Maximum observed depth: {stats['max_observed_depth']}",
        f"  Average depth: {stats['avg_depth']:.2f}",
        f"  Depth distribution: {dict(sorted(stats['depth_distribution'].items()))}",
    ]
    
    # Print the report
    print("\n".join(report))
    
    # Save the report if an output file is specified
    if output_file:
        with open(output_file, 'w') as f:
            f.write("\n".join(report))
        
        # Save detailed results as CSV
        df = pd.DataFrame(validation_results["detailed_results"])
        csv_file = Path(output_file).with_suffix('.csv')
        df.to_csv(csv_file, index=False)
        print(f"Detailed results saved to {csv_file}")
        
        if stats["total_sequences"] > 0 :
            # Generate plots
            plot_file = Path(output_file).with_suffix('.png')
	    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

            # Length distribution
            lengths = np.array(stats["lengths"])
            ax1.hist(lengths, bins=range(min(lengths) - 2, max(lengths) + 2), alpha=0.7)
            ax1.axvline(stats["expected_length"], color='r', linestyle='--', label=f'Expected ({stats["expected_length"]})')
            ax1.axvline(stats["avg_length"], color='g', linestyle='-', label=f'Average ({stats["avg_length"]:.2f})')
            ax1.set_title('Sequence Length Distribution')
            ax1.set_xlabel('Length')
	    ax1.set_ylabel('Count')
	    ax1.legend()

            # Depth distribution
            depths = np.array(stats["depths"])
	    ax2.hist(depths, bins=range(0, max(depths) + 2), alpha=0.7)
	    ax2.axvline(stats["max_allowed_depth"]+0.5, color='r', linestyle='--', label=f'Max allowed ({stats["max_allowed_depth"]})')
            ax2.axvline(stats["avg_depth"], color='g', linestyle='-', label=f'Average ({stats["avg_depth"]:.2f})')
            ax2.set_title('Nesting Depth Distribution')
            ax2.set_xlabel('Depth')
	    ax2.set_ylabel('Count')
	    ax2.legend()
        
	    plt.tight_layout()
            plt.savefig(plot_file)
	    print(f"Plots saved to {plot_file}")


def main():
    parser = argparse.ArgumentParser(description="Validate sequences against a formal language.")
    parser.add_argument("--metadata", type=str, required=True, help="Path to metadata JSON file")
    parser.add_argument("--input", type=str, required=True, help="Path to input file with sequences to validate")
    parser.add_argument("--output", type=str, help="Path to output file for validation report")
    
    args = parser.parse_args()
    
    # Load metadata
    metadata = load_metadata(args.metadata)
    print(f"Loaded metadata for {metadata['language']} language")
    
    # Reconstruct the language
    language, tokenizer = reconstruct_language(metadata)
    print(f"Reconstructed {metadata['language']} language")
    
    # Load sequences
    sequences, support_size = load_sequences(args.input, tokenizer)
    print(f"Loaded {len(sequences)} sequences for validation")
    
    # Validate sequences
    validation_results = validate_sequences(sequences, language, metadata)
    validation_results["summary"]["support_size"] = support_size
    
    # Generate report
    generate_report(validation_results, args.output)
    
    print("Validation complete!")


if __name__ == "__main__":
    main()
