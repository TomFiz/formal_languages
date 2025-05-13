#!/bin/bash
#SBATCH -J eval_lang
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --time=23:59:00
#SBATCH --nodelist=puck6
#SBATCH --output=./logs/%x-%j.log

export LM_DIR=/scratch2/tfizycki/lm-main
export FORMAL_DIR=/scratch2/tfizycki/formal_languages
export CHECKPOINT_DIR=${LM_DIR}/workdir/formal_languages/SD_100M_64
export METADATA_PATH=/scratch2/tfizycki/data/formal/1024/SD_100M_64_metadata.json
export RESULTS_DIR=${FORMAL_DIR}/results

# Create directories if they don't exist
mkdir -p ${RESULTS_DIR}
mkdir -p ${RESULTS_DIR}/reports
mkdir -p ${RESULTS_DIR}/plots

# Define temperature and top-k parameters
TEMPERATURES=(0)
TOP_K_VALUES=(0)

# Part 1: Generate samples for each checkpoint
cd $LM_DIR
source lm/.venv/bin/activate

# Find all checkpoint files
echo "Looking for checkpoints in ${CHECKPOINT_DIR}..."
CHECKPOINTS=($(find ${CHECKPOINT_DIR} -name "step_*.pt" | sort -V))
echo "Found ${#CHECKPOINTS[@]} checkpoints."

# Generate samples for each checkpoint with different parameters
for CHECKPOINT in "${CHECKPOINTS[@]}"; do
    STEP=$(basename "$CHECKPOINT" .pt)
    echo "Processing checkpoint: $STEP"
    
    for TEMP in "${TEMPERATURES[@]}"; do
        for TOP_K in "${TOP_K_VALUES[@]}"; do
            for seed in {0..9}; do
                # Generate samples with the current parameters
                echo "Generating samples with seed $seed..."
                
                # Create a unique output file name based on parameters
                CHECKPOINT_NAME=$(basename "$CHECKPOINT")
                OUTPUT_FILE="generated_tokens_${STEP}_temp${TEMP}_topk${TOP_K}.jsonl"
                PARAM_STR="${STEP}_temp${TEMP}_topk${TOP_K}"
                
                echo "Generating with temperature=$TEMP, top-k=$TOP_K"
                python -m lm.generate "$CHECKPOINT" meta-llama/Llama-2-7b-hf \
                    --prompt="" \
                    --temperature="$TEMP" \
                    --top-k="$TOP_K" \
                    --max_new_tokens=1024 \
                    --output="$OUTPUT_FILE" \
                    --seed="$seed"
                    
                echo "Generated $OUTPUT_FILE"
                
            # Part 2: Evaluate the generated samples
            cd $FORMAL_DIR
            source .venv/bin/activate
            
            echo "Evaluating generated sequences..."
            python -m formal_languages.evaluate \
                --metadata="$METADATA_PATH" \
                --input="${LM_DIR}/${OUTPUT_FILE}" \
                --output="${RESULTS_DIR}/reports/report_${PARAM_STR}.txt"
                
            echo "Evaluation complete for $PARAM_STR"
            
            # Return to LM directory for next generation
            cd $LM_DIR
            source lm/.venv/bin/activate
        done
    done
done

# Part 3: Extract metrics and plot evolution over checkpoints
cd $FORMAL_DIR
source .venv/bin/activate

# Run the plotting script
python ${RESULTS_DIR}/plot_evolution.py

echo "Evaluation and plotting complete!"
echo "Results available in ${RESULTS_DIR}"
echo "Dashboard available at ${RESULTS_DIR}/dashboard.html"