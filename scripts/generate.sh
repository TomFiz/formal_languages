#!/bin/bash
#SBATCH -J generate_formal_data
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=cpu
#SBATCH --time=01:00:00
#SBATCH --output=./logs/%x-%j.log


srun python -m formal_languages.generate \
	--language=ShuffleDyck \
	--max_depth=5\
	--vocab_size=64\
	--num_samples=640000\
	--sequence_length=14\
	--output_file=/scratch2/tfizycki/data/formal/1024/SD_100M_64_valid.jsonl
