import argparse
import random
import string
import json
import numpy as np

from .languages import Dyck, ShuffleDyck, Tokenizer

def generate_unique_chars(n):
    """Generate n unique characters for opening brackets and n for closing brackets."""
    all_chars = list(string.ascii_letters + string.digits + string.punctuation)
    # Remove characters that might cause confusion in the output
    for c in ['(', ')', '[', ']', '{', '}', '<', '>', ' ', '\t', '\n', '"', "'"]:
        if c in all_chars:
            all_chars.remove(c)
    
    # Ensure we have enough unique characters
    if len(all_chars) < 2 * n:
        raise ValueError(f"Cannot generate {n} unique characters for both opening and closing brackets. Maximum is {len(all_chars) // 2}.")
    
    # Select unique characters for opening and ensure no overlap with closing
    random.shuffle(all_chars)
    opening = ''.join(all_chars[:n])
    closing = ''.join(all_chars[n:2*n])
    
    return opening, closing

def generate_diverse_samples(language_class, opening, closing, max_depth, num_samples, sequence_length, seed=42):
    """Generate diverse samples from the language."""
    language = language_class(opening, closing, max_depth)
    samples = []
    seeds = list(range(seed, seed + num_samples))
    
    # For ShuffleDyck, use different distribution strategies to increase diversity
    if language_class == ShuffleDyck:
        distributions = ['type-uniform', 'bracket-uniform', 'length-penalty']
        penalties = [0.5, 1.0, 2.0]
        
        for i in range(num_samples):
            dist = distributions[i % len(distributions)]
            penalty = penalties[(i // len(distributions)) % len(penalties)] if dist == 'length-penalty' else 1.0
            samples.append(language.sample(sequence_length, distribution=dist, seed=seeds[i], penalty=penalty))
    else:
        for i in range(num_samples):
            samples.append(language.sample(sequence_length, seed=seeds[i]))
    
    return samples, language.tokenizer

def split_into_batches(token_sequences, batch_size=1024):
    """Split tokenized sequences into batches."""
    batches = []
    current_batch = []
    current_size = 0
    
    for seq in token_sequences:
        if current_size + len(seq) > batch_size:
            batches.append(current_batch)
            current_batch = [seq]
            current_size = len(seq)
        else:
            current_batch.append(seq)
            current_size += len(seq)
    
    if current_batch:
        batches.append(current_batch)
    
    return batches

def save_as_jsonl(batches, output_file):
    """Save batches in jsonl format, with each line containing a chunk index and tokens."""
    with open(output_file, 'w') as f:
        for i, batch in enumerate(batches):
            # Flatten the batch into a single token sequence
            tokens = [token for sequence in batch for token in sequence]
            line = json.dumps({"chunk": i, "tokens": tokens})
            f.write(line + '\n')
    return i + 1  # Return the number of chunks saved

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("language", type=str, help="Type of language to sample from, Dyck or ShuffleDyck")
    parser.add_argument("max_depth", type=int, help="Maximal depth of sequences")
    parser.add_argument("vocab_size", type=int, default=64, help="Vocabulary size")
    parser.add_argument("num_samples", type=int, help="Number of samples to generate")
    parser.add_argument("sequence_length", type=int, help="Length of the sequences to generate")
    parser.add_argument("output_file", type=str, help="Output file to save the samples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    random.seed(args.seed)
    
    # Generate unique opening and closing characters
    opening, closing = generate_unique_chars(args.vocab_size)
    print(f"Generated opening brackets: '{opening}'")
    print(f"Generated closing brackets: '{closing}'")
    
    # Select the language class
    if args.language.lower() == "dyck":
        language_class = Dyck
    elif args.language.lower() == "shuffledyck":
        language_class = ShuffleDyck
    else:
        raise ValueError(f"Unsupported language: {args.language}. Choose either 'Dyck' or 'ShuffleDyck'.")
    
    # Generate samples
    print(f"Generating {args.num_samples} samples of {args.sequence_length} length...")
    samples, tokenizer = generate_diverse_samples(
        language_class, 
        opening, 
        closing, 
        args.max_depth,
        args.num_samples, 
        args.sequence_length,
        args.seed
    )
    
    # Tokenize samples
    tokenized_samples = [tokenizer.encode(sample) for sample in samples]
    
    # Split into batches
    batches = split_into_batches(tokenized_samples, 1024)
    print(f"Split samples into {len(batches)} batches")
    
    # Save to output file in jsonl format
    num_chunks = save_as_jsonl(batches, args.output_file)
    print(f"Successfully saved {num_chunks} chunks to {args.output_file}")
    
    # Also save metadata to a separate JSON file
    metadata_file = args.output_file.rsplit('.', 1)[0] + '_metadata.json'
    metadata = {
        "language": args.language,
        "opening": opening,
        "closing": closing,
        "max_depth": args.max_depth,
        "vocab_size": args.vocab_size,
        "sequence_length": args.sequence_length,
        "num_samples": args.num_samples,
        "num_chunks": num_chunks,
        "bos_token": tokenizer.bos_token,
        "eos_token": tokenizer.eos_token,
        "char_to_int": tokenizer.char_to_int
    }
    
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Successfully saved metadata to {metadata_file}")