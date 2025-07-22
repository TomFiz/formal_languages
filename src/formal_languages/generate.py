import argparse
import random
import json

from .languages import Dyck, ShuffleDyck

def catalan_number(n):
    """Calculate the nth Catalan number (for Dyck words of length 2n)."""
    if n == 0:
        return 1
    catalan = 1
    for i in range(n):
        catalan = catalan * (2 * (2 * i + 1)) // (i + 2)
    return catalan

def get_vocab(vocab_file, n=64):
    """Generate n unique characters for opening brackets and n for closing brackets."""
    vocab = json.load(open(vocab_file, 'r'))
    all_tokens = list(vocab.keys())
    
    # Ensure we have enough unique characters
    if len(all_tokens) < 2 * n:
        raise ValueError(f"Cannot generate {n} unique characters for both opening and closing brackets. Maximum is {len(all_tokens) // 2}.")
    
    opening = all_tokens[3:n+3]
    closing = all_tokens[n+3:2*n+3]
    
    return opening, closing

def generate_diverse_samples(language_class, opening, closing, bias, max_depth, num_samples, sequence_length, impose_length_closing, seed=42):
    """Generate diverse samples from the language."""
    language = language_class(opening, closing, max_depth, p=0.5, bias=bias)
    language.tokenizer.char_to_int = json.load(open("vocab.json", 'r'))
    samples = []
    seeds = list(range(seed, seed + num_samples))

    probs = [catalan_number(i) for i in [1,2,4,8,16,32]]
    length = 2*random.choices([1, 2, 4, 8, 16, 32], weights=probs/sum(probs))[0]

    # For ShuffleDyck, use different distribution strategies to increase diversity
    if language_class == ShuffleDyck:
        distributions = [
                         'type-uniform', 
                        #  'bracket-uniform', 
                        #  'length-penalty',
                         ]
        penalties = [0.5, 1.0, 2.0]
        
        for i in range(num_samples):
            dist = distributions[i % len(distributions)]
            penalty = penalties[(i // len(distributions)) % len(penalties)] if dist == 'length-penalty' else 1.0
            samples.append(language.sample(length, impose_length_closing=impose_length_closing, 
                                           distribution=dist, seed=seeds[i], penalty=penalty))
    else:
        for i in range(num_samples):
            samples.append(language.sample(length, seed=seeds[i]))

    return samples, language.tokenizer

def split_into_batches(token_sequences, batch_size=128):
    """
    Split tokenized sequences into batches of exactly batch_size tokens.
    Sequences can be cut in the middle if necessary.
    """
    batches = []
    
    # Flatten all sequences into one long sequence with separators
    all_tokens = []
    for seq in token_sequences:
        all_tokens.extend(seq)
    
    # Split into batches of exactly batch_size
    for i in range(0, len(all_tokens), batch_size):
        batch = all_tokens[i:i + batch_size]
        if len(batch) == batch_size:  # Only keep full batches
            batches.append(batch)
    
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
    parser.add_argument("--language", type=str, default='Dyck', help="Type of language to sample from, Dyck or ShuffleDyck")
    parser.add_argument("--max_depth", type=int, default=5, help="Maximal depth of sequences")
    parser.add_argument("--impose_length_closing", action='store_true', help="Impose length closing for Dyck language")
    parser.add_argument("--vocab_size", type=int, default=64, help="Vocabulary size")
    parser.add_argument("--bias_a", type=float, default=0, help="Zipf parameters for biases : a (NL = 1)")
    parser.add_argument("--bias_b", type=float, default=0, help="Zipf parameters for biases : b (NL = 2.7)")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples to generate")
    parser.add_argument("--sequence_length", type=int, default=10, help="Length of the sequences to generate")
    parser.add_argument("--output_file", type=str, default='output.jsonl', help="Output file to save the samples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    random.seed(args.seed)
    
    # Generate unique opening and closing characters
    opening, closing = get_vocab("./vocab.json",args.vocab_size)
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
    bias ={"a": args.bias_a, "b": args.bias_b}
    samples, tokenizer = generate_diverse_samples(
        language_class, 
        opening, 
        closing,
        bias,
        args.max_depth,
        args.num_samples, 
        args.sequence_length,
        args.impose_length_closing,
        args.seed
    )
    
    # Tokenize samples
    tokenized_samples = [tokenizer.encode(sample) for sample in samples]
    
    # Split into batches
    batches = split_into_batches(tokenized_samples, 128)
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
        "bias": bias,
        "impose_length_closing": args.impose_length_closing,
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
