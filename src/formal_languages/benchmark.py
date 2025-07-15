import json
import random
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
import numpy as np

from .languages import Dyck, ShuffleDyck, Language


class FormalLanguageBenchmark:
    """Generator for BLIMP-style benchmarks for formal languages."""
    
    def __init__(self, language: Language, metadata: Dict[str, Any]):
        """
        Initialize the benchmark generator.
        
        Args:
            language: The formal language instance (Dyck or ShuffleDyck)
            metadata: Dataset metadata containing training configuration
        """
        self.language = language
        self.metadata = metadata
        self.perturbations = []
        
        # Register perturbations based on language type
        if isinstance(language, (Dyck, ShuffleDyck)):
            self.perturbations.extend([
                "close_wrong_bracket",
                "close_unopened_bracket"
            ])
    
    def generate_close_wrong_bracket_pair(self, base_sequence: str) -> Optional[Tuple[str, str]]:
        """
        Generate a pair where the bad sentence closes a bracket that was opened but not the last one.
        Only applies to standard Dyck language (not ShuffleDyck).
        """
        if isinstance(self.language, ShuffleDyck):
            return None  # This perturbation doesn't apply to ShuffleDyck
            
        if not self.language.is_valid(base_sequence):
            return None
            
        # Find positions where we can make this perturbation
        stack = []
        positions = []
        
        for i, char in enumerate(base_sequence):
            if char in self.language.opening:
                stack.append((char, i))
            elif char in self.language.closing:
                if stack:
                    opened_char, open_pos = stack.pop()
                    # If there were other brackets opened before this one was closed,
                    # we could have closed one of those instead
                    if len(stack) > 0:
                        positions.append((i, opened_char, [item[0] for item in stack]))
        
        if not positions:
            return None
            
        # Pick a random position to make the perturbation
        pos, correct_char, available_chars = random.choice(positions)
        wrong_char = random.choice(available_chars)
        wrong_closing = self.language.closing[self.language.opening.index(wrong_char)]
        
        # Create the bad sequence
        bad_sequence = list(base_sequence)
        bad_sequence[pos] = wrong_closing
        
        return base_sequence, ''.join(bad_sequence)
    
    def generate_close_unopened_bracket_pair(self, base_sequence: str) -> Optional[Tuple[str, str]]:
        """
        Generate a pair where the bad sentence closes a bracket that wasn't opened yet.
        """
        if not self.language.is_valid(base_sequence):
            return None
            
        # Find a position where we can insert a closing bracket that violates the constraint
        opened_types = set()
        positions = []
        
        for i, char in enumerate(base_sequence):
            if char in self.language.opening:
                opened_types.add(char)
            elif char in self.language.closing:
                opened_type = self.language.opening[self.language.closing.index(char)]
                if opened_type in opened_types:
                    # Before this closing bracket, we could insert a closing bracket
                    # for a type that hasn't been opened yet
                    unopened_types = set(self.language.opening) - opened_types
                    if unopened_types:
                        positions.append((i, list(unopened_types)))
        
        if not positions:
            # Try inserting at the beginning
            if len(self.language.opening) > 0:
                unopened_closing = self.language.closing[0]
                bad_sequence = unopened_closing + base_sequence
                return base_sequence, bad_sequence
            return None
        
        # Pick a random position and unopened type
        pos, unopened_types = random.choice(positions)
        unopened_type = random.choice(unopened_types)
        unopened_closing = self.language.closing[self.language.opening.index(unopened_type)]
        
        # Insert the unopened closing bracket
        bad_sequence = base_sequence[:pos] + unopened_closing + base_sequence[pos:]
        
        return base_sequence, bad_sequence

    def generate_pair(self, base_sequence: str, perturbation: str) -> Optional[Tuple[str, str]]:
        """Generate a good/bad pair for a specific perturbation."""
        if perturbation == "close_wrong_bracket":
            return self.generate_close_wrong_bracket_pair(base_sequence)
        elif perturbation == "close_unopened_bracket":
            return self.generate_close_unopened_bracket_pair(base_sequence)
        else:
            return None
    
    def generate_benchmark(self, 
                          num_samples_per_perturbation: int = 100,
                          length_range: Tuple[int, int] = (4, 20),
                          seed: Optional[int] = 42) -> List[Dict[str, Any]]:
        """
        Generate the complete benchmark dataset.
        
        Args:
            num_samples_per_perturbation: Number of pairs to generate per perturbation type
            length_range: Range of sequence lengths to generate
            seed: Random seed for reproducibility
            
        Returns:
            List of benchmark items in BLIMP format
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        benchmark_data = []
        pair_id = 0
        
        all_perturbations = self.perturbations
        
        for perturbation in all_perturbations:
            generated_pairs = 0
            attempts = 0
            max_attempts = num_samples_per_perturbation * 10
            
            while generated_pairs < num_samples_per_perturbation and attempts < max_attempts:
                attempts += 1
                
                # Generate a base valid sequence
                length = random.randrange(length_range[0], length_range[1] + 1, 2)  # Even lengths only
                try:
                    base_sequence = self.language.sample(length=length, seed=None)
                    
                    if not self.language.is_valid(base_sequence):
                        continue
                    
                    # Generate the perturbation pair
                    pair = self.generate_pair(base_sequence, perturbation)
                    
                    if pair is None:
                        continue
                    
                    good_sequence, bad_sequence = pair
                    
                    # Verify the pair is valid (good is valid, bad is invalid)
                    if (self.language.is_valid(good_sequence) and 
                        not self.language.is_valid(bad_sequence)):
                        
                        # Create benchmark item
                        item = {
                            "sentence_good": good_sequence,
                            "sentence_bad": bad_sequence,
                            "field": "formal_languages",
                            "linguistics_term": perturbation,
                            "UID": f"{type(self.language).__name__.lower()}_{perturbation}",
                            "simple_LM_method": True,
                            "one_prefix_method": True,
                            "two_prefix_method": False,
                            "lexically_identical": False,  # Sequences differ by one token
                            "pair_id": pair_id,
                            "language_type": type(self.language).__name__,
                            "sequence_length": len(good_sequence)
                        }
                        
                        benchmark_data.append(item)
                        generated_pairs += 1
                        pair_id += 1
                        
                except Exception as e:
                    continue
        
        return benchmark_data
    
    def save_benchmark(self, 
                      output_path: Path,
                      num_samples_per_perturbation: int = 100,
                      length_range: Tuple[int, int] = (4, 20),
                      seed: Optional[int] = 42) -> None:
        """
        Generate and save the benchmark to a JSONL file.
        
        Args:
            output_path: Path to save the benchmark file
            num_samples_per_perturbation: Number of pairs per perturbation
            length_range: Range of sequence lengths
            seed: Random seed
        """
        benchmark_data = self.generate_benchmark(
            num_samples_per_perturbation=num_samples_per_perturbation,
            length_range=length_range,
            seed=seed
        )
        
        with open(output_path, 'w') as f:
            for item in benchmark_data:
                f.write(json.dumps(item) + '\n')
        
        print(f"Generated {len(benchmark_data)} benchmark pairs")
        print(f"Saved to {output_path}")


def create_benchmark_from_metadata(metadata_path: Path, 
                                  output_dir: Path,
                                  num_samples_per_perturbation: int = 100) -> None:
    """
    Create benchmarks from a metadata file describing the training dataset.
    
    Args:
        metadata_path: Path to the metadata JSON file
        output_dir: Directory to save benchmark files
        num_samples_per_perturbation: Number of pairs per perturbation type
    """
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract language configuration from metadata
    lang_type = metadata.get('language', 'Dyck')
    opening = metadata.get('opening', '([{')
    closing = metadata.get('closing', ')]}')
    max_depth = metadata.get('max_depth', 10)
    bias = metadata.get('bias', None)
    sequence_length = metadata.get('sequence_length', 128)
    
    # Create language instance based on metadata
    if lang_type.lower() == 'dyck':
        language = Dyck(
            opening=opening,
            closing=closing,
            max_depth=max_depth,
            bias=bias
        )
    elif lang_type.lower() == 'shuffledyck':
        language = ShuffleDyck(
            opening=opening,
            closing=closing,
            max_depth=max_depth,
            bias=bias
        )
    else:
        raise ValueError(f"Unsupported language type: {lang_type}")
    
    # Generate benchmark
    benchmark = FormalLanguageBenchmark(language, metadata)
    
    # Use fixed sequence length from metadata
    length_range = (sequence_length, sequence_length)
    
    output_path = output_dir / f"{lang_type.lower()}_benchmark.jsonl"
    benchmark.save_benchmark(
        output_path=output_path,
        num_samples_per_perturbation=num_samples_per_perturbation,
        length_range=length_range,
        seed=42
    )


if __name__ == "__main__":
    # Example usage
    from pathlib import Path
    
    create_benchmark_from_metadata(
        metadata_path=Path("metadata.json"),
        output_dir=Path("benchmarks"),
        num_samples_per_perturbation=100
    )