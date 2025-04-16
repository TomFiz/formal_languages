import random
from typing import List, Dict, Optional

class Language:
    """Base class for formal languages."""
    
    def sample(self, length: int, seed: Optional[int]) -> str:
        raise NotImplementedError

    def enumerate(self, length: int) -> List[str]:
        raise NotImplementedError

    def is_valid(self, sequence: str) -> bool:
        raise NotImplementedError

class Tokenizer:
    """Utility class for encoding and decoding sequences."""
    
    def __init__(self, char_to_int: Dict[str, int], bos_token: int, eos_token: int):
        self.char_to_int = char_to_int
        self.int_to_char = {v: k for k, v in char_to_int.items()}
        self.bos_token = bos_token
        self.eos_token = eos_token

    def encode(self, sequence: str) -> List[int]:
        """
        Encode a character sequence into token IDs.
        
        Args:
            sequence: String to encode
            
        Returns:
            List of token IDs including BOS and EOS tokens
        """
        return [self.bos_token] + [self.char_to_int[c] for c in sequence] + [self.eos_token]

    def decode(self, tokens: List[int]) -> str:
        """
        Decode a list of token IDs back to a character sequence.
        
        Args:
            tokens: List of token IDs
            
        Returns:
            The decoded string (excluding BOS and EOS tokens)
        """
        return ''.join(self.int_to_char[t] for t in tokens[1:-1])

class Dyck(Language):
    """Implementation of the Dyck language of balanced parentheses."""
    
    def __init__(self, opening: str, closing: str, max_depth: int, p: float = 0.5):
        """      
        Args:
            opening: String of opening bracket characters
            closing: String of closing bracket characters (in matching order with opening)
            max_depth: Maximum nesting depth allowed
            p: Probability of generating an opening bracket during sampling
        """
        self.max_depth = max_depth
        self.opening = opening
        self.closing = closing
        self.p = p
        self.char_to_int = {c: i for i, c in enumerate(opening + closing)}
        self.tokenizer = Tokenizer(self.char_to_int, len(self.char_to_int), len(self.char_to_int) + 1)

    def sample(self, length: int, seed: Optional[int]) -> str:
        """
        Sample a valid Dyck word of the specified length.
        
        Args:
            length: The length of the sequence to generate
            seed: Random seed for reproducibility
            
        Returns:
            A valid Dyck word
        """
        sequence = []
        stack = []
        random.seed(seed)

        while len(sequence) < length:
            if (stack and random.random() > self.p) or len(stack) >= min(self.max_depth, length - len(sequence)):
                sequence.append(self.closing[stack.pop()])
            else:
                opening_index = random.randint(0, len(self.opening) - 1)
                opening_char = self.opening[opening_index]
                sequence.append(opening_char)
                stack.append(opening_index)
        return ''.join(sequence)

    def enumerate(self, length: int) -> List[str]:
        """
        Enumerate all valid Dyck words of the given length.
        
        Args:
            length: The length of the sequences to enumerate
            
        Returns:
            List of all valid Dyck words with the specified length
        """
        results = []
        if length % 2 != 0:
            return results
        self._enumerate_helper(length, [], [], results)
        return results

    def _enumerate_helper(self, remaining_length: int, stack: List[int], 
                          current: List[str], results: List[str]) -> None:
        """
        Recursive helper for enumeration of Dyck words.
        
        Args:
            remaining_length: Remaining positions to fill
            stack: Current stack of open brackets (indices)
            current: Current partial sequence
            results: List to collect valid sequences
        """
        if remaining_length == 0:
            if not stack:  # Stack must be empty for valid Dyck word
                results.append(''.join(current))
            return
        
        # Option 1: Add any opening bracket if we have room for future closing brackets
        if len(stack) < min(remaining_length, self.max_depth):
            for i, opening_char in enumerate(self.opening):
                self._enumerate_helper(remaining_length - 1, stack + [i], current + [opening_char], results)
        
        # Option 2: Close the most recently opened bracket
        if stack:
            last_opening_idx = stack.pop()
            closing_char = self.closing[last_opening_idx]
            self._enumerate_helper(remaining_length - 1, stack, current + [closing_char], results)

    def is_valid(self, sequence: str) -> bool:
        """
        Check if a sequence is a valid Dyck word.
        
        Args:
            sequence: The string to check
            
        Returns:
            True if the sequence is a valid Dyck word, False otherwise
        """
        stack = []
        for char in sequence:
            if char in self.opening:
                stack.append(char)
            elif char in self.closing:
                if not stack or self.opening.index(stack.pop()) != self.closing.index(char):
                    return False
        return not stack

class ShuffleDyck(Dyck):
    """
    Implementation of the Shuffle-Dyck language, where closing brackets
    can match any previous opening bracket of the same type.
    """
    
    def sample(self, length: int, seed: Optional[int]) -> str:
        """
        Sample a valid Shuffle-Dyck word of the specified length.
        
        Args:
            length: The length of the sequence to generate
            seed: Random seed for reproducibility
            
        Returns:
            A valid Shuffle-Dyck word
        """
        sequence = []
        stack = []
        random.seed(seed)

        while len(sequence) < length:
            if stack and random.random() > self.p or len(stack) >= min(self.max_depth, length - len(sequence)):
                closing_index = stack.pop(random.randint(0, len(stack) - 1))
                closing_char = self.closing[closing_index]
                sequence.append(closing_char)
            else:
                opening_index = random.randint(0, len(self.opening) - 1)
                opening_char = self.opening[opening_index]
                sequence.append(opening_char)
                stack.append(opening_index)
        return ''.join(sequence)

    def _enumerate_helper(self, remaining_length: int, stack: List[int], 
                          current: List[str], results: List[str]) -> None:
        """
        Recursive helper for enumeration of Shuffle-Dyck words.
        
        Args:
            remaining_length: Remaining positions to fill
            stack: Current stack of open brackets (indices)
            current: Current partial sequence
            results: List to collect valid sequences
        """
        if remaining_length == 0:
            if not stack:  # Stack must be empty for valid Shuffle-Dyck word
                results.append(''.join(current))
            return
        
        # Option 1: Add any opening bracket if we have room for future closing brackets
        if len(stack) < min(remaining_length, self.max_depth):
            for i, opening_char in enumerate(self.opening):
                self._enumerate_helper(remaining_length - 1, stack + [i], current + [opening_char], results)
        
        # Option 2: Close a random opened bracket
        if stack:
            for i in range(len(list(set(stack)))):
                closing_index = stack[i]
                closing_char = self.closing[closing_index]
                self._enumerate_helper(remaining_length - 1, stack[:i] + stack[i+1:], current + [closing_char], results)

    def is_valid(self, sequence: str) -> bool:
        """
        Check if a sequence is a valid Shuffle-Dyck word.
        
        Args:
            sequence: The string to check
            
        Returns:
            True if the sequence is a valid Shuffle-Dyck word, False otherwise
        """
        counts = {char: 0 for char in self.opening + self.closing}
        for char in sequence:
            if char in self.opening:
                counts[char] += 1
            elif char in self.closing:
                if counts[self.opening[self.closing.index(char)]] == 0:
                    return False
                counts[self.opening[self.closing.index(char)]] -= 1
        return all(counts[char] == 0 for char in self.opening)


# Example usage:
# dyck = Dyck('([', ')]', 5)
# print("Dyck Language Sample:", dyck.sample(10, 42))
# print("Dyck Language Enumerate:", dyck.enumerate(4))
# print("Dyck Language Valid:", dyck.is_valid("[([])]"))

# shuffleDyck = ShuffleDyck('([', ')]', 5)
# print("Shuffle Dyck Language Sample:", shuffleDyck.sample(10, 42))
# print("Shuffle Dyck Language Enumerate:", shuffleDyck.enumerate(4))
# print("Shuffle Dyck Language Valid:", shuffleDyck.is_valid("[(][)]"))

# print("S(3,2) =", S(3, 2))