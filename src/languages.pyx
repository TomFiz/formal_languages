import random
import numpy as np
import numpy.random as rd
from libc.stdlib cimport rand, srand, RAND_MAX
from cpython.list cimport PyList_Append, PyList_GET_SIZE, PyList_GET_ITEM, PyList_New
from typing import List, Dict, Optional, Set

cdef double c_random():
    """Fast C implementation of random number generation."""
    return rand() / <double>RAND_MAX

cdef class Language:
    """Base class for formal languages."""
    
    def sample(self, int length, seed: Optional[int]) -> str:
        """
        Generate a random sample from the language.
        
        Args:
            length: The length of the sequence to generate
            seed: Random seed for reproducibility
            
        Returns:
            A string representing a valid sentence in the language
        """
        raise NotImplementedError

    def enumerate(self, int length) -> list:
        """
        Enumerate all valid strings of the given length in the language.
        
        Args:
            length: The length of the sequences to enumerate
            
        Returns:
            List of all valid strings with the specified length
        """
        raise NotImplementedError

    def is_valid(self, str sequence) -> bool:
        """
        Check if the given sequence is valid in the language.
        
        Args:
            sequence: The string to check
            
        Returns:
            True if the sequence is valid, False otherwise
        """
        raise NotImplementedError

class Tokenizer:
    """Utility class for encoding and decoding sequences."""
    
    def __init__(self, Dict[str, int] char_to_int, int bos_token, int eos_token):
        """
        Initialize the tokenizer.
        
        Args:
            char_to_int: Mapping from characters to integer tokens
            bos_token: Beginning of sequence token
            eos_token: End of sequence token
        """
        self.char_to_int = char_to_int
        self.int_to_char = {v: k for k, v in char_to_int.items()}
        self.bos_token = bos_token
        self.eos_token = eos_token

    def encode(self, str sequence) -> List[int]:
        """
        Encode a character sequence into token IDs.
        
        Args:
            sequence: String to encode
            
        Returns:
            List of token IDs including BOS and EOS tokens
        """
        cdef list result = [self.bos_token]
        cdef str c
        for c in sequence:
            result.append(self.char_to_int[c])
        result.append(self.eos_token)
        return result

    def decode(self, list tokens) -> str:
        """
        Decode a list of token IDs back to a character sequence.
        
        Args:
            tokens: List of token IDs
            
        Returns:
            The decoded string (excluding BOS and EOS tokens)
        """
        cdef int t
        cdef list chars = []
        for t in tokens[1:-1]:
            chars.append(self.int_to_char[t])
        return ''.join(chars)

cdef class Dyck(Language):
    """Implementation of the Dyck language of balanced parentheses."""
    
    cdef public int max_depth
    cdef public str opening
    cdef public str closing
    cdef public double p
    cdef public dict char_to_int
    cdef public object tokenizer
    
    def __init__(self, str opening, str closing, int max_depth, double p=0.5):
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

        assert self.max_depth > 0, "Max depth must be positive"
        assert self.p >= 0 and self.p <= 1, "Probability p must be between 0 and 1"
        assert len(self.opening) == len(self.closing), "Opening and closing brackets must match in length"
        assert len(self.opening) == len(set(self.opening)), "Opening brackets must be unique"
        assert len(self.closing) == len(set(self.closing)), "Closing brackets must be unique"

    def sample(self, int length, seed: Optional[int]) -> str:
        """
        Sample a valid Dyck word of the specified length.
        
        Args:
            length: The length of the sequence to generate
            seed: Random seed for reproducibility
            
        Returns:
            A valid Dyck word
        """
        assert length % 2 == 0, "Length must be even for Dyck words"
        
        cdef list sequence = []
        cdef list stack = []
        cdef int opening_index
        cdef str opening_char
        
        if seed is not None:
            rd.seed(seed)
        
        while len(sequence) < length:
            if (stack and rd.random() > self.p) or len(stack) >= min(self.max_depth, length - len(sequence)):
                sequence.append(self.closing[stack.pop()])
            else:
                opening_index = rd.randint(0, len(self.opening))
                opening_char = self.opening[opening_index]
                sequence.append(opening_char)
                stack.append(opening_index)
        return ''.join(sequence)

    def enumerate(self, int length) -> List[str]:
        """
        Enumerate all valid Dyck words of the given length.
        
        Args:
            length: The length of the sequences to enumerate
            
        Returns:
            List of all valid Dyck words with the specified length
        """
        cdef list results = []
        if length % 2 != 0:
            return results
        self._enumerate_helper(length, [], [], results)
        return results

    def _enumerate_helper(self, int remaining_length, list stack, 
                          list current, list results):
        """
        Recursive helper for enumeration of Dyck words.
        
        Args:
            remaining_length: Remaining positions to fill
            stack: Current stack of open brackets (indices)
            current: Current partial sequence
            results: List to collect valid sequences
        """
        cdef int i, last_opening_idx
        cdef str opening_char, closing_char
        
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
            stack_copy = stack.copy()
            last_opening_idx = stack_copy.pop()
            closing_char = self.closing[last_opening_idx]
            self._enumerate_helper(remaining_length - 1, stack_copy, current + [closing_char], results)

    def is_valid(self, str sequence) -> bool:
        """
        Check if a sequence is a valid Dyck word.
        
        Args:
            sequence: The string to check
            
        Returns:
            True if the sequence is a valid Dyck word, False otherwise
        """
        cdef list stack = []
        cdef str char
        
        for char in sequence:
            if char in self.opening:
                stack.append(char)
            elif char in self.closing:
                if not stack or self.opening.index(stack.pop()) != self.closing.index(char):
                    return False
        return not stack

cdef class ShuffleDyck(Dyck):
    """
    Implementation of the Shuffle-Dyck language, where closing brackets
    can match any previous opening bracket of the same type.
    """
    
    def sample(self, int length, str distribution="type-uniform", seed: Optional[int]=42, 
               double penalty=1.0, bint verbose=False) -> str:
        """
        Sample a valid Shuffle-Dyck word of the specified length.
        
        Args:
            length: The length of the sequence to generate
            distribution: The distribution of closing brackets
             - type-uniform : pick a random type of opened bracket to close according to a uniform distribution, then pop the first occurrence of that type
             - bracket-uniform : pick a random bracket to close according to a uniform distribution, then pop that bracket
             - length_penalty : pick a random type of opened bracket to close according to a length penalty distribution, then pop the first occurrence of that type
            penalty: Penalty for closing brackets based on their length : 0 means no penalty, +inf means the longest opened bracket is always closed first
            seed: Random seed for reproducibility
            verbose: If True, print additional information during sampling
            
        Returns:
            A valid Shuffle-Dyck word
        """
        cdef list sequence = []
        cdef list stack = []
        cdef dict stack_pointers = {key: [] for key in range(len(self.opening))}
        cdef int closing_bracket_type, closing_index, opening_index, closing_stack_id
        cdef str opening_char, closing_char
        cdef list opened_bracket_lengths, probabilities
        
        if seed is not None:
            rd.seed(seed)

        while len(sequence) < length:
            if stack and rd.random() > self.p or len(stack) >= min(self.max_depth, length - len(sequence)):
                if distribution == 'type-uniform':
                    # Randomly select a type of opened bracket to close
                    closing_bracket_type = list(set(stack))[rd.randint(0, len(set(stack)))]
                    # Pop the first occurrence of that type
                    closing_index = stack.pop(stack.index(closing_bracket_type))
                elif distribution == 'bracket-uniform':
                    closing_index = stack.pop(rd.randint(0, len(stack)))
                elif distribution == 'length-penalty':
                    # Get the length of the actually opened brackets for each type
                    opened_bracket_lengths = [len(sequence) - stack_pointers[i][-1] for i in stack]
                    probabilities = [np.tanh(penalty*length + 1e-3) for length in opened_bracket_lengths]
                    probabilities = np.array(probabilities) / np.sum(probabilities)
                    closing_stack_id = rd.choice(len(stack), p=probabilities)
                    closing_index = stack.pop(closing_stack_id)
                
                stack_pointers[closing_index].pop()
                closing_char = self.closing[closing_index]
                sequence.append(closing_char)
            else:
                opening_index = rd.randint(0, len(self.opening))
                opening_char = self.opening[opening_index]
                sequence.append(opening_char)
                stack.append(opening_index)
                # Store the index of the opening bracket in the sequence
                stack_pointers[opening_index].append(len(sequence) - 1)
        return ''.join(sequence)

    def _enumerate_helper(self, int remaining_length, list stack, 
                          list current, list results):
        """
        Recursive helper for enumeration of Shuffle-Dyck words.
        
        Args:
            remaining_length: Remaining positions to fill
            stack: Current stack of open brackets (indices)
            current: Current partial sequence
            results: List to collect valid sequences
        """
        cdef int i, closing_index
        cdef str opening_char, closing_char
        cdef list unique_stack_items
        
        if remaining_length == 0:
            if not stack:  # Stack must be empty for valid Shuffle-Dyck word
                results.append(''.join(current))
            return
        
        # Option 1: Add any opening bracket if we have room for future closing brackets
        if len(stack) < min(remaining_length, self.max_depth):
            for i, opening_char in enumerate(self.opening):
                self._enumerate_helper(remaining_length - 1, stack + [i], current + [opening_char], results)
        
        # Option 2: Close opened brackets
        if stack:
            unique_stack_items = list(set(stack))
            for i in range(len(unique_stack_items)):
                closing_index = unique_stack_items[i]
                new_stack = stack.copy()
                new_stack.remove(closing_index)
                closing_char = self.closing[closing_index]
                self._enumerate_helper(remaining_length - 1, new_stack, current + [closing_char], results)

    def is_valid(self, str sequence) -> bool:
        """
        Check if a sequence is a valid Shuffle-Dyck word.
        
        Args:
            sequence: The string to check
            
        Returns:
            True if the sequence is a valid Shuffle-Dyck word, False otherwise
        """
        cdef dict counts = {char: 0 for char in self.opening + self.closing}
        cdef str char
        cdef int idx
        
        for char in sequence:
            if char in self.opening:
                counts[char] += 1
            elif char in self.closing:
                idx = self.closing.index(char)
                if counts[self.opening[idx]] == 0:
                    return False
                counts[self.opening[idx]] -= 1
        
        return all(counts[char] == 0 for char in self.opening)