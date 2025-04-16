# Formal Languages

A Python/Cython library for working with formal languages, particularly Dyck languages and their variants.

## Overview

This repository provides implementations of formal language classes, with a focus on:

- **Dyck Language**: The language of balanced parentheses, where each closing bracket must match the most recently opened bracket of the corresponding type.
- **Shuffle-Dyck Language**: A variant where closing brackets can match any previous opening bracket of the same type.

The implementation is available in both pure Python and optimized Cython versions for performance-critical applications.

## Features

- Generate random valid samples from languages
- Enumerate all valid strings of a given length
- Validate whether a string belongs to a language
- Encode/decode between character strings and token IDs
- Optimized Cython implementation for performance

## Installation

### Requirements

- Python 3.13+
- Cython
- NumPy
- setuptools

### Setup

1. Clone this repository:

```bash
git clone https://github.com/TomFiz/formal_languages.git
cd formal_languages
```

2. Install dependencies:

```bash
pip install -e .
```

3. Build the Cython extension:

```bash
python setup.py build_ext --inplace
```

## Usage

### Basic Usage (Python version)

```python
from languages import Dyck, ShuffleDyck

# Create a Dyck language instance with '(' and '[' as opening brackets
# and ')' and ']' as corresponding closing brackets, with max nesting depth of 5
dyck = Dyck('([', ')]', 5)

# Generate a random sample of length 10 with random seed 42
sample = dyck.sample(10, 42)
print("Dyck Language Sample:", sample)

# Check if a string is valid in the language
print("Is '[()]' valid?", dyck.is_valid('[()]'))

# Enumerate all valid strings of length 4
all_strings = dyck.enumerate(4)
print("All valid Dyck words of length 4:", all_strings)

# Create a Shuffle-Dyck language instance
shuffle_dyck = ShuffleDyck('([', ')]', 5)
print("Is '[(])' valid in Shuffle-Dyck?", shuffle_dyck.is_valid('[(])'))
```

### Using the Cython Version (for performance)

```python
import clanguages as cl

# The API is identical to the Python version, but with much better performance
dyck = cl.Dyck('([', ')]', 5)
sample = dyck.sample(100, 42)
print("Quickly generated sample:", sample)
```

### Working with Tokenizers

```python
from languages import Dyck

dyck = Dyck('([', ')]', 5)
tokenizer = dyck.tokenizer

# Encode a string to token IDs
tokens = tokenizer.encode('[()]')
print("Encoded tokens:", tokens)

# Decode tokens back to string
decoded = tokenizer.decode(tokens)
print("Decoded string:", decoded)
```

## Project Structure

- languages.py - Pure Python implementation of language classes
- languages.pyx - Cython implementation for performance
- setup.py - Build script for the Cython extension
- pyproject.toml - Project metadata and dependencies