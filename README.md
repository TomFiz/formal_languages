# Formal Languages

A Python library for working with formal languages, particularly Dyck languages and their variants.

## Overview

This repository provides implementations of formal language classes, with a focus on:

- **Dyck Language**: The language of balanced parentheses, where each closing bracket must match the most recently opened bracket of the corresponding type.
- **Shuffle-Dyck Language**: A variant where closing brackets can match any previous opening bracket of the same type.


## Features

- Generate random valid samples from languages
- Enumerate all valid strings of a given length
- Validate whether a string belongs to a language
- Encode/decode between character strings and token IDs

## Setup

 Clone this repository:

```bash
git clone https://github.com/TomFiz/formal_languages.git
cd formal_languages
```

## Usage

### Basic Usage

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
- pyproject.toml - Project metadata and dependencies