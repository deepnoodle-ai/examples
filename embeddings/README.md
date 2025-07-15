# OpenAI Embeddings Similarity Search Demo

This Python program demonstrates how to use OpenAI's embedding API to perform similarity search on text.

## Features

- Generates embeddings for a collection of sample texts using OpenAI's `text-embedding-3-small` model
- Implements cosine similarity to measure text similarity
- Provides both pre-defined queries and interactive search functionality
- Shows the top 3 most similar texts for each query

## Prerequisites

- Python 3.7 or higher
- An OpenAI API key

## Installation

1. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Set your OpenAI API key as an environment variable:
   ```bash
   export OPENAI_API_KEY='your-api-key-here'
   ```

## Usage

Run the program:
```bash
python main.py
```

The program will:
1. Generate embeddings for 8 sample texts
2. Run similarity searches for 4 pre-defined queries
3. Enter an interactive mode where you can input your own queries
4. Type 'quit' to exit the interactive mode

## How It Works

1. **Embeddings**: The program uses OpenAI's embedding API to convert text into high-dimensional vectors
2. **Similarity**: Cosine similarity is used to measure how similar two embeddings are (closer to 1 = more similar)
3. **Search**: For each query, the program finds the texts with the highest similarity scores

## Example Output

```
Query: 'A dog jumping'
----------------------------------------
1. A fast auburn canine leaps above a sleepy hound
   Similarity: 0.8234
2. The quick brown fox jumps over the lazy dog
   Similarity: 0.7891
3. It's a beautiful day with clear skies
   Similarity: 0.4567
``` 