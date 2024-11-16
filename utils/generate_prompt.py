#!/usr/bin/env python3

import tiktoken
import random
import string
import sys

def generate_random_prompt(token_count, model_encoding="o200k_base"):
    enc = tiktoken.get_encoding(model_encoding)

    # Generate random characters (adjust as necessary to suit token structure)
    random_text = ''.join(random.choices(string.ascii_letters + string.digits + ' ', k=10000))

    # Tokenize the random text
    tokens = enc.encode(random_text)

    # Trim or extend the tokenized text to exactly `token_count` tokens
    if len(tokens) > token_count:
        tokens = tokens[:token_count]
    else:
        while len(tokens) < token_count:
            random_text += ''.join(random.choices(string.ascii_letters + string.digits + ' ', k=100))
            tokens = enc.encode(random_text)

    # Decode back to a string (if you need the raw text)
    final_prompt = enc.decode(tokens)

    return final_prompt

if __name__ == "__main__":
    # Accept token count as a command-line argument
    token_count = int(sys.argv[1])

    # Generate and print the random prompt with the specified number of tokens
    prompt = generate_random_prompt(token_count)
    print(prompt)