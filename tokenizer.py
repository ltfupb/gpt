from datasets import load_dataset
from collections import Counter

ds = load_dataset("wikimedia/wikipedia", "20231101.ko", split="train", streaming=True)

max_chars = 50_000

texts = []
current_length = 0


for article in ds:
    article_text = article["text"]

    if current_length + len(article_text) > max_chars:
        break

    texts.append(article_text)
    current_length += len(article_text)

text = "\n".join(texts)


chars = sorted(set(text))


token_to_id = {}
id_to_token = {}


for i, token in enumerate(chars):
    token_to_id[token] = i

for i, token in enumerate(chars):
    id_to_token[i] = token

tokens = []

for token in text:
    token_id = token_to_id[token]
    tokens.append(token_id)

def encode(text):
    tokens = []

    for token in text:
        token_id = token_to_id[token]
        tokens.append(token_id)

    for pair, new_id in merges.items():
        merged_tokens = []
        i = 0

        while i < len(tokens):
            if (
                i < len(tokens) - 1
                and (tokens[i], tokens[i + 1]) == pair
            ):
                merged_tokens.append(new_id)
                i += 2
            else:
                merged_tokens.append(tokens[i])
                i += 1

        tokens = merged_tokens

    return tokens

def decode(ids):
    text = ""

    for token_id in ids:
        token = id_to_token[token_id]
        text += token
    return text


vocab_size = 1300
merges = {}

while len(token_to_id) < vocab_size:
    pair_counts = Counter(zip(tokens, tokens[1:]))

    most_common_pair = max(pair_counts, key=pair_counts.get)

    new_token = id_to_token[most_common_pair[0]] + id_to_token[most_common_pair[1]]
    new_id = len(id_to_token)

    token_to_id[new_token] = new_id
    id_to_token[new_id] = new_token
    merges[most_common_pair] = new_id

    merged_tokens = []

    i = 0

    while i < len(tokens):
        if (i < len(tokens) - 1 and (tokens[i], tokens[i+1]) == most_common_pair):
            merged_tokens.append(new_id)
            i += 2
        else:
            merged_tokens.append(tokens[i])
            i += 1

    tokens = merged_tokens

