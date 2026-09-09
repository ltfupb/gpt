from datasets import load_dataset
from collections import Counter
import json

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

def save_tokenizer():
    vocabfile = open("vocab.json", "w", encoding="utf-8")
    json.dump(token_to_id, vocabfile, ensure_ascii=False, indent=2)
    vocabfile.close()

    merge_list = []
    mergefile = open("merges.json", "w", encoding="utf-8")
    for pair, new_id in merges.items():
        merge_list.append([pair[0], pair[1], new_id])
    json.dump(merge_list, mergefile, ensure_ascii=False, indent=2)
    mergefile.close()

def load_tokenizer():
    vocabfile = open("vocab.json", "r", encoding="utf-8")
    token_to_id = json.load(vocabfile)
    vocabfile.close()

    id_to_token = {}

    for token, token_id in token_to_id.items():
        id_to_token[token_id] = token

    mergefile = open("merges.json", "r", encoding="utf-8")
    merge_list = json.load(mergefile)
    mergefile.close()

    merges = {}

    for item in merge_list:
        merges[(item[0], item[1])] = item[2]

    return token_to_id, id_to_token, merges


vocab_size = 1300
merges = {}

while len(token_to_id) < vocab_size:
    pair_counts = Counter(zip(tokens, tokens[1:]))

    most_common_pair = max(pair_counts, key=pair_counts.get)

    new_token = id_to_token[most_common_pair[0]] + id_to_token[most_common_pair[1]]
    new_id = len(id_to_token)

    merges[most_common_pair] = new_id
    token_to_id[new_token] = new_id
    id_to_token[new_id] = new_token

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

pair_positions = {}

i = 0

while i < len(tokens) - 1:
    pair = (tokens[i], tokens[i+1])
    if pair not in pair_positions:
        pair_positions[pair] = [i]
    else:
        pair_positions[pair].append(i)
    i += 1

prev = []
next = []

for i in range(len(tokens)):
    if i == 0:
        prev.append(-1)
    else:
        prev.append(i - 1)

    if i == len(tokens) - 1:
        next.append(-1)
    else:
        next.append(i + 1)

left = 1
right = next[left]
after = next[right]

tokens[left] = new_id
next[left] = after

if after != -1:
    prev[after] = left
