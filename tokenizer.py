from datasets import load_dataset

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
tokens = list(text)


token_to_id = {}
id_to_token = {}


for i, token in enumerate(chars):
    token_to_id[token] = i

for i, token in enumerate(chars):
    id_to_token[i] = token


def encode(text):
    tokens = list(text)

    for pair in merges:
        merged_tokens = []
        i = 0

        while i < len(tokens):
            if (
                i < len(tokens) - 1
                and (tokens[i], tokens[i + 1]) == pair
            ):
                new_token = pair[0] + pair[1]
                merged_tokens.append(new_token)
                i += 2
            else:
                merged_tokens.append(tokens[i])
                i += 1

        tokens = merged_tokens

    ids = []

    for token in tokens:
        token_id = token_to_id[token]
        ids.append(token_id)

    return ids


def decode(ids):
    text = ""

    for token_id in ids:
        token = id_to_token[token_id]
        text += token
    return text


vocab_size = 1300
merges = []

while len(token_to_id) < vocab_size:
    pair_counts = {}

    for i in range(len(tokens)-1):
        pair = (tokens[i], tokens[i+1])

        if pair in pair_counts:
            pair_counts[pair] += 1
        else:
            pair_counts[pair] = 1


    most_common_pair = max(pair_counts, key=pair_counts.get)
    merges.append(most_common_pair)

    new_token = most_common_pair[0] + most_common_pair[1]
    new_id = len(id_to_token)

    token_to_id[new_token] = new_id
    id_to_token[new_id] = new_token

    merged_tokens = []

    i = 0

    while i < len(tokens):
        if (i < len(tokens) - 1 and (tokens[i], tokens[i+1]) == most_common_pair):
            merged_tokens.append(new_token)
            i += 2
        else:
            merged_tokens.append(tokens[i])
            i += 1

    tokens = merged_tokens


sample = "대한민국은"

encoded = encode(sample)
decoded = decode(encoded)

print("원문:", sample)
print("인코딩:", encoded)
print("디코딩:", decoded)
print("복원 성공:", decoded == sample)
print("문자 수:", len(sample))
print("토큰 수:", len(encoded))

print("merge 예시:", merges[:20])