import re 

def extract_vocab(structs):
    vocab = list(set(''.join(structs))) + ['<UNK>']
    token2id = {v:i for i, v in enumerate(vocab)}
    id2token = {i:v for i, v in enumerate(vocab)}
    return vocab, token2id, id2token

def encode(s, vocab, token2id):
    pattern = r'|'.join(map(re.escape, vocab))
    tokens = re.findall(pattern, s)
    encoded_tokens = []
    for token in tokens:
        if token in token2id:
            encoded_tokens.append(token2id[token])
        else:
            encoded_tokens.extend([token2id[char] for char in token if char in token2id])
    return encoded_tokens


