import numpy as np
from datasets import Dataset


def tokenize(example, tokenizer, label2id, max_length):

    # rebuild text from tokens
    text = []
    token_map = []
    labels = []

    idx = 0

    for t, l, ws in zip(
        example["tokens"], example["provided_labels"], example["trailing_whitespace"]
    ):
        text.append(t)
        token_map.extend([idx] * len(t))
        labels.extend([l] * len(t))

        if ws:
            text.append(" ")
            labels.append("O")
            token_map.append(-1)

        idx += 1

    # actual tokenization
    tokenized = tokenizer(
        "".join(text), return_offsets_mapping=True, max_length=max_length
    )

    labels = np.array(labels)

    text = "".join(text)
    token_labels = []

    for start_idx, end_idx in tokenized.offset_mapping:
        # CLS token
        if start_idx == 0 and end_idx == 0:
            token_labels.append(label2id["O"])
            continue

        # case when token starts with whitespace
        if text[start_idx].isspace():
            start_idx += 1

        token_labels.append(label2id[labels[start_idx]])

    length = len(tokenized.input_ids)

    return {
        **tokenized,
        "labels": token_labels,
        "length": length,
        "token_map": token_map,
    }


def create_dataset(data, tokenizer, max_length, label2id):
    ds = Dataset.from_dict(
        {
            "full_text": data.full_text.tolist(),
            "document": data.document.tolist(),
            "tokens": data.tokens.tolist(),
            "trailing_whitespace": data.trailing_whitespace.tolist(),
            "provided_labels": data.labels.tolist(),
            "token_indices": data.token_indices.tolist(),
        }
    )
    ds = ds.map(
        tokenize,
        fn_kwargs={
            "tokenizer": tokenizer,
            "label2id": label2id,
            "max_length": max_length,
        },
        num_proc=3,
    )
    return ds
