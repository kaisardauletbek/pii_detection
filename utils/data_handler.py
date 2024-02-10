import json
import os
from datasets import Dataset, features
import numpy as np


def do_load_data(DATA_PATH):
    data = json.load(open(DATA_PATH+"train.json"))

    # downsampling of negative examples
    p=[] # positive samples (contain relevant labels)
    n=[] # negative samples (presumably contain entities that are possibly wrongly classified as entity)
    for d in data:
        if any(np.array(d["labels"]) != "O"): p.append(d)
        else: n.append(d)
    print("original datapoints: ", len(data))

    external = json.load(open(DATA_PATH+"pii_dataset_fixed.json"))
    print("external datapoints: ", len(external))

    # moredata = json.load(open(DATA_PATH+"moredata_dataset_fixed.json"))
    # print("moredata datapoints: ", len(moredata))

    mixtral = json.load(open(DATA_PATH+"mixtral-8x7b-v1.json"))
    print("mixtral ", len(mixtral))

    data = external+mixtral+p+n[:len(n)//3] # moredata
    print("combined: ", len(data))
    return data

def do_tokenize(example, tokenizer, label2id, max_length):

    # rebuild text from tokens
    text = []
    labels = []

    for t, l, ws in zip(
        example["tokens"], example["provided_labels"], example["trailing_whitespace"]
    ):
        text.append(t)
        labels.extend([l] * len(t))

        if ws:
            text.append(" ")
            labels.append("O")

    # actual tokenization
    tokenized = tokenizer("".join(text), return_offsets_mapping=True, max_length=max_length)

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

    return {**tokenized, "labels": token_labels, "length": length}

def do_hf_dataset(tokenizer, label2id, data, TRAINING_MAX_LENGTH):
    ds = Dataset.from_dict({
    "full_text": [x["full_text"] for x in data],
    "document": [str(x["document"]) for x in data],
    "tokens": [x["tokens"] for x in data],
    "trailing_whitespace": [x["trailing_whitespace"] for x in data],
    "provided_labels": [x["labels"] for x in data],
    })
    ds = ds.map(do_tokenize, fn_kwargs={"tokenizer": tokenizer, "label2id": label2id, "max_length": TRAINING_MAX_LENGTH}, num_proc=3)
    return ds