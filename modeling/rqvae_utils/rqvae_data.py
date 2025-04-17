import gzip
import json
import pickle
import random

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

tqdm.pandas()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "google-t5/t5-small"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)


def parse(path):
    g = gzip.open(path, "rb")
    for line in g:
        yield eval(line)


def getDF(path):
    i = 0
    df = {}
    for d in parse(path):
        df[i] = d
        i += 1
    return pd.DataFrame.from_dict(df, orient="index")


def encode_text(text):
    enc = tokenizer(text, return_tensors="pt", truncation=True).to(device)

    output = model.encoder(
        input_ids=enc["input_ids"],
        attention_mask=enc["attention_mask"],
        return_dict=True,
    )

    embeddings = output.last_hidden_state.mean(
        dim=1
    ).squeeze()  # mean over all tokens (mb CLS?)

    return embeddings.cpu().detach()


def preprocess(row: pd.Series):
    row = row.fillna("unknown")  # empty?
    # remove column description / title / cat?
    return f"Description: {row['description']}. Title: {row['title']}. Categories: {', '.join(row['categories'][0])}"


def get_data(cached=True):
    print("fda")
    if not cached:
        df = getDF("../data/meta_Beauty.json.gz")

        file_name = "../data/reviews_Beauty_5.json"

        unique_items = set()
        unique_users = set()

        with open(file_name, "r") as file:
            for line in file:
                review = json.loads(line.strip())
                unique_items.add(review["asin"])
                unique_users.add(review["reviewerID"])

        df = df[df["asin"].isin(unique_items)]

        df["combined_text"] = df.apply(preprocess, axis=1)

        with torch.no_grad():
            df["embeddings"] = df["combined_text"].progress_apply(encode_text)
    else:
        print("bababa", flush=True)
        with open('final_data_reduced.pkl', 'rb') as file:
            data_reduced = pickle.load(file)

        df = torch.from_numpy(data_reduced)
        print(df, flush=True)

    return df


def search_similar_items(items_with_tuples, clust2search, max_cnt=5):
    random.shuffle(items_with_tuples)
    cnt = 0
    similars = []
    for asin, item, clust_tuple in items_with_tuples:
        if clust_tuple[: len(clust2search)] == clust2search:
            similars.append((asin, item, clust_tuple))
            cnt += 1
        if cnt >= max_cnt:
            return similars
    return similars
