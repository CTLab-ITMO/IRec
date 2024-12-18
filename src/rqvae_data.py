import pandas as pd
import json
import gzip
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import random

from tqdm import tqdm

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
        df = torch.load("../data/df_with_embs.pt", weights_only=False)
        
    return df

def get_cb_tuples(rqvae, embeddings):
    ind_lists = []
    for cb in rqvae.codebooks:
        dist = torch.cdist(rqvae.encoder(embeddings), cb)
        ind_lists.append(dist.argmin(dim=-1).cpu().numpy())

    return zip(*ind_lists)


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



