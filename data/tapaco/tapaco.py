import pandas as pd
from datasets import load_dataset
from tqdm import tqdm
import random

#Loading dataset
dataset = load_dataset('tapaco', 'en')

def process_tapaco_dataset(dataset):
    tapaco = []
    # The dataset has only train split.
    for data in tqdm(dataset["train"]):
        keys = data.keys()
        tapaco.append([data[key] for key in keys])
    tapaco_df = pd.DataFrame(
        data=tapaco,
        columns=[
            "paraphrase_set_id",
            "sentence_id",
            "paraphrase",
            "lists",
            "tags",
            "language",
        ],
    )
    return tapaco_df


def generate_true_tapaco_paraphrase_dataset(dataset):
    dataset_df = dataset[["paraphrase", "paraphrase_set_id"]]
    non_single_labels = (
        dataset_df["paraphrase_set_id"]
        .value_counts()[dataset_df["paraphrase_set_id"].value_counts() > 1]
        .index.tolist()
    )
    tapaco_df_sorted = dataset_df.loc[
        dataset_df["paraphrase_set_id"].isin(non_single_labels)
    ]
    tapaco_paraphrases_dataset = []

    for paraphrase_set_id in tqdm(tapaco_df_sorted["paraphrase_set_id"].unique()):
        id_wise_paraphrases = tapaco_df_sorted[
            tapaco_df_sorted["paraphrase_set_id"] == paraphrase_set_id
        ]
        len_id_wise_paraphrases = (
            id_wise_paraphrases.shape[0]
            if id_wise_paraphrases.shape[0] % 2 == 0
            else id_wise_paraphrases.shape[0] - 1
        )
        for ix in range(0, len_id_wise_paraphrases, 2):
            current_phrase = id_wise_paraphrases.iloc[ix][0]
            for count_ix in range(ix + 1, ix + 2):
                next_phrase = id_wise_paraphrases.iloc[ix + 1][0]
                tapaco_paraphrases_dataset.append([current_phrase, next_phrase, 1])
    tapaco_paraphrases_dataset_df = pd.DataFrame(
        tapaco_paraphrases_dataset, columns=["Text", "Paraphrase", "Match"]
    )
    return tapaco_paraphrases_dataset_df

def generate_false_tapaco_paraphrase_dataset(dataset, length):
    dataset_df = dataset[["paraphrase", "paraphrase_set_id"]]
    non_single_labels = (
        dataset_df["paraphrase_set_id"]
        .value_counts()[dataset_df["paraphrase_set_id"].value_counts() > 1]
        .index.tolist()
    )
    tapaco_df_sorted = dataset_df.loc[
        dataset_df["paraphrase_set_id"].isin(non_single_labels)
    ]
    tapaco_false_df = set()
    while len(tapaco_false_df) < length:
        # randomly pick a sentence
        sen1 = random.randint(0, length)
        sen2 = random.randint(0, length)
        while tapaco_df_sorted['paraphrase_set_id'][sen1] == tapaco_df_sorted['paraphrase_set_id'][sen2]:
            sen2 = random.randint(0, length)
        tapaco_false_df.add(tuple([tapaco_df_sorted['paraphrase'][sen1], tapaco_df_sorted['paraphrase'][sen2], 0]))
    tapaco_paraphrases_dataset_df = pd.DataFrame(
        tapaco_false_df, columns=["Text", "Paraphrase", "Match"]
    )
    return tapaco_paraphrases_dataset_df




tapaco_df = process_tapaco_dataset(dataset)
tapaco_true_df = generate_true_tapaco_paraphrase_dataset(tapaco_df)
tapaco_false_df = generate_false_tapaco_paraphrase_dataset(tapaco_df, tapaco_true_df.shape[0])
df = pd.concat([tapaco_true_df, tapaco_false_df], ignore_index=True, sort=False)
df.to_csv("out_file.tsv", sep="\t", index=None)



