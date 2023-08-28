"""
Source: https://github.com/yukiar/CEFR-SP/tree/main
copy "CEFR-SP" and put in `training_data/CEFR-SP`
"""


import pandas as pd
from torch.utils.data import Dataset
from pathlib import Path

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

nltk.download("stopwords")
nltk.download("wordnet")


def clean_text(text):
    whitespace = re.compile(r"\s+")
    user = re.compile(r"(?i)@[a-z0-9_]+")
    text = whitespace.sub(" ", text)
    text = user.sub("", text)
    text = re.sub(r"\[[^()]*\]", "", text)
    text = re.sub("\d+", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"(?:@\S*|#\S*|http(?=.*://)\S*)", "", text)
    text = text.lower()
    # return text

    # removing stop-words
    text = [word for word in text.split() if word not in list(stopwords.words("english"))]

    # word lemmatization
    sentence = []
    lemmatizer = WordNetLemmatizer()
    for word in text:
      sentence.append(lemmatizer.lemmatize(word, "v"))

    return " ".join(sentence)


class CEFR_SP(Dataset):
    LABELS = ("A1", "A2", "B1", "B2", "C1", "C2")

    def __init__(self, folder_path, split="train"):
        super().__init__()

        assert split in ("train", "dev", "test")

        data_path1 = Path(folder_path) / "SCoRE" / f"CEFR-SP_SCoRE_{split}.txt"
        df1 = pd.read_csv(data_path1, delimiter="\t", names=["text", "target_a", "target_b"])

        data_path2 = Path(folder_path) / "Wiki-Auto" / f"CEFR-SP_Wikiauto_{split}.txt"
        df2 = pd.read_csv(data_path2, delimiter="\t", names=["text", "target_a", "target_b"])

        df = pd.concat([df1, df2], ignore_index=True)
        df["text"] = df["text"].apply(clean_text)

        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index) -> tuple[str, int]:
        text = self.df["text"][index]
        level1 = self.df["target_a"][index]
        level2 = self.df["target_b"][index]
        target = max(level1, level2) - 1

        return text, target

    @classmethod
    def label2target(cls, label):
        return cls.LABELS.index(label)

    @classmethod
    def target2label(cls, target):
        return cls.LABELS[target]


def _test():
    dataset = CEFR_SP("training_data/CEFR-SP")
    print(dataset.df.keys())
    print(dataset.df.shape)
    print(dataset.df.head())
    print(len(dataset.LABELS))
    print(dataset[0])


if __name__ == "__main__":
    _test()
