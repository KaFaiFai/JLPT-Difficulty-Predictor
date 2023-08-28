"""
Source: https://www.kaggle.com/datasets/rmisra/news-category-dataset
copy "News_Category_Dataset_v3.json" and put in `training_data/News_Category_Dataset_v3.json`
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
    return text

    # removing stop-words
    # text = [word for word in text.split() if word not in list(stopwords.words("english"))]

    # word lemmatization
    # sentence = []
    # lemmatizer = WordNetLemmatizer()
    # for word in text:
    #   sentence.append(lemmatizer.lemmatize(word, "v"))

    # return " ".join(sentence)


class NewsCategory(Dataset):
    LABELS = (
        "U.S. NEWS",
        "COMEDY",
        "PARENTING",
        "WORLD NEWS",
        "CULTURE & ARTS",
        "TECH",
        "SPORTS",
        "ENTERTAINMENT",
        "POLITICS",
        "WEIRD NEWS",
        "ENVIRONMENT",
        "EDUCATION",
        "CRIME",
        "SCIENCE",
        "WELLNESS",
        "BUSINESS",
        "STYLE & BEAUTY",
        "FOOD & DRINK",
        "MEDIA",
        "QUEER VOICES",
        "HOME & LIVING",
        "WOMEN",
        "BLACK VOICES",
        "TRAVEL",
        "MONEY",
        "RELIGION",
        "LATINO VOICES",
        "IMPACT",
        "WEDDINGS",
        "COLLEGE",
        "PARENTS",
        "ARTS & CULTURE",
        "STYLE",
        "GREEN",
        "TASTE",
        "HEALTHY LIVING",
        "THE WORLDPOST",
        "GOOD NEWS",
        "WORLDPOST",
        "FIFTY",
        "ARTS",
        "DIVORCE",
    )

    def __init__(self, df_path):
        super().__init__()
        df = pd.read_json(df_path, lines=True)
        df["news"] = df["headline"] + " " + df["short_description"]
        df = df[["category", "news"]]
        # df["news"] = df["news"].apply(clean_text)

        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index) -> tuple[str, int]:
        text = self.df["news"][index]
        label = self.df["category"][index]
        target = self.label2target(label)

        return text, target

    @classmethod
    def label2target(cls, label):
        return cls.LABELS.index(label)

    @classmethod
    def target2label(cls, target):
        return cls.LABELS[target]


def _test():
    dataset = NewsCategory("training_data/News_Category_Dataset_v3.json")
    print(dataset.df.keys())
    print(dataset.df.shape)
    print(dataset.df.head())
    print(len(dataset.LABELS))
    print(dataset[0])


if __name__ == "__main__":
    _test()
