"""
Source: https://www.kaggle.com/datasets/amontgomerie/cefr-levelled-english-texts  
download it and put in `training_data/cefr_leveled_texts.csv`
"""


import pandas as pd
from torch.utils.data import Dataset


class EnglishCEFRDataset(Dataset):
    LEVELS = ("A1", "A2", "B1", "B2", "C1", "C2")

    def __init__(self, df_path):
        super().__init__()
        self.df = pd.read_csv(df_path)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index) -> tuple[str, int]:
        sentence = self.df["text"][index]
        level = self.df["label"][index]
        label = self.level2target(level)

        return sentence, label

    @classmethod
    def level2target(cls, level):
        return cls.LEVELS.index(level)

    @classmethod
    def target2level(cls, target):
        return cls.LEVELS[target]


def _test():
    dataset = EnglishCEFRDataset("training_data/cefr_leveled_texts.csv")
    print(dataset.df.keys())
    print(dataset[0])


if __name__ == "__main__":
    _test()
