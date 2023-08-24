import pandas as pd
from torch.utils.data import Dataset


class EnglishCEFRDataset(Dataset):
    LEVELS = ("A1", "A2", "B1", "B2", "C1", "C2")

    def __init__(self, df_path):
        super().__init__()
        self.df = pd.read_csv(df_path, index_col=0)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index) -> tuple[str, int]:
        sentence = self.df["Sentence"][index]
        level = self.df["Level"][index]
        label = self.level2target(level)

        return sentence, label

    @classmethod
    def level2target(cls, level):
        return cls.LEVELS.index(level)

    @classmethod
    def target2level(cls, target):
        return cls.LEVELS[target]
