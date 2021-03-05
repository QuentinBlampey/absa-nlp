from torch.utils.data import Dataset

class AbsaDataset:
    def __init__(self, df):
        self.df = df
        self.categories = set(df.category)
        self.category_to_index = {category: i for i, category in enumerate(self.categories)}
        self.num_categories = len(self.categories)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        return self.df.sentence[index], self.category_to_index[self.df.category[index]], self.df.label[index]