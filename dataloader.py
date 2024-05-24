from torch.utils.data import DataLoader
from dataset import CustomTrainDataset, CustomTestDataset


class CustomTrainDataloader(DataLoader):
    def __init__(self, custom_dataset: CustomTrainDataset, batch_size, shuffle):
        super().__init__(custom_dataset, batch_size=batch_size, shuffle=shuffle)
        self.custom_dataset = custom_dataset

    def get_examples(self):
        return self.custom_dataset.examples


class CustomTestDataloader(DataLoader):
    def __init__(self, custom_dataset: CustomTestDataset, batch_size, shuffle):
        super().__init__(custom_dataset, batch_size=batch_size, shuffle=shuffle)
        self.custom_dataset = custom_dataset

    def get_examples(self):
        return self.custom_dataset.examples

    def get_hypotheses(self):
        return self.custom_dataset.hyps_refs
