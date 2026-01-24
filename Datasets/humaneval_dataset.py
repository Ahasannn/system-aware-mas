from typing import Union, Literal
import pandas as pd
from datasets import load_dataset


class HumanEvalDataset:
    def __init__(self, split: Union[Literal['train'], Literal['test']], split_ratio: float = 0.2, seed: int = 42):
        """
        HumanEval dataset loader that auto-downloads from Hugging Face.

        Args:
            split: 'train' or 'test' split
            split_ratio: Ratio for train/test split (default 0.2 means 20% train, 80% test)
            seed: Random seed for reproducible splits
        """
        # Load the full dataset from Hugging Face
        dataset = load_dataset('openai_humaneval', split='test')

        # Convert to pandas DataFrame for easier manipulation
        df = pd.DataFrame(dataset)

        # Shuffle with seed for reproducibility
        df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

        # Split into train and test
        split_index = int(len(df) * split_ratio)

        if split == 'train':
            self.df = df.iloc[:split_index].reset_index(drop=True)
        elif split == 'test':
            self.df = df.iloc[split_index:].reset_index(drop=True)
        else:
            raise ValueError(f"Invalid split: {split}. Must be 'train' or 'test'")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        return self.df.iloc[index]


class HumanEvalDataLoader:
    def __init__(self, dataset, batch_size=1, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = list(range(len(dataset)))
        if self.shuffle:
            self._shuffle_indices()
        self.index = 0

    def _shuffle_indices(self):
        import random
        random.shuffle(self.indices)

    def __iter__(self):
        batch = []
        for i in range(len(self.indices)):
            batch.append(self.dataset[self.indices[i]])
            if len(batch) == self.batch_size or i == len(self.indices) - 1:
                yield batch
                batch = []

    def __next__(self):
        if self.index >= len(self.dataset):
            raise StopIteration

        batch_indices = self.indices[self.index:self.index + self.batch_size]
        batch = [self.dataset[i] for i in batch_indices]
        self.index += self.batch_size
        return batch
