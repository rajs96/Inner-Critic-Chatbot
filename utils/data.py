from typing import List, Tuple, Callable
import torch
from torch import Tensor
from torch.utils.data import Dataset


class TextDataset(Dataset):
    def __init__(
        self, text_list: List[str], tokenizer_func: Callable[[List[str]], Tensor]
    ):
        self.text_list = text_list
        self.tokenizer_func = tokenizer_func

    def __len__(self) -> int:
        return len(self.text_list)

    def collate_fn(self, batches: List[Tuple[Tensor, Tensor]]) -> Tuple[Tensor, Tensor]:
        return tuple(map(lambda l: torch.cat(l, dim=0), zip(*batches)))

    def __getitem__(self, idx: int | slice) -> Tuple[Tensor, Tensor]:
        batch = self.text_list[idx]
        if isinstance(batch, str):
            # it's a single value
            input_ids_batch = self.tokenizer_func([batch])["input_ids"]
            attention_mask_batch = self.tokenizer_func([batch])["attention_mask"]
        else:
            input_ids_batch = self.tokenizer_func(batch)["input_ids"]
            attention_mask_batch = self.tokenizer_func(batch)["attention_mask"]

        return input_ids_batch, attention_mask_batch
