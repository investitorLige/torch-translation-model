from torch.utils.data import Dataset, DataLoader
from .model import en_tokenizer, sr_tokenizer, device


class TranslationDataset(Dataset):
    def __init__(self, en_texts, sr_texts, en_tokenizer, sr_tokenizer):
        self.en_texts = en_texts
        self.sr_texts = sr_texts
        self.en_tokenizer = en_tokenizer
        self.sr_tokenizer = sr_tokenizer

    def __len__(self):
        return len(self.en_texts)

    def __getitem__(self, idx):
        en_encoded = self.en_tokenizer(self.en_texts[idx], return_tensors="pt", padding=False, truncation=True,
                                       max_length=512)
        sr_encoded = self.sr_tokenizer(self.sr_texts[idx], return_tensors="pt", padding=False, truncation=True,
                                       max_length=512)
        return {
            "input_ids": en_encoded["input_ids"].squeeze(),
            "src_attention_mask": en_encoded["attention_mask"].squeeze(),
            "labels": sr_encoded["input_ids"].squeeze(),
            "tgt_attention_mask": sr_encoded["attention_mask"].squeeze(),
        }


def create_dataset(dataset):
    if sr_tokenizer.pad_token is None:
        sr_tokenizer.add_special_tokens({'pad_token': '<pad>'})
    if en_tokenizer.pad_token is None:
        en_tokenizer.add_special_tokens({'pad_token': '<pad>'})

    def collate_fn(batch):
        en_lengths = [len(x["input_ids"]) for x in batch]
        #print(en_lengths, "en")
        sr_lengths = [len(x["labels"]) for x in batch]
        #print(sr_lengths, "sr")
        max_len = min(512, max(max(en_lengths), max(sr_lengths)))  # Shared max length

        # Pad both sides equally
        en_padded = en_tokenizer.pad(
            {"input_ids": [x["input_ids"] for x in batch],
             "attention_mask": [x["src_attention_mask"] for x in batch]},
            padding="max_length",
            max_length=max_len,
            return_tensors="pt"
        )

        sr_padded = sr_tokenizer.pad(
            {"input_ids": [x["labels"] for x in batch],
             "attention_mask": [x["tgt_attention_mask"] for x in batch]},
            padding="max_length",
            max_length=max_len,
            return_tensors="pt"
        )

        return {
            "input_ids": en_padded["input_ids"].to(device),
            "src_attention_mask": en_padded["attention_mask"].to(device),
            "labels": sr_padded["input_ids"].to(device),
            "tgt_attention_mask": sr_padded["attention_mask"].to(device),
        }

    return DataLoader(
        dataset,
        batch_size=8,
        collate_fn=collate_fn,
        shuffle=False
    )
