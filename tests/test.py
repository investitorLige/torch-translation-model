import torch
from src import train_model, translate_single, en_tokenizer, sr_tokenizer


def test_train_model_runs():
    dummy_dataloader = [{"input_ids": torch.randint(0, 100, (2, 10)),
                         "labels": torch.randint(0, 100, (2, 10)),
                         "src_attention_mask": torch.ones(2, 10),
                         "tgt_attention_mask": torch.ones(2, 10)}]

    model = train_model(
        w_dataloader=dummy_dataloader,
        all_dataloader=dummy_dataloader,
        p_dataloader=dummy_dataloader,
        val_loader=None,
        epoch_len=1
    )

    # Check if model is returned
    assert model is not None
    print(translate_single(model, "a", en_tokenizer, sr_tokenizer))
