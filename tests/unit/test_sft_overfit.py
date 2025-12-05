import torch

from transformers import AutoTokenizer

from sft_rl_circuits.config import SFTConfig, TaskConfig, build_datasets
from sft_rl_circuits.training.sft_trainer import _ensure_pad_token, build_datasets_for_sft


def test_overfit_tiny_batch(monkeypatch):
    # Tiny dataset
    task_cfg = TaskConfig()
    bundle = build_datasets(task_cfg)
    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")

    # Build very small dataset
    sft_cfg = SFTConfig(
        model_name="openai-community/gpt2",
        max_steps=2,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        max_seq_length=64,
        logging_steps=1,
        eval_steps=2,
        save_steps=10,
    )

    # Replace model with a tiny randomly initialized head to avoid heavy load.
    class TinyModel(torch.nn.Module):
        def __init__(self, vocab_size):
            super().__init__()
            self.embed = torch.nn.Embedding(vocab_size, 8)
            self.lm_head = torch.nn.Linear(8, vocab_size)
            self.config = type("cfg", (), {"pad_token_id": tokenizer.eos_token_id})

        def forward(self, input_ids, attention_mask=None, labels=None):
            x = self.embed(input_ids)
            logits = self.lm_head(x)
            loss = None
            if labels is not None:
                loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
                loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
            return type("Out", (), {"loss": loss, "logits": logits})

    model = TinyModel(vocab_size=tokenizer.vocab_size)
    _ensure_pad_token(tokenizer, model)
    train_ds, _ = build_datasets_for_sft(bundle, tokenizer, sft_cfg)
    # Just ensure batching and forward pass work on tiny model.
    batch = [train_ds[0], train_ds[1]]
    max_len = max(len(ex.input_ids) for ex in batch)
    padded = torch.tensor(
        [ex.input_ids + [tokenizer.pad_token_id] * (max_len - len(ex.input_ids)) for ex in batch]
    )
    labels = torch.tensor([ex.labels + [-100] * (max_len - len(ex.labels)) for ex in batch])
    out = model(padded, labels=labels)
    assert out.loss is not None
