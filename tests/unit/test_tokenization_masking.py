from transformers import AutoTokenizer

from sft_rl_circuits.training.tokenization import tokenize_formatted_example


class DummyExample:
    def __init__(self, prompt: str, target: str):
        self.prompt = prompt
        self.target = target
        self.metadata = {}


def test_prompt_tokens_are_masked():
    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
    ex = DummyExample(prompt="Rule: Add\nCards:\nRED 1\n\nAnswer:", target="Answer: 5")
    tok = tokenize_formatted_example(
        example=ex,
        tokenizer=tokenizer,
        max_length=32,
        pad_to_max_length=False,
        mask_prompt_loss=True,
    )
    # Labels for prompt portion should be -100.
    prompt_ids = tokenizer(ex.prompt, add_special_tokens=False)["input_ids"]
    assert tok.labels[: len(prompt_ids)] == [-100] * len(prompt_ids)
    # Target token should be unmasked.
    assert tok.labels[len(prompt_ids)] != -100


def test_no_masking_when_disabled():
    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
    ex = DummyExample(prompt="Rule: Add\nCards:\nRED 1\n\nAnswer:", target="Answer: 5")
    tok = tokenize_formatted_example(
        example=ex,
        tokenizer=tokenizer,
        max_length=32,
        pad_to_max_length=False,
        mask_prompt_loss=False,
    )
    assert all(label != -100 for label in tok.labels[: len(tok.input_ids)])
