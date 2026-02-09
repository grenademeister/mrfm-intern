import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset


def read_jsonl(path: Path) -> list[str]:
    samples: list[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            text = str(obj.get("text", "")).strip()
            if text:
                samples.append(text)
    return samples


class TextDataset(Dataset):
    def __init__(self, texts: list[str], tokenizer, max_length: int) -> None:
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


@dataclass
class TrainArgs:
    model_path: str
    data_path: Path
    output_dir: Path
    max_length: int
    per_device_batch_size: int
    grad_accum_steps: int
    lr: float
    epochs: int
    warmup_ratio: float
    weight_decay: float
    logging_steps: int
    save_steps: int
    use_fast: bool
    dtype: str
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    lora_target_modules: list[str]
    save_total_limit: int


def main() -> None:
    parser = argparse.ArgumentParser(description="LoRA DAPT for Qwen using MRI caption text.")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_path", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--per_device_batch_size", type=int, default=8)
    parser.add_argument("--grad_accum_steps", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--logging_steps", type=int, default=50)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--use_fast", type=lambda x: x.lower() in ("1", "true", "t", "yes"), default=True)
    parser.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "fp32"])
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj",
        help="Comma-separated module names",
    )
    parser.add_argument("--save_total_limit", type=int, default=2)
    args = parser.parse_args()

    train_args = TrainArgs(
        model_path=args.model_path,
        data_path=args.data_path,
        output_dir=args.output_dir,
        max_length=args.max_length,
        per_device_batch_size=args.per_device_batch_size,
        grad_accum_steps=args.grad_accum_steps,
        lr=args.lr,
        epochs=args.epochs,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        use_fast=args.use_fast,
        dtype=args.dtype,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=args.lora_target_modules.split(","),
        save_total_limit=args.save_total_limit,
    )

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
    except ImportError as exc:
        raise ImportError("transformers is required for LoRA DAPT training") from exc

    try:
        from peft import LoraConfig, get_peft_model
    except ImportError as exc:
        raise ImportError("peft is required for LoRA DAPT training") from exc

    tokenizer = AutoTokenizer.from_pretrained(
        train_args.model_path,
        use_fast=train_args.use_fast,
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.float16 if train_args.dtype == "fp16" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        train_args.model_path,
        dtype=dtype,
        trust_remote_code=True,
    )

    lora_cfg = LoraConfig(
        r=train_args.lora_r,
        lora_alpha=train_args.lora_alpha,
        lora_dropout=train_args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=train_args.lora_target_modules,
    )
    model = get_peft_model(model, lora_cfg)

    texts = read_jsonl(train_args.data_path)
    dataset = TextDataset(texts, tokenizer, train_args.max_length)

    train_args.output_dir.mkdir(parents=True, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=str(train_args.output_dir),
        num_train_epochs=train_args.epochs,
        per_device_train_batch_size=train_args.per_device_batch_size,
        gradient_accumulation_steps=train_args.grad_accum_steps,
        learning_rate=train_args.lr,
        warmup_ratio=train_args.warmup_ratio,
        weight_decay=train_args.weight_decay,
        logging_steps=train_args.logging_steps,
        disable_tqdm=True,
        save_steps=train_args.save_steps,
        save_total_limit=train_args.save_total_limit,
        fp16=train_args.dtype == "fp16",
        bf16=False,
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )
    trainer.train()

    model.save_pretrained(train_args.output_dir)
    tokenizer.save_pretrained(train_args.output_dir)


if __name__ == "__main__":
    main()
