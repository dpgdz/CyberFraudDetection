import os
import random
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, f1_score

from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModel,
    get_linear_schedule_with_warmup,
)

from peft import LoraConfig, get_peft_model, TaskType


# ============================================================
# 0. Utils
# ============================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def is_modernbert(model):
    return "modernbert" in model.config.model_type.lower()


def get_lora_target_modules(model):
    model_type = model.config.model_type.lower()

    if "modernbert" in model_type:
        return ["Wqkv", "Wo"]

    if "deberta" in model_type:
        return ["query", "key", "value", "dense"]

    if model_type in ["bert", "roberta", "xlm-roberta"]:
        return ["query", "key", "value"]

    return []


# ============================================================
# 1. Dataset
# ============================================================

class FraudDataset(Dataset):
    def __init__(self, csv_file):
        df = pd.read_csv(csv_file)
        self.texts = df["text"].astype(str).tolist()
        self.labels = df["label"].astype(int).tolist()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "text": self.texts[idx],
            "labels": self.labels[idx],
        }


@dataclass
class TokenizeCollator:
    tokenizer: any
    max_length: int = 256

    def __call__(self, features):
        texts = [f["text"] for f in features]
        labels = torch.tensor([f["labels"] for f in features], dtype=torch.long)

        enc = self.tokenizer(
            texts,
            truncation=True,
            max_length=self.max_length,
            padding=True,
            return_tensors="pt",
        )
        enc["labels"] = labels
        return enc


# ============================================================
# 2. MODEL (CHUẨN KIẾN TRÚC)
# ============================================================

class SequenceClassifier(nn.Module):
    def __init__(self, model_name, num_labels=2):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.hidden = self.encoder.config.hidden_size
        self.is_modernbert = is_modernbert(self.encoder)

        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.hidden, num_labels)
        self.loss_fct = nn.CrossEntropyLoss()

    def pool(self, last_hidden_state, attention_mask):
        if self.is_modernbert:
            # mean pooling
            mask = attention_mask.unsqueeze(-1)
            x = (last_hidden_state * mask).sum(1)
            x = x / mask.sum(1)
            return x
        else:
            # CLS pooling
            return last_hidden_state[:, 0, :]

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        x = self.pool(outputs.last_hidden_state, attention_mask)
        x = self.dropout(x)
        logits = self.classifier(x)

        loss = None
        if labels is not None:
            loss = self.loss_fct(logits, labels)

        return {
            "loss": loss,
            "logits": logits
        }


# ============================================================
# 3. Evaluation
# ============================================================

@torch.no_grad()
def evaluate(model, dataloader, device, return_preds=False):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(**batch)

        loss = out["loss"]
        logits = out["logits"]
        preds = torch.argmax(logits, dim=1)

        total_loss += float(loss.item())
        correct += int((preds == batch["labels"]).sum())
        total += batch["labels"].size(0)

        if return_preds:
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch["labels"].cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    acc = correct / total

    if return_preds:
        f1 = f1_score(all_labels, all_preds, average="binary")
        return avg_loss, acc, f1, np.array(all_preds), np.array(all_labels)

    return avg_loss, acc


# ============================================================
# 4. Train
# ============================================================

def train(
    model_name,
    train_csv,
    valid_csv,
    test_csv,
    out_dir,
    num_epochs=10,
    batch_size=16,
    lr=2e-5,
    max_length=256,
    warmup_ratio=0.1,
    use_lora=False,
    lora_r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    patience=3,
):
    os.makedirs(out_dir, exist_ok=True)
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token

    model = SequenceClassifier(model_name)

    # ===== LoRA =====
    if use_lora:
        targets = get_lora_target_modules(model.encoder)
        lora_cfg = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=targets,
            lora_dropout=lora_dropout,
            bias="none",
            task_type=TaskType.FEATURE_EXTRACTION,
        )
        model.encoder = get_peft_model(model.encoder, lora_cfg)
        model.encoder.print_trainable_parameters()

    model.to(device)

    train_ds = FraudDataset(train_csv)
    val_ds = FraudDataset(valid_csv)
    test_ds = FraudDataset(test_csv)

    collator = TokenizeCollator(tokenizer, max_length)
    train_loader = DataLoader(train_ds, batch_size, True, collate_fn=collator)
    val_loader = DataLoader(val_ds, batch_size, False, collate_fn=collator)
    test_loader = DataLoader(test_ds, batch_size, False, collate_fn=collator)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    total_steps = len(train_loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        int(total_steps * warmup_ratio),
        total_steps
    )

    best_loss, best_f1 = float("inf"), 0.0
    best_loss_path = os.path.join(out_dir, "best_loss.pt")
    best_f1_path = os.path.join(out_dir, "best_f1.pt")

    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
    no_improve = 0

    for epoch in range(1, num_epochs + 1):
        model.train()
        running = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")

        for batch in pbar:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                out = model(**batch)
                loss = out["loss"]

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            running += loss.item()
            pbar.set_postfix(loss=loss.item())

        train_loss = running / len(train_loader)
        val_loss, val_acc, val_f1, _, _ = evaluate(
            model, val_loader, device, return_preds=True
        )

        print(
            f"Epoch {epoch} | train={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | val_acc={val_acc:.4f} | val_f1={val_f1:.4f}"
        )

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), best_loss_path)
            no_improve = 0
        else:
            no_improve += 1

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), best_f1_path)

        if no_improve >= patience:
            print("Early stopping")
            break

    print("\n=== TEST (BEST BY LOSS) ===")
    model.load_state_dict(torch.load(best_loss_path))
    loss, acc, f1, preds, labels = evaluate(model, test_loader, device, True)
    print(f"loss={loss:.4f} acc={acc:.4f} f1={f1:.4f}")

    print("\n=== TEST (BEST BY F1) ===")
    model.load_state_dict(torch.load(best_f1_path))
    loss, acc, f1, preds, labels = evaluate(model, test_loader, device, True)
    print(f"loss={loss:.4f} acc={acc:.4f} f1={f1:.4f}")

    print(classification_report(labels, preds, digits=4))


# ============================================================
# 5. RUN
# ============================================================

if __name__ == "__main__":
    train(
        model_name="answerdotai/ModernBERT-base",  # hoặc deberta-v3-base
        train_csv="data/train_clean.csv",
        valid_csv="data/validation_clean.csv",
        test_csv="data/test_clean.csv",
        out_dir="outputs_modernbert",
        num_epochs=10,
        batch_size=16,
        lr=3e-5,
        use_lora=False,
    )
