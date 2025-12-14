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

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_lora_target_modules(model):
    model_type = getattr(model.config, "model_type", "").lower()
    # DeBERTa (v2/v3) + BERT-like: thường ổn với query/key/value/dense
    if model_type in ["deberta", "deberta-v2", "bert", "albert", "electra", "xlm-roberta", "roberta"]:
        return ["query", "key", "value", "dense"]
    # fallback
    return ["q_proj", "k_proj", "v_proj", "o_proj", "dense"]


# ============================================================
# 1. Dataset + domain mapping (fit trên train)
# ============================================================

class MultiTaskFraudDataset(Dataset):
    """
    Trả về: text, labels (fraud), domain_labels
    domain2id phải được fit từ train để valid/test dùng chung mapping
    """
    def __init__(self, csv_file: str, domain2id: dict = None, fit_domain: bool = False):
        df = pd.read_csv(csv_file)
        needed = {"text", "label", "domain"}
        if not needed.issubset(df.columns):
            raise ValueError(f"CSV phải có cột {needed}.")

        self.texts = df["text"].astype(str).tolist()
        self.labels = df["label"].astype(int).tolist()

        raw_domains = df["domain"].astype(str).tolist()

        if fit_domain:
            uniq = sorted(set(raw_domains))
            self.domain2id = {d: i for i, d in enumerate(uniq)}
        else:
            if domain2id is None:
                raise ValueError("domain2id is required when fit_domain=False")
            self.domain2id = domain2id

        # map domain; domain lạ (không có trong train) -> 0 (hoặc bạn tạo thêm 'UNK')
        self.domain_labels = [self.domain2id.get(d, 0) for d in raw_domains]
        self.num_domains = len(self.domain2id)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "text": self.texts[idx],
            "labels": self.labels[idx],               # fraud label
            "domain_labels": self.domain_labels[idx], # domain label
        }


@dataclass
class TokenizeCollator:
    tokenizer: any
    max_length: int = 256

    def __call__(self, features):
        texts = [f["text"] for f in features]
        labels = torch.tensor([f["labels"] for f in features], dtype=torch.long)
        domain_labels = torch.tensor([f["domain_labels"] for f in features], dtype=torch.long)

        enc = self.tokenizer(
            texts,
            truncation=True,
            max_length=self.max_length,
            padding=True,
            return_tensors="pt",
        )
        enc["labels"] = labels
        enc["domain_labels"] = domain_labels
        return enc


# ============================================================
# 2. Model: shared encoder + 2 heads
# ============================================================

class DebertaMultiTask(nn.Module):
    def __init__(self, model_name: str, num_labels: int, num_domains: int, dropout: float = None):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)

        hidden = self.encoder.config.hidden_size
        p = dropout if dropout is not None else getattr(self.encoder.config, "hidden_dropout_prob", 0.1)

        self.dropout = nn.Dropout(p)
        self.fraud_head = nn.Linear(hidden, num_labels)
        self.domain_head = nn.Linear(hidden, num_domains)

        self.loss_fct = nn.CrossEntropyLoss()
        self.loss_domain_fct = nn.CrossEntropyLoss()

    def forward(self, input_ids=None, attention_mask=None, labels=None, domain_labels=None, lambda_domain: float = 0.3):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)

        if getattr(self.encoder.config, "model_type", "").lower().find("modernbert") >= 0:
            # mean pooling with mask
            last = outputs.last_hidden_state
            mask = attention_mask.unsqueeze(-1).float()
            pooled = (last * mask).sum(1) / mask.sum(1).clamp(min=1e-6)
        else:
            pooled = outputs.last_hidden_state[:, 0, :]
        x = self.dropout(pooled)


        fraud_logits = self.fraud_head(x)
        domain_logits = self.domain_head(x)

        loss = None
        loss_fraud = None
        loss_domain = None

        if labels is not None:
            loss_fraud = self.loss_fct(fraud_logits, labels)
        if domain_labels is not None:
            loss_domain = self.loss_domain_fct(domain_logits, domain_labels)

        if (loss_fraud is not None) and (loss_domain is not None):
            loss = loss_fraud + lambda_domain * loss_domain
        elif loss_fraud is not None:
            loss = loss_fraud
        elif loss_domain is not None:
            loss = loss_domain

        return {
            "loss": loss,
            "loss_fraud": loss_fraud,
            "loss_domain": loss_domain,
            "logits": fraud_logits,
            "domain_logits": domain_logits,
        }


# ============================================================
# 3. Early Stopping
# ============================================================

class EarlyStopping:
    def __init__(self, patience=3, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0

    def step(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience


# ============================================================
# 4. Evaluation (fraud metrics + optional domain acc)
# ============================================================

@torch.no_grad()
def evaluate(model, dataloader, device, lambda_domain=0.3, return_preds=False):
    model.eval()
    total_loss = 0.0

    fraud_correct = 0
    fraud_total = 0

    domain_correct = 0
    domain_total = 0

    all_preds = []
    all_labels = []

    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}

        out = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
            domain_labels=batch["domain_labels"],
            lambda_domain=lambda_domain,
        )

        loss = out["loss"]
        fraud_logits = out["logits"]
        domain_logits = out["domain_logits"]

        total_loss += float(loss.item())

        fraud_preds = torch.argmax(fraud_logits, dim=1)
        fraud_correct += int((fraud_preds == batch["labels"]).sum().item())
        fraud_total += int(batch["labels"].size(0))

        domain_preds = torch.argmax(domain_logits, dim=1)
        domain_correct += int((domain_preds == batch["domain_labels"]).sum().item())
        domain_total += int(batch["domain_labels"].size(0))

        if return_preds:
            all_preds.extend(fraud_preds.cpu().numpy())
            all_labels.extend(batch["labels"].cpu().numpy())

    avg_loss = total_loss / max(1, len(dataloader))
    fraud_acc = fraud_correct / max(1, fraud_total)
    domain_acc = domain_correct / max(1, domain_total)

    if return_preds:
        f1 = f1_score(np.array(all_labels), np.array(all_preds), average="binary")
        return avg_loss, fraud_acc, f1, domain_acc, np.array(all_preds), np.array(all_labels)

    return avg_loss, fraud_acc, domain_acc


def plot_confusion_matrix(y_true, y_pred, save_path, class_names=("Legitimate", "Fraud")):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={"label": "Count"})
    plt.title("Confusion Matrix", fontsize=14, fontweight="bold")
    plt.ylabel("True Label", fontsize=12)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Confusion matrix saved to: {save_path}")

def plot_history(history, out_dir):
    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss
    plt.figure()
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training & Validation Loss")
    plt.savefig(os.path.join(out_dir, "loss_curve.png"), dpi=300)
    plt.close()

    # Fraud metrics
    plt.figure()
    plt.plot(epochs, history["val_acc"], label="Val Accuracy")
    plt.plot(epochs, history["val_f1"], label="Val F1")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.legend()
    plt.title("Validation Metrics")
    plt.savefig(os.path.join(out_dir, "metrics_curve.png"), dpi=300)
    plt.close()

    # Domain acc
    plt.figure()
    plt.plot(epochs, history["val_domain_acc"], label="Val Domain Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Domain Accuracy")
    plt.savefig(os.path.join(out_dir, "domain_acc_curve.png"), dpi=300)
    plt.close()



# ============================================================
# 5. Train
# ============================================================

def train(
    model_name="microsoft/deberta-v3-base",
    train_csv="data/train_clean.csv",
    valid_csv="data/validation_clean.csv",
    test_csv="data/test_clean.csv",
    out_dir="outputs",
    num_epochs=10,
    batch_size=16,
    lr=2e-5,
    max_length=256,
    warmup_ratio=0.1,
    weight_decay=0.01,
    grad_clip=1.0,
    seed=42,
    use_lora=True,
    lora_r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    patience=3,
    min_delta=1e-4,
    resume_from_checkpoint=None,
    lambda_domain=0.3,  # trọng số loss domain
):
    os.makedirs(out_dir, exist_ok=True)
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token is not None else tokenizer.unk_token

    # fit domain mapping từ train
    train_ds_tmp = MultiTaskFraudDataset(train_csv, fit_domain=True)
    domain2id = train_ds_tmp.domain2id
    num_domains = train_ds_tmp.num_domains

    train_ds = train_ds_tmp
    val_ds = MultiTaskFraudDataset(valid_csv, domain2id=domain2id, fit_domain=False)
    test_ds = MultiTaskFraudDataset(test_csv, domain2id=domain2id, fit_domain=False)

    model = DebertaMultiTask(model_name=model_name, num_labels=2, num_domains=num_domains)

    # LoRA chỉ bắn vào encoder
    if use_lora:
        target_modules = get_lora_target_modules(model.encoder)
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type=TaskType.FEATURE_EXTRACTION,
        )
        model.encoder = get_peft_model(model.encoder, lora_config)
        model.encoder.print_trainable_parameters()

    model.to(device)

    collator = TokenizeCollator(tokenizer=tokenizer, max_length=max_length)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collator)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collator)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collator)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    total_steps = len(train_loader) * num_epochs
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    early_stopper = EarlyStopping(patience=patience, min_delta=min_delta)
    best_val_loss = float("inf")
    best_val_f1 = 0.0  # F1 càng cao càng tốt
    best_epoch = 0
    best_loss_epoch = 0
    best_loss_path = os.path.join(out_dir, "best_model_loss.pt")
    best_f1_path = os.path.join(out_dir, "best_model_f1.pt")
    checkpoint_path = os.path.join(out_dir, "last_checkpoint.pt")

    start_epoch = 1
    history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_f1": [], "val_domain_acc": []}

    if resume_from_checkpoint and os.path.exists(resume_from_checkpoint):
        ckpt = torch.load(resume_from_checkpoint, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        best_val_f1 = ckpt.get("best_val_f1", 0.0)
        best_epoch = ckpt.get("best_epoch", ckpt["epoch"])
        best_loss_epoch = ckpt.get("best_loss_epoch", ckpt["epoch"])
        history = ckpt.get("history", history)
        print(f"Resumed from epoch {ckpt['epoch']} | best_val_loss={best_val_loss:.4f} (epoch {best_loss_epoch}) | best_val_f1={best_val_f1:.4f} (epoch {best_epoch})")

    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    total_start_time = time.time()
    
    for epoch in range(start_epoch, num_epochs + 1):
        epoch_start_time = time.time()
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}", leave=True)

        for step, batch in enumerate(pbar, start=1):
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                out = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                    domain_labels=batch["domain_labels"],
                    lambda_domain=lambda_domain,
                )
                loss = out["loss"]

            scaler.scale(loss).backward()
            if grad_clip and grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            running_loss += float(loss.item())
            pbar.set_postfix(loss=f"{loss.item():.4f}", avg_loss=f"{running_loss/step:.4f}")

        train_loss = running_loss / max(1, len(train_loader))

        val_start = time.time()
        val_loss, val_acc, val_f1, val_domain_acc, _, _ = evaluate(
            model, val_loader, device, lambda_domain=lambda_domain, return_preds=True
        )
        val_time = time.time() - val_start
        
        epoch_time = time.time() - epoch_start_time

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)
        history["val_domain_acc"].append(val_domain_acc)

        print(
            f"Epoch {epoch} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} "
            f"| fraud_acc={val_acc:.4f} | fraud_f1={val_f1:.4f} | domain_acc={val_domain_acc:.4f} "
            f"| epoch_time={epoch_time:.1f}s (val={val_time:.1f}s)"
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            torch.save(model.state_dict(), best_f1_path)
            print(f"  → Saved best F1 model (epoch {epoch}): {best_f1_path}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_loss_epoch = epoch
            torch.save(model.state_dict(), best_loss_path)
            print(f"  → Saved best loss model (epoch {epoch}): {best_loss_path}")

        ckpt = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val_f1": best_val_f1,
            "best_val_loss": best_val_loss,
            "best_epoch": best_epoch,
            "best_loss_epoch": best_loss_epoch,
            "history": history,
            "domain2id": domain2id,
        }
        torch.save(ckpt, checkpoint_path)

        if early_stopper.step(val_loss):
            print("Early stopping triggered.")
            break

    # Total training time
    total_time = time.time() - total_start_time
    print(f"\n{'='*60}")
    print(f"Total training time: {total_time/60:.2f} minutes ({total_time:.1f}s)")
    print(f"Average time per epoch: {total_time/(epoch-start_epoch+1):.1f}s")
    print(f"{'='*60}\n")

    print("\nEvaluating on test set (best by LOSS)...")
    model.load_state_dict(torch.load(best_loss_path, map_location=device))
    loss_test_loss, loss_test_acc, loss_test_f1, loss_test_domain_acc, loss_test_preds, loss_test_labels = evaluate(
        model, test_loader, device, lambda_domain=lambda_domain, return_preds=True
    )
    
    print("\nEvaluating on test set (best by F1)...")
    model.load_state_dict(torch.load(best_f1_path, map_location=device))
    f1_test_loss, f1_test_acc, f1_test_f1, f1_test_domain_acc, f1_test_preds, f1_test_labels = evaluate(
        model, test_loader, device, lambda_domain=lambda_domain, return_preds=True
    )
    
    print("\n===============================")
    print(" FINAL TEST RESULTS (Multitask)")
    print("===============================")
    print(f"Best-by-LOSS  (epoch {best_loss_epoch}) | loss={loss_test_loss:.4f} | acc={loss_test_acc:.4f} | f1={loss_test_f1:.4f} | domain_acc={loss_test_domain_acc:.4f}")
    print(f"Best-by-F1    (epoch {best_epoch}) | loss={f1_test_loss:.4f} | acc={f1_test_acc:.4f} | f1={f1_test_f1:.4f} | domain_acc={f1_test_domain_acc:.4f}")
    print("===============================")
    
    plot_confusion_matrix(loss_test_labels, loss_test_preds, os.path.join(out_dir, "confusion_matrix_test_best_loss.png"))
    plot_confusion_matrix(f1_test_labels, f1_test_preds, os.path.join(out_dir, "confusion_matrix_test_best_f1.png"))
    
    print("\n[Best-by-LOSS] Classification report:")
    print(classification_report(loss_test_labels, loss_test_preds, target_names=["Legitimate", "Fraud"], digits=4))
    
    print("\n[Best-by-F1] Classification report:")
    print(classification_report(f1_test_labels, f1_test_preds, target_names=["Legitimate", "Fraud"], digits=4))


if __name__ == "__main__":
    train(
        model_name="microsoft/deberta-v3-base",
        train_csv="data/train_clean.csv",
        valid_csv="data/validation_clean.csv",
        test_csv="data/test_clean.csv",
        out_dir="outputs_multitaskfull_deberta",
        num_epochs=15,
        batch_size=16,
        lr=2e-5,
        max_length=256,
        warmup_ratio=0.1,
        use_lora=False,
        patience=3,
        resume_from_checkpoint=None,
        lambda_domain=0.3,
    )
