"""
Optimized fine tuning pipeline for Phi-3.5-mini
- 4bit + LoRA
- instruction-style reasoning data (fraud detection)
- proper chat formatting and label masking
"""

import inspect
import re
import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    BitsAndBytesConfig,
    EvalPrediction,
    TrainerCallback,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import bitsandbytes as bnb
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def extract_label(text: str) -> int:
    """Extract fraud label from output text. Returns 1 for fraud, 0 for legitimate"""
    if not text:
        return 0
    text_lower = text.lower()
    # look for classification line
    match = re.search(r'classification:\\s*(fraud|legitimate)', text_lower)
    if match:
        return 1 if match.group(1) == 'fraud' else 0
    # fallback: check if 'fraud' appears before 'legitimate'
    fraud_pos = text_lower.find('fraud')
    legit_pos = text_lower.find('legitimate')
    if fraud_pos != -1 and (legit_pos == -1 or fraud_pos < legit_pos):
        return 1
    return 0


class MetricsCallback(TrainerCallback):
    """Custom callback to compute fraud detection metrics during training"""
    
    def __init__(self, trainer_instance, eval_dataset):
        self.trainer_instance = trainer_instance
        self.eval_dataset = eval_dataset
    
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """Compute custom metrics after each evaluation"""
        if metrics is not None:
            custom_metrics = self.trainer_instance.compute_metrics_with_generation(self.eval_dataset)
            metrics.update(custom_metrics)
        return control


class FraudReasoningTrainer:
    """Fine tune a small language model for fraud detection with reasoning"""

    def __init__(
        self,
        model_name: str = "microsoft/Phi-3.5-mini-instruct",
        train_path: str = "data/fraud_train_multi_task.jsonl",
        val_path: str = "data/fraud_val_multi_task.jsonl",
        test_path: str = "data/fraud_test_multi_task.jsonl",
        output_dir: str = "models/phi-3.5-fraud-reasoning",
    ):
        self.model_name = model_name
        self.paths = {
            "train": train_path,
            "validation": val_path,
            "test": test_path,
        }
        self.output_dir = output_dir
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    # ----------------------------------------------------------
    # load tokenizer + model (4bit optional)
    # ----------------------------------------------------------
    def load_model_and_tokenizer(self, use_4bit: bool = True):
        print(f"loading model {self.model_name} ...")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

        if use_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=bnb_config,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
            )

        # prepare for kbit + LoRA
        self.model = prepare_model_for_kbit_training(self.model)
        
        # important for LoRA on 4bit
        if hasattr(self.model, "enable_input_require_grads"):
            self.model.enable_input_require_grads()
        
        # disable cache during training (required for gradient checkpointing)
        if hasattr(self.model, "config"):
            self.model.config.use_cache = False

        print("model loaded")

    # ----------------------------------------------------------
    # LoRA optimization config
    # ----------------------------------------------------------
    def setup_lora(self):
        print("initializing lora...")

        lora_config = LoraConfig(
            r=32,
            lora_alpha=64,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )

        self.model = get_peft_model(self.model, lora_config)

        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())

        print("lora setup complete")
        print(f"trainable params: {trainable:,} ({100 * trainable / total:.2f} percent)")
        print(f"total params: {total:,}")

    # ----------------------------------------------------------
    # dataset prep: chat formatting + tokenization + masking
    # ----------------------------------------------------------
    def prepare_dataset(self, max_length: int = 2048, dataset_fraction: float = 1.0):
        print("loading dataset...")

        dataset = load_dataset(
            "json",
            data_files=self.paths,
        )
        
        # reduce dataset size by percentage (reproducible with seed)
        if dataset_fraction < 1.0:
            for split in ["train", "validation", "test"]:
                original_size = len(dataset[split])
                num_samples = int(original_size * dataset_fraction)
                dataset[split] = dataset[split].shuffle(seed=42).select(range(num_samples))
                print(f"reduced {split} dataset from {original_size} to {len(dataset[split])} samples ({dataset_fraction*100:.1f}%)")

        def preprocess(example):
            """
            We build a chat conversation:
            system: "you provide clear, logical fraud reasoning"
            user: instruction + message
            assistant: reasoning (label)
            Then mask labels for system+user, only train on assistant text.
            """

            instruction = example["instruction"]
            message = example["input"]
            reasoning = example["output"]
            
            # extract ground truth label
            label_id = extract_label(reasoning)

            user_content = f"{instruction}\n\nmessage: {message}"

            messages_full = [
                {"role": "system", "content": "you provide clear, logical fraud reasoning"},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": reasoning},
            ]

            # full conversation (including assistant answer)
            full_text = self.tokenizer.apply_chat_template(
                messages_full,
                tokenize=False,
                add_generation_prompt=False,
            )

            # prompt only (system + user, with assistant role opened, no content)
            prompt_messages = messages_full[:-1]
            prompt_text = self.tokenizer.apply_chat_template(
                prompt_messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            full_tok = self.tokenizer(
                full_text,
                truncation=True,
                max_length=max_length,
            )
            prompt_tok = self.tokenizer(
                prompt_text,
                truncation=True,
                max_length=max_length,
            )

            input_ids = full_tok["input_ids"]
            attention_mask = full_tok["attention_mask"]

            labels = input_ids.copy()
            prompt_len = len(prompt_tok["input_ids"])

            # mask system + user part
            if prompt_len >= len(labels):
                labels = [-100] * len(labels)
            else:
                labels[:prompt_len] = [-100] * prompt_len


            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
                "label_id": label_id,
            }

        # apply preprocess to all splits
        column_names = dataset["train"].column_names
        dataset = dataset.map(
            preprocess,
            remove_columns=column_names,
        )

        # shuffle training set for better learning
        dataset["train"] = dataset["train"].shuffle(seed=42)

        self.train_dataset = dataset["train"]
        self.val_dataset = dataset["validation"]
        self.test_dataset = dataset["test"]

        print(f"train samples: {len(self.train_dataset)}")
        print(f"val samples: {len(self.val_dataset)}")
        print(f"test samples: {len(self.test_dataset)}")

    # ----------------------------------------------------------
    # compute metrics for evaluation
    # ----------------------------------------------------------
    def compute_metrics_with_generation(self, eval_dataset_sample):
        """Generate predictions and compute fraud detection metrics"""
        print("\ncomputing metrics with generation...")
        
        # sample small subset for evaluation (to save time)
        sample_size = min(50, len(eval_dataset_sample))
        indices = np.random.choice(len(eval_dataset_sample), sample_size, replace=False)
        
        true_labels = []
        pred_labels = []
        
        for idx in indices:
            example = eval_dataset_sample[int(idx)]
            
            # get true label
            true_label = example.get("label_id", 0)
            true_labels.append(true_label)
            
            # reconstruct input for generation
            input_ids = torch.tensor([example["input_ids"]]).to(self.device)
            
            # find where prompt ends (where labels != -100)
            labels = example["labels"]
            prompt_len = next((i for i, l in enumerate(labels) if l != -100), len(labels))
            prompt_ids = input_ids[:, :prompt_len]
            
            # generate prediction
            with torch.no_grad():
                outputs = self.model.generate(
                    prompt_ids,
                    max_new_tokens=100,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            
            # decode and extract label
            generated_text = self.tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=True)
            pred_label = extract_label(generated_text)
            pred_labels.append(pred_label)
        
        # compute metrics
        accuracy = accuracy_score(true_labels, pred_labels)
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, pred_labels, average='binary', zero_division=0
        )
        
        metrics = {
            "eval_accuracy": accuracy,
            "eval_precision": precision,
            "eval_recall": recall,
            "eval_f1": f1,
        }
        
        print(f"metrics: {metrics}")
        return metrics
    
    def compute_metrics(self, eval_pred):
        """Placeholder - actual metrics computed in callback"""
        return {}
    
    # ----------------------------------------------------------
    # training
    # ----------------------------------------------------------
    def train(self, num_epochs: int = 3, batch_size: int = 4):
        print("starting training...")

        # construct training kwargs and only pass those accepted by the
        # installed transformers.TrainingArguments to maintain compatibility
        training_kwargs = dict(
            output_dir=self.output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=8,  # increased from 4 to save memory
            learning_rate=2e-4,
            lr_scheduler_type="cosine",
            warmup_ratio=0.05,
            logging_steps=50,
            logging_first_step=True,
            eval_strategy="steps", 
            eval_steps=500,
            save_strategy="steps",
            save_steps=500,
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="eval_recall",
            greater_is_better=True,
            bf16=True,
            optim="paged_adamw_8bit",
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},  
            report_to="none",
            max_grad_norm=1.0,  # gradient clipping 
        )

        sig = inspect.signature(TrainingArguments.__init__)
        valid_kwargs = {k: v for k, v in training_kwargs.items() if k in sig.parameters}

        training_args = TrainingArguments(**valid_kwargs)

        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            padding=True,
            label_pad_token_id=-100,
            pad_to_multiple_of=8,
            return_tensors="pt",
        )


        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
            callbacks=[MetricsCallback(self, self.val_dataset)],
        )

        trainer.train()

        trainer.save_model(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)
        print(f"model saved to {self.output_dir}")

        print("\nevaluating on test set with generation...")
        test_metrics = self.compute_metrics_with_generation(self.test_dataset)
        print("test metrics:", test_metrics)
        
        # also compute standard eval loss
        test_loss_metrics = trainer.evaluate(self.test_dataset)
        print("test loss metrics:", test_loss_metrics)

    # ----------------------------------------------------------
    # full pipeline
    # ----------------------------------------------------------
    def run_full_training(self, dataset_fraction: float = 1.0):
        self.load_model_and_tokenizer(use_4bit=True)
        self.setup_lora()
        self.prepare_dataset(max_length=1024, dataset_fraction=dataset_fraction)  # reduced to 1024 for safety
        self.train(num_epochs=2, batch_size=2)


# ----------------------------------------------------------
# run script
# ----------------------------------------------------------
if __name__ == "__main__":
    trainer = FraudReasoningTrainer(
        model_name="microsoft/Phi-3.5-mini-instruct",
        train_path="data/fraud_train_clean_multi_task.jsonl",
        val_path="data/fraud_val_clean_multi_task.jsonl",
        test_path="data/fraud_test_clean_multi_task.jsonl",
        output_dir="models/phi-3.5-fraud-reasoning",
    )

    trainer.run_full_training(
        dataset_fraction=0.5,  # use only 5% of all datasets
    )
