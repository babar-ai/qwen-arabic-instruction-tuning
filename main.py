"""
Arabic LLM Fine-Tuning Engine
Domain Adaptation of Qwen2.5-7B-Instruct using QLoRA, SFT Prompt Loss Masking, and Fast Logit Evaluation.
Benchmark: PalmX 2025 (https://palmx.dlnlp.ai/)
"""

import os
import gc
import json
import logging
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    prepare_model_for_kbit_training,
    PeftModel,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("ArabicLLMFineTuner")


class ArabicLLMFineTuner:
    """
    Production-grade Fine-tuning & Evaluation Engine for Qwen2.5 on Arabic Cultural & Islamic QA.
    """

    def __init__(self, model_id: str = "Qwen/Qwen2.5-7B-Instruct", token: str = None):
        self.model_id = model_id
        self.token = token or os.getenv("HF_TOKEN")
        self.model = None
        self.tokenizer = None
        self.peft_model = None

    def setup_model_and_tokenizer(self, use_bf16: bool = True):
        """Setup 4-bit NF4 quantized base model and aligned tokenizer."""
        logger.info(f"Loading base model: {self.model_id}")

        compute_dtype = torch.bfloat16 if (use_bf16 and torch.cuda.is_bf16_supported()) else torch.float16
        logger.info(f"Using compute dtype: {compute_dtype}")

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            token=self.token,
            torch_dtype=compute_dtype,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            token=self.token
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.tokenizer.padding_side = "right"
        self.model = prepare_model_for_kbit_training(self.model)

        logger.info("Base model quantized and tokenizer configured successfully.")

    def setup_improved_lora_config(
        self,
        r: int = 32,
        lora_alpha: int = 64,
        lora_dropout: float = 0.05,
        target_modules: list = None,
        save_embed_modules: bool = False
    ):
        """
        Configure PEFT LoRA adapter.
        By default, excludes full-weight embed_tokens from modules_to_save to keep adapter compact (<150MB).
        """
        if target_modules is None:
            target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]

        modules_to_save = ["embed_tokens", "lm_head"] if save_embed_modules else None

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            bias="none",
            modules_to_save=modules_to_save,
        )

        self.peft_model = get_peft_model(self.model, lora_config)
        self.peft_model.print_trainable_parameters()
        return lora_config

    def load_and_prepare_data(
        self,
        task: str = "subtask2",
        validation_split: float = 0.2,
        augment_permutations: bool = False
    ):
        """
        Load datasets with caching safeguards, stratified splitting, and optional choice-permutation augmentation.
        """
        logger.info(f"Loading dataset for {task}")

        dataset_name = (
            "UBC-NLP/palmx_2025_subtask1_culture"
            if task == "subtask1"
            else "UBC-NLP/palmx_2025_subtask2_islamic"
        )

        try:
            train_data = load_dataset(dataset_name, split="train", download_mode="force_redownload")
        except Exception:
            train_data = load_dataset(dataset_name, split="train")

        logger.info(f"Raw training samples: {len(train_data)}")

        try:
            eval_data = load_dataset(dataset_name, split="dev")
            logger.info(f"Loaded existing dev split: {len(eval_data)} samples")
        except Exception:
            logger.info(f"Creating stratified {validation_split*100:.0f}% validation split from training data...")
            train_eval_split = train_data.train_test_split(
                test_size=validation_split,
                seed=42,
                shuffle=True,
                stratify_by_column="answer"
            )
            train_data = train_eval_split["train"]
            eval_data = train_eval_split["test"]

        if augment_permutations:
            logger.info("Applying option-permutation augmentation to training set...")
            train_data = self._augment_with_permutations(train_data)
            logger.info(f"Augmented training set size: {len(train_data)} samples")

        logger.info(f"Final dataset - Train: {len(train_data)}, Validation: {len(eval_data)}")
        return train_data, eval_data

    def _augment_with_permutations(self, dataset):
        """Augment multiple-choice questions by cyclically shifting options to remove position bias."""
        augmented_records = []
        labels = ["A", "B", "C", "D"]

        for item in dataset:
            augmented_records.append(dict(item))
            original_options = [item["A"], item["B"], item["C"], item["D"]]
            original_ans_idx = labels.index(item["answer"])

            # Add 1 cyclical shift
            shifted_options = original_options[1:] + original_options[:1]
            new_ans_idx = (original_ans_idx - 1) % 4
            shifted_record = {
                "id": f"{item.get('id', 'sample')}_perm1",
                "question": item["question"],
                "A": shifted_options[0],
                "B": shifted_options[1],
                "C": shifted_options[2],
                "D": shifted_options[3],
                "answer": labels[new_ans_idx]
            }
            augmented_records.append(shifted_record)

        import datasets
        return datasets.Dataset.from_list(augmented_records)

    def format_training_prompt(self, example: dict) -> tuple:
        """
        Formats example into (prompt_prefix, target_suffix) for precise SFT loss masking.
        """
        prompt_prefix = (
            f"<|im_start|>system\n"
            f"أنت خبير في الثقافة الإسلامية. أجب على السؤال متعدد الخيارات بتقديم حرف الإجابة الصحيحة فقط (A، B، C، أو D).<|im_end|>\n"
            f"<|im_start|>user\n"
            f"السؤال: {example['question']}\n\n"
            f"A. {example['A']}\n"
            f"B. {example['B']}\n"
            f"C. {example['C']}\n"
            f"D. {example['D']}\n\n"
            f"الجواب:<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        target_suffix = f"{example['answer']}<|im_end|>"
        return prompt_prefix, target_suffix

    def prepare_dataset_improved(self, dataset, max_length: int = 1024, mask_prompt_loss: bool = True):
        """
        Tokenizes dataset with SFT prompt loss masking (labels=-100 for prompt tokens).
        Ensures gradients update solely on predicting the correct answer letter.
        """
        logger.info(f"Tokenizing dataset (mask_prompt_loss={mask_prompt_loss})...")
        formatted_examples = []

        for i, item in enumerate(tqdm(dataset, desc="Processing examples")):
            try:
                prompt_prefix, target_suffix = self.format_training_prompt(item)

                if mask_prompt_loss:
                    prompt_ids = self.tokenizer.encode(prompt_prefix, add_special_tokens=False)
                    target_ids = self.tokenizer.encode(target_suffix, add_special_tokens=False)

                    full_input_ids = prompt_ids + target_ids
                    if len(full_input_ids) > max_length:
                        full_input_ids = full_input_ids[:max_length]

                    # Mask prompt tokens with -100, calculate loss only on answer
                    num_prompt = min(len(prompt_ids), len(full_input_ids))
                    labels = [-100] * num_prompt + full_input_ids[num_prompt:]
                else:
                    full_text = prompt_prefix + target_suffix
                    tokenized = self.tokenizer(
                        full_text,
                        truncation=True,
                        max_length=max_length,
                        return_tensors=None
                    )
                    full_input_ids = tokenized["input_ids"]
                    labels = list(full_input_ids)

                formatted_examples.append({
                    "input_ids": full_input_ids,
                    "attention_mask": [1] * len(full_input_ids),
                    "labels": labels
                })
            except Exception as e:
                logger.error(f"Error processing example {i}: {e}")
                continue

        return formatted_examples

    def create_improved_data_collator(self):
        """Dynamic batch collator with -100 padding for ignored label tokens."""
        def data_collator(features):
            max_length = max(len(f["input_ids"]) for f in features)
            batch = {"input_ids": [], "attention_mask": [], "labels": []}

            for feature in features:
                seq_len = len(feature["input_ids"])
                pad_len = max_length - seq_len

                batch["input_ids"].append(feature["input_ids"] + [self.tokenizer.pad_token_id] * pad_len)
                batch["attention_mask"].append(feature["attention_mask"] + [0] * pad_len)
                batch["labels"].append(feature["labels"] + [-100] * pad_len)

            return {k: torch.tensor(v, dtype=torch.long) for k, v in batch.items()}

        return data_collator

    def compute_metrics(self, eval_pred):
        """Calculate token-level evaluation accuracy on unmasked target tokens."""
        predictions, labels = eval_pred
        preds = np.argmax(predictions, axis=-1)

        mask = labels != -100
        filtered_preds = preds[mask]
        filtered_labels = labels[mask]

        if len(filtered_labels) == 0:
            return {"accuracy": 0.0, "eval_samples": 0}

        accuracy = (filtered_preds == filtered_labels).mean()
        return {
            "eval_accuracy": float(accuracy),
            "eval_samples": int(len(filtered_labels))
        }

    def fine_tune_improved(
        self,
        train_data,
        eval_data,
        output_dir: str = "./qwen_improved_finetuned",
        epochs: int = 5,
        lr: float = 1e-4,
        batch_size: int = 2,
        grad_accum: int = 8,
        use_bf16: bool = True
    ):
        """Execute SFT training loop with early stopping and automatic checkpoint selection."""
        train_dataset = self.prepare_dataset_improved(train_data, mask_prompt_loss=True)
        eval_dataset = self.prepare_dataset_improved(eval_data, mask_prompt_loss=True)

        logger.info(f"Prepared training samples: {len(train_dataset)}")
        logger.info(f"Prepared validation samples: {len(eval_dataset)}")

        has_bf16 = use_bf16 and torch.cuda.is_bf16_supported()

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size * 2,
            gradient_accumulation_steps=grad_accum,
            warmup_ratio=0.1,
            learning_rate=lr,
            weight_decay=0.01,
            bf16=has_bf16,
            fp16=not has_bf16,
            logging_steps=25,
            save_steps=100,
            eval_steps=100,
            eval_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_accuracy",
            greater_is_better=True,
            remove_unused_columns=False,
            dataloader_pin_memory=False,
            gradient_checkpointing=True,
            report_to="none",
            save_total_limit=3,
            lr_scheduler_type="cosine",
            max_grad_norm=1.0,
        )

        trainer = Trainer(
            model=self.peft_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=self.create_improved_data_collator(),
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )

        logger.info("Beginning fine-tuning...")
        trainer.train()

        logger.info(f"Saving best adapter and tokenizer to {output_dir}...")
        trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        return trainer

    def evaluate_with_baseline_format(self, eval_data, max_length: int = 1024):
        """
        Fast Log-Likelihood Evaluation via single forward pass per question (4x speedup).
        Directly extracts next-token logits for choices ['A', 'B', 'C', 'D'].
        """
        if self.peft_model is None:
            raise ValueError("No model loaded for evaluation.")

        self.peft_model.eval()
        correct_predictions = 0
        total_questions = len(eval_data)
        submission_data = []

        # Find the token IDs for ' A', ' B', ' C', ' D' in Qwen tokenizer
        choice_token_ids = []
        for choice in ["A", "B", "C", "D"]:
            tok = self.tokenizer.encode(f" {choice}", add_special_tokens=False)[-1]
            choice_token_ids.append(tok)

        logger.info(f"Evaluating on {total_questions} validation questions via fast single-pass logit extraction...")

        for i, example in enumerate(tqdm(eval_data, desc="Evaluating")):
            try:
                question = example["question"]
                options = [example["A"], example["B"], example["C"], example["D"]]

                prompt = (
                    f"{question}\n\n"
                    f"A. {options[0]}\n"
                    f"B. {options[1]}\n"
                    f"C. {options[2]}\n"
                    f"D. {options[3]}\n\n"
                    f"الجواب:"
                )

                prompt_inputs = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=max_length
                ).to(self.peft_model.device)

                with torch.no_grad():
                    outputs = self.peft_model(**prompt_inputs)
                    # Next-token logits at the last position of prompt
                    last_token_logits = outputs.logits[0, -1, :]
                    choice_scores = [last_token_logits[tok_id].item() for tok_id in choice_token_ids]

                predicted_idx = int(np.argmax(choice_scores))
                predicted_answer = ["A", "B", "C", "D"][predicted_idx]
                is_correct = predicted_answer == example["answer"]

                submission_data.append({
                    "id": example.get("id", f"sample_{i}"),
                    "prediction": predicted_answer,
                    "correct_answer": example["answer"],
                    "is_correct": is_correct,
                    "scores": choice_scores
                })

                if is_correct:
                    correct_predictions += 1

            except Exception as e:
                logger.error(f"Error evaluating sample {i}: {e}")
                submission_data.append({
                    "id": example.get("id", f"sample_{i}"),
                    "prediction": "A",
                    "correct_answer": example["answer"],
                    "is_correct": False,
                    "scores": [0.0, 0.0, 0.0, 0.0]
                })

            if i % 100 == 0:
                torch.cuda.empty_cache()

        accuracy = (correct_predictions / total_questions) * 100.0
        logger.info(f"\n=== Evaluation Accuracy: {accuracy:.2f}% ({correct_predictions}/{total_questions}) ===")
        return submission_data, accuracy

    def evaluate_with_tta(self, eval_data, max_length: int = 1024):
        """
        Test-Time Augmentation (TTA) via 4 cyclical choice permutations.
        Averages softmax probabilities across permutations to cancel position bias.
        """
        self.peft_model.eval()
        correct_predictions = 0
        total_questions = len(eval_data)
        submission_data = []
        labels = ["A", "B", "C", "D"]

        choice_token_ids = [self.tokenizer.encode(f" {c}", add_special_tokens=False)[-1] for c in labels]
        logger.info(f"Running Test-Time Augmentation (TTA 4x) on {total_questions} questions...")

        for i, example in enumerate(tqdm(eval_data, desc="TTA Evaluation")):
            try:
                base_options = [example["A"], example["B"], example["C"], example["D"]]
                aggregated_probs = np.zeros(4)

                # Iterate through 4 cyclical permutations
                for shift in range(4):
                    perm_options = base_options[shift:] + base_options[:shift]
                    prompt = (
                        f"{example['question']}\n\n"
                        f"A. {perm_options[0]}\n"
                        f"B. {perm_options[1]}\n"
                        f"C. {perm_options[2]}\n"
                        f"D. {perm_options[3]}\n\n"
                        f"الجواب:"
                    )
                    inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length).to(self.peft_model.device)

                    with torch.no_grad():
                        logits = self.peft_model(**inputs).logits[0, -1, :]
                        raw_scores = np.array([logits[tok_id].item() for tok_id in choice_token_ids])
                        probs = np.exp(raw_scores - np.max(raw_scores))
                        probs = probs / probs.sum()

                    # Map probabilities back to original option index
                    for perm_idx, prob in enumerate(probs):
                        orig_idx = (perm_idx + shift) % 4
                        aggregated_probs[orig_idx] += prob

                predicted_idx = int(np.argmax(aggregated_probs))
                predicted_answer = labels[predicted_idx]
                is_correct = (predicted_answer == example["answer"])

                submission_data.append({
                    "id": example.get("id", f"sample_{i}"),
                    "prediction": predicted_answer,
                    "correct_answer": example["answer"],
                    "is_correct": is_correct,
                    "aggregated_probs": aggregated_probs.tolist()
                })

                if is_correct:
                    correct_predictions += 1
            except Exception as e:
                logger.error(f"Error in TTA for sample {i}: {e}")

        accuracy = (correct_predictions / total_questions) * 100.0
        logger.info(f"\n=== TTA Evaluation Accuracy: {accuracy:.2f}% ===")
        return submission_data, accuracy

    def analyze_errors(self, results: list, eval_data):
        """Comprehensive error analysis: class distribution, confusion matrix, and low-confidence edge cases."""
        error_analysis = {
            "by_answer": {"A": 0, "B": 0, "C": 0, "D": 0},
            "confusion_matrix": {},
            "difficult_questions": []
        }

        for i, result in enumerate(results):
            correct_ans = result["correct_answer"]
            pred_ans = result["prediction"]

            if not result["is_correct"]:
                error_analysis["by_answer"][correct_ans] = error_analysis["by_answer"].get(correct_ans, 0) + 1

                if correct_ans not in error_analysis["confusion_matrix"]:
                    error_analysis["confusion_matrix"][correct_ans] = {}
                error_analysis["confusion_matrix"][correct_ans][pred_ans] = (
                    error_analysis["confusion_matrix"][correct_ans].get(pred_ans, 0) + 1
                )

                scores = result.get("scores")
                if scores:
                    sorted_scores = sorted(scores)
                    confidence = sorted_scores[-1] - sorted_scores[-2]
                    if confidence < 1.0:
                        error_analysis["difficult_questions"].append({
                            "id": result["id"],
                            "question": eval_data[i]["question"][:200] + "...",
                            "confidence": float(confidence),
                            "predicted": pred_ans,
                            "correct": correct_ans
                        })

        return error_analysis


# Backwards compatibility alias
ImprovedQwenFineTuner = ArabicLLMFineTuner


def main():
    parser = argparse.ArgumentParser(description="Arabic LLM Fine-tuning & Evaluation Runner")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--task", type=str, default="subtask2", choices=["subtask1", "subtask2"])
    parser.add_argument("--output_dir", type=str, default="./arabic_llm_qwen_finetuned")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--r", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--augment", action="store_true", help="Enable option permutation augmentation")
    parser.add_argument("--use_tta", action="store_true", help="Run Test-Time Augmentation on evaluation")
    args = parser.parse_args()

    ft = ArabicLLMFineTuner(model_id=args.model_id)

    try:
        ft.setup_model_and_tokenizer(use_bf16=True)
        ft.setup_improved_lora_config(r=args.r, lora_alpha=args.lora_alpha)

        train_data, eval_data = ft.load_and_prepare_data(
            task=args.task,
            validation_split=0.2,
            augment_permutations=args.augment
        )

        ft.fine_tune_improved(
            train_data,
            eval_data,
            output_dir=args.output_dir,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            grad_accum=args.grad_accum
        )

        if args.use_tta:
            results, accuracy = ft.evaluate_with_tta(eval_data)
        else:
            results, accuracy = ft.evaluate_with_baseline_format(eval_data)

        error_analysis = ft.analyze_errors(results, eval_data)

        # Save artifacts
        pd.DataFrame(results).to_csv(f"improved_results_{accuracy:.1f}acc.csv", index=False)
        with open(f"error_analysis_{accuracy:.1f}acc.json", "w", encoding="utf-8") as f:
            json.dump(error_analysis, f, indent=2, ensure_ascii=False)

        logger.info(f"Target Baseline (NileChat-3B): 69.5%")
        logger.info(f"Model Accuracy Achieved: {accuracy:.2f}%")
    finally:
        torch.cuda.empty_cache()
        gc.collect()


if __name__ == "__main__":
    main()
