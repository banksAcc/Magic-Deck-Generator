#!/usr/bin/env python
"""
a04_train_gpt2.py

Baseline training GPT-2 per MTG×MHA (config-driven).

- Legge config unificato (configs/config.yaml) via a00_utils.load_config.
- Usa SEMPRE i file validati:
    data/processed/train_validated.jsonl
    data/processed/val_validated.jsonl
  prodotti da a03_validate, e usa solo i record con "valid": true.
- Ogni carta è un esempio: un blocco intero (raw) con header + <|gen_card|> + parte generata.
- Iperparametri da config.training.gpt2
    - model_name, lr_full, num_epochs, batch_size, gradient_accumulation_steps, ecc.
- Controlli globali da config.training:
    - subset_fraction (unica frazione per train/val)
    - eval_every_n_steps
    - save_every_n_steps
    - run_name_prefix
- Nessun max_steps né seed: la durata del training è solo in termini di epoche.
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import inspect
from pathlib import Path
from typing import Dict, Any

import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)

from .a00_utils import (
    load_config,
    get_special_tokens,
    read_validated_blocks,
    CardDataset,
    apply_subset_fraction,
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GPT-2 baseline training for MTG×MHA (config-driven)."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path al file di configurazione YAML.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(message)s",
    )
    logger = logging.getLogger("a04_train_gpt2")

    # ------------------------------------------------------------------ #
    # Config & paths
    # ------------------------------------------------------------------ #
    cfg = load_config(args.config)
    data_cfg = cfg.get("data", {})
    training_root = cfg.get("training", {})   # contiene gpt2, mistral, subset_fraction, ecc.
    wandb_cfg = cfg.get("wandb", {})

    gpt2_cfg = training_root.get("gpt2", {})
    if not gpt2_cfg:
        raise RuntimeError("Config.training.gpt2 mancante o vuota: controlla configs/config.yaml")

    processed_dir = Path(data_cfg.get("processed_dir", "data/processed")).resolve()
    train_jsonl = processed_dir / "train_validated.jsonl"
    val_jsonl = processed_dir / "val_validated.jsonl"
    seq_len = int(data_cfg.get("seq_len", 512))

    special_tokens = get_special_tokens(cfg)

    logger.info(f"Config: {args.config}")
    logger.info(f"Processed dir: {processed_dir}")
    logger.info(f"Train JSONL:  {train_jsonl}")
    logger.info(f"Val JSONL:    {val_jsonl}")
    logger.info(f"Seq len:      {seq_len}")
    logger.info(f"Special tokens (da config): {special_tokens}")

    # ------------------------------------------------------------------ #
    # W&B tramite HF Trainer (se abilitato)
    # ------------------------------------------------------------------ #
    if wandb_cfg.get("enabled", False):
        os.environ["WANDB_PROJECT"] = wandb_cfg.get("project", "mtg-mha")
        entity = wandb_cfg.get("entity")
        if entity:
            os.environ["WANDB_ENTITY"] = entity
        tags = wandb_cfg.get("tags", [])
        if tags:
            os.environ["WANDB_TAGS"] = ",".join(tags)
        logger.info("W&B abilitato via HuggingFace Trainer.")
        report_to = ["wandb"]
    else:
        os.environ["WANDB_MODE"] = "disabled"
        report_to = []
        logger.info("W&B disabilitato da config.")

    # ------------------------------------------------------------------ #
    # 1) Lettura blocchi da train/val VALIDATI (solo valid=True)
    # ------------------------------------------------------------------ #
    train_blocks = read_validated_blocks(train_jsonl, logger=logger)
    val_blocks = read_validated_blocks(val_jsonl, logger=logger)

    logger.info(f"Train blocks (raw, valid): {len(train_blocks)}")
    logger.info(f"Val blocks   (raw, valid): {len(val_blocks)}")

    # ------------------------------------------------------------------ #
    # 2) Parametri da config.training
    # ------------------------------------------------------------------ #
    subset_fraction = training_root.get("subset_fraction", None)
    eval_every = int(training_root.get("eval_every_n_steps", 500))
    save_every = int(training_root.get("save_every_n_steps", 1000))
    run_name_prefix = training_root.get("run_name_prefix", "gpt2")

    base_model_name = gpt2_cfg.get("model_name", "gpt2")
    batch_size = int(gpt2_cfg.get("batch_size", 1))
    grad_accum = int(gpt2_cfg.get("gradient_accumulation_steps", 1))
    num_epochs = int(gpt2_cfg.get("num_epochs", 3))
    learning_rate = float(gpt2_cfg.get("lr_full", 2e-5))
    weight_decay = float(gpt2_cfg.get("weight_decay", 0.0))
    warmup_ratio = gpt2_cfg.get("warmup_ratio", None)
    gradient_checkpointing = bool(gpt2_cfg.get("gradient_checkpointing", False))

    # ------------------------------------------------------------------ #
    # 3) Applicazione subset_fraction a train/val
    # ------------------------------------------------------------------ #
    train_blocks = apply_subset_fraction(
        train_blocks, subset_fraction, label="train", logger=logger
    )
    val_blocks = apply_subset_fraction(
        val_blocks, subset_fraction, label="val", logger=logger
    )

    logger.info(f"Train blocks (used): {len(train_blocks)}")
    logger.info(f"Val blocks   (used): {len(val_blocks)}")

    if len(train_blocks) == 0:
        raise RuntimeError("Dataset di train vuoto dopo il subset: controlla il config.training.")

    # ------------------------------------------------------------------ #
    # 4) Tokenizer & modello GPT-2
    # ------------------------------------------------------------------ #
    logger.info(f"Base model: {base_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    additional_specials = [tok for tok in special_tokens if tok not in tokenizer.get_vocab()]
    if additional_specials:
        logger.info(f"Aggiungo {len(additional_specials)} special token al tokenizer.")
        tokenizer.add_special_tokens({"additional_special_tokens": additional_specials})

    model = AutoModelForCausalLM.from_pretrained(base_model_name)
    if additional_specials:
        model.resize_token_embeddings(len(tokenizer))

    if gradient_checkpointing:
        logger.info("Abilito gradient checkpointing sul modello (da config.training.gpt2).")
        model.gradient_checkpointing_enable()

    # ------------------------------------------------------------------ #
    # 5) Dataset & hyperparametri
    # ------------------------------------------------------------------ #
    train_dataset = CardDataset(train_blocks, tokenizer, seq_len)
    val_dataset = CardDataset(val_blocks, tokenizer, seq_len) if len(val_blocks) > 0 else None

    logger.info("--- Hyperparameters GPT-2 ---")
    logger.info(f"batch_size={batch_size}, grad_accum={grad_accum}, num_epochs={num_epochs}")
    logger.info(f"lr={learning_rate}, weight_decay={weight_decay}")
    logger.info(f"warmup_ratio={warmup_ratio}")
    logger.info(f"subset_fraction={subset_fraction}")
    logger.info(f"eval_every={eval_every}, save_every={save_every}")

    run_name = f"{run_name_prefix}-gpt2"
    output_dir = Path("outputs") / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"output_dir={output_dir}")
    logger.info(f"HF Trainer report_to={report_to}")
    logger.info(f"run_name={run_name}")

    # Strategia di evaluation: standard HF
    evaluation_strategy = "steps" if len(val_blocks) > 0 else "no"

    # Warmup: usiamo solo il ratio (standard HF)
    warmup_steps = 0
    warmup_ratio_arg = float(warmup_ratio) if warmup_ratio is not None else 0.0

    # ------------------------------------------------------------------ #
    # 6) TrainingArguments & Trainer (standard API)
    # ------------------------------------------------------------------ #
    training_sig = inspect.signature(TrainingArguments.__init__).parameters

    def _maybe_add(param: str, value: Any) -> None:
        if param in training_sig:
            training_kwargs[param] = value

    training_kwargs: Dict[str, Any] = {}
    _maybe_add("output_dir", str(output_dir))
    _maybe_add("run_name", run_name)
    _maybe_add("per_device_train_batch_size", batch_size)
    _maybe_add("per_device_eval_batch_size", batch_size)
    _maybe_add("gradient_accumulation_steps", grad_accum)
    _maybe_add("learning_rate", learning_rate)
    _maybe_add("weight_decay", weight_decay)
    _maybe_add("num_train_epochs", num_epochs)
    _maybe_add("warmup_steps", warmup_steps)
    _maybe_add("warmup_ratio", warmup_ratio_arg)
    _maybe_add("logging_steps", max(1, eval_every // 5))
    _maybe_add("save_steps", save_every)
    _maybe_add("save_total_limit", 3)
    _maybe_add("report_to", report_to)
    _maybe_add("fp16", torch.cuda.is_available())

    # Evaluation strategy: support old/new HF APIs.
    if "evaluation_strategy" in training_sig or "eval_strategy" in training_sig:
        key = "evaluation_strategy" if "evaluation_strategy" in training_sig else "eval_strategy"
        training_kwargs[key] = evaluation_strategy
        if evaluation_strategy == "steps" and "eval_steps" in training_sig:
            training_kwargs["eval_steps"] = eval_every
    elif "evaluate_during_training" in training_sig:
        training_kwargs["evaluate_during_training"] = evaluation_strategy != "no"
        if evaluation_strategy != "no" and "evaluate_during_training_steps" in training_sig:
            training_kwargs["evaluate_during_training_steps"] = eval_every

    training_args = TrainingArguments(**training_kwargs)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )

    # ------------------------------------------------------------------ #
    # 7) Train + valutazione + salvataggio
    # ------------------------------------------------------------------ #
    logger.info("Inizio training GPT-2 baseline...")
    trainer.train()
    logger.info("Training completato.")

    if val_dataset is not None:
        logger.info("Valutazione finale su validation set...")
        metrics = trainer.evaluate()
        logger.info(f"Validation metrics: {metrics}")
        if "eval_loss" in metrics:
            ppl = math.exp(metrics["eval_loss"])
            logger.info(f"Validation perplexity (exp(eval_loss)): {ppl:.4f}")

    logger.info("Salvataggio modello e tokenizer finali...")
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    logger.info(f"Modello e tokenizer salvati in {output_dir}.")


if __name__ == "__main__":
    main()
