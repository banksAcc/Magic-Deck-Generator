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
- Legge gli iperparametri da config.training.gpt2 (model, lr, batch, ecc.)
  e i controlli globali da config.training (subset_fraction, eval_every, ecc.).
- Applica i limiti:
    - training.subset_fraction        (frazione globale per train/val)
    - training.max_train_examples
    - training.max_val_examples
  con priorità: max_* > subset_fraction > tutto il dataset.
- Costruisce tokenizer e modello GPT-2:
    - model_name da training.gpt2.model_name (default: "gpt2").
    - aggiunge i special tokens del progetto (cfg.data.special_tokens).
    - opzionale gradient checkpointing da training.gpt2.gradient_checkpointing.
- Usa HuggingFace Trainer per il fine-tuning.
- W&B opzionale tramite config.wandb.

Esecuzione tipica (dal root della repo):

    python -m src.a04_train_gpt2 --config configs/config.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Any

import torch
from torch.utils.data import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    set_seed,
)

from .a00_utils import load_config, get_special_tokens

Block = str


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
# Lettura blocchi da *_validated.jsonl
# ---------------------------------------------------------------------------

def read_validated_blocks(path: Path, logger: logging.Logger) -> List[Block]:
    """
    Legge un file *.jsonl prodotto dal validator (train_validated.jsonl, val_validated.jsonl, ...),
    e ritorna una lista di blocchi di testo (field 'raw').

    - Usa solo i record con "valid": true.
    - Ignora eventuali record senza 'raw' valido.
    """
    if not path.is_file():
        raise FileNotFoundError(f"File JSONL non trovato: {path}")

    blocks: List[Block] = []
    total_records = 0
    used_records = 0

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            total_records += 1
            rec = json.loads(line)

            if not rec.get("valid", False):
                continue

            raw = rec.get("raw", None)
            if isinstance(raw, str) and raw.strip():
                blocks.append(raw)
                used_records += 1

    logger.info(
        f"{path.name}: record totali={total_records}, "
        f"validi usati={used_records}, scartati={total_records - used_records}"
    )
    return blocks


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class CardDataset(Dataset):
    """
    Dataset semplice: ogni esempio è un blocco carta completo.
    Viene tokenizzato on-the-fly e troncato/paddato a seq_len.
    """

    def __init__(self, blocks: List[Block], tokenizer, seq_len: int):
        self.blocks = blocks
        self.tokenizer = tokenizer
        self.seq_len = seq_len

    def __len__(self) -> int:
        return len(self.blocks)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.blocks[idx]
        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.seq_len,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)
        labels = input_ids.clone()  # causal LM: predict next token su tutto

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


# ---------------------------------------------------------------------------
# Subset logic (max_* + subset_fraction)
# ---------------------------------------------------------------------------

def maybe_limit_blocks(
    blocks: List[Block],
    *,
    max_examples: Any,
    subset_fraction: Any,
    label: str,
    logger: logging.Logger,
) -> List[Block]:
    """
    Applica la logica di subset per un singolo split (train/val):

      - se max_examples non è None: usa al massimo max_examples blocchi
      - altrimenti, se subset_fraction non è None: usa quella frazione
      - altrimenti: usa tutti i blocchi

    Ritorna la lista eventualmente ridotta.
    """
    n = len(blocks)
    if n == 0:
        logger.warning(f"{label}: dataset vuoto prima del subset.")
        return blocks

    # Limite assoluto ha priorità
    if max_examples is not None:
        max_examples = int(max_examples)
        if max_examples < n:
            logger.info(f"{label}: uso solo i primi {max_examples} esempi su {n} (max_examples).")
            return blocks[:max_examples]
        else:
            logger.info(f"{label}: max_examples >= dataset, uso tutti i {n} esempi.")
            return blocks

    # Frazione globale (uguale per tutti i set)
    if subset_fraction is not None:
        frac = float(subset_fraction)
        if not (0.0 < frac <= 1.0):
            raise ValueError(f"{label}: subset_fraction deve essere in (0,1], trovato {frac}.")
        k = int(n * frac)
        k = max(1, k)
        if k < n:
            logger.info(f"{label}: uso solo una frazione {frac:.3f} -> {k} esempi su {n}.")
            return blocks[:k]
        else:
            logger.info(f"{label}: subset_fraction ~1, uso tutti i {n} esempi.")
            return blocks

    logger.info(f"{label}: nessun limite, uso tutti i {n} esempi.")
    return blocks


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
    # livello root training
    subset_fraction = training_root.get("subset_fraction", None)
    max_train_examples = training_root.get("max_train_examples", None)
    max_val_examples = training_root.get("max_val_examples", None)
    eval_every = int(training_root.get("eval_every_n_steps", 500))
    save_every = int(training_root.get("save_every_n_steps", 1000))
    run_name_prefix = training_root.get("run_name_prefix", "gpt2")
    max_steps = training_root.get("max_steps", None)
    seed = training_root.get("seed", None)

    # sottosezione gpt2
    base_model_name = gpt2_cfg.get("model_name", "gpt2")
    batch_size = int(gpt2_cfg.get("batch_size", 1))
    grad_accum = int(gpt2_cfg.get("gradient_accumulation_steps", 1))
    num_epochs = int(gpt2_cfg.get("num_epochs", 3))
    learning_rate = float(gpt2_cfg.get("lr_full", 2e-5))
    weight_decay = float(gpt2_cfg.get("weight_decay", 0.0))
    warmup_ratio = gpt2_cfg.get("warmup_ratio", None)
    warmup_steps = None  # usiamo il ratio, non i passi assoluti

    # ------------------------------------------------------------------ #
    # 3) Applicazione subset (max_* e subset_fraction globale)
    # ------------------------------------------------------------------ #
    train_blocks = maybe_limit_blocks(
        train_blocks,
        max_examples=max_train_examples,
        subset_fraction=subset_fraction,
        label="train",
        logger=logger,
    )
    val_blocks = maybe_limit_blocks(
        val_blocks,
        max_examples=max_val_examples,
        subset_fraction=subset_fraction,
        label="val",
        logger=logger,
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

    # Gradient checkpointing opzionale (da config.training.gpt2.gradient_checkpointing)
    if gpt2_cfg.get("gradient_checkpointing", False):
        logger.info("Abilito gradient checkpointing sul modello.")
        model.gradient_checkpointing_enable()

    # Seed opzionale
    if seed is not None:
        seed = int(seed)
        logger.info(f"Imposto seed globale a {seed}.")
        set_seed(seed)

    # ------------------------------------------------------------------ #
    # 5) Dataset & hyperparametri
    # ------------------------------------------------------------------ #
    train_dataset = CardDataset(train_blocks, tokenizer, seq_len)
    val_dataset = CardDataset(val_blocks, tokenizer, seq_len) if len(val_blocks) > 0 else None

    logger.info("--- Hyperparameters ---")
    logger.info(f"batch_size={batch_size}, grad_accum={grad_accum}, num_epochs={num_epochs}")
    logger.info(f"lr={learning_rate}, weight_decay={weight_decay}")
    logger.info(f"warmup_ratio={warmup_ratio}, max_steps={max_steps}")
    logger.info(f"subset_fraction={subset_fraction}")
    logger.info(f"eval_every={eval_every}, save_every={save_every}")

    run_name = f"{run_name_prefix}-gpt2"
    output_dir = Path("outputs") / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"output_dir={output_dir}")
    logger.info(f"HF Trainer report_to={report_to}")
    logger.info(f"run_name={run_name}")

    # Strategia di evaluation
    evaluation_strategy = "steps" if len(val_blocks) > 0 else "no"

    # Warmup: da ratio (se definito)
    if warmup_steps is not None:
        warmup_steps = int(warmup_steps)
        warmup_ratio_arg = 0.0
    else:
        warmup_steps = 0
        warmup_ratio_arg = float(warmup_ratio) if warmup_ratio is not None else 0.0

    # ------------------------------------------------------------------ #
    # 6) TrainingArguments & Trainer (con fallback per versioni HF vecchie)
    # ------------------------------------------------------------------ #
    base_kwargs = dict(
        output_dir=str(output_dir),
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        num_train_epochs=num_epochs,
        max_steps=int(max_steps) if max_steps is not None else -1,
        warmup_steps=warmup_steps,
        warmup_ratio=warmup_ratio_arg,
        logging_steps=max(1, eval_every // 5),
        save_steps=save_every,
        save_total_limit=3,
        report_to=report_to,
        fp16=torch.cuda.is_available(),
    )

    try:
        training_args = TrainingArguments(
            run_name=run_name,
            evaluation_strategy=evaluation_strategy,
            eval_steps=eval_every if evaluation_strategy == "steps" else None,
            **base_kwargs,
        )
    except TypeError as e:
        logger.warning(
            "TrainingArguments non supporta alcuni parametri "
            "(probabile versione diversa di transformers). "
            "Riprovo senza evaluation_strategy / eval_steps / run_name. "
            f"Dettagli: {e}"
        )
        training_args = TrainingArguments(**base_kwargs)

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
    # 7) Train + salvataggio
    # ------------------------------------------------------------------ #
    logger.info("Inizio training GPT-2 baseline...")
    trainer.train()
    logger.info("Training completato.")

    logger.info("Salvataggio modello e tokenizer finali...")
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    logger.info(f"Modello e tokenizer salvati in {output_dir}.")


if __name__ == "__main__":
    main()
