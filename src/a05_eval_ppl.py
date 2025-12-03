#!/usr/bin/env python
"""
a05_eval_ppl.py

Valutazione di un modello di language modeling causale (perplexity).
FIX: Supporto per modelli QLoRA/PEFT con token embeddings ridimensionati.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
from pathlib import Path
from typing import List, Dict, Any

import torch
from torch.utils.data import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)

# Import necessario per gestire gli adapter QLoRA
try:
    from peft import PeftModel, PeftConfig
except ImportError:
    PeftModel = None
    PeftConfig = None

from .a00_utils import load_config

Block = str


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Valutazione PPL di un modello CausalLM (MTG×MHA)."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path al file di configurazione YAML (default: configs/config.yaml).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helpers: lettura JSONL validati
# ---------------------------------------------------------------------------

def read_validated_blocks(path: Path, logger: logging.Logger) -> List[Block]:
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


def apply_subset_fraction(
    blocks: List[Block],
    subset_fraction: Any,
    *,
    label: str,
    logger: logging.Logger,
) -> List[Block]:
    n = len(blocks)
    if n == 0:
        logger.warning(f"{label}: dataset vuoto prima del subset.")
        return blocks

    if subset_fraction is None:
        logger.info(f"{label}: subset_fraction non definita, uso tutti i {n} esempi.")
        return blocks

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


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class CardDataset(Dataset):
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
        labels = input_ids.clone()

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(message)s",
    )
    logger = logging.getLogger("a05_eval_ppl")

    # ------------------------------------------------------------------ #
    # Config & paths
    # ------------------------------------------------------------------ #
    cfg = load_config(args.config)
    data_cfg = cfg.get("data", {})
    ppl_cfg = cfg.get("ppl_eval", {})
    wandb_cfg = cfg.get("wandb", {})

    if not ppl_cfg:
        raise RuntimeError("Sezione 'ppl_eval' mancante nel config.yaml.")

    run_name = ppl_cfg.get("run_name", None)
    if not run_name:
        raise RuntimeError("ppl_eval.run_name mancante nel config.yaml.")

    processed_dir = Path(data_cfg.get("processed_dir", "data/processed")).resolve()
    seq_len = int(data_cfg.get("seq_len", 512))

    split = ppl_cfg.get("split", "test")
    eval_subset_fraction = ppl_cfg.get("subset_fraction", None)
    eval_batch_size = int(ppl_cfg.get("batch_size", 4))

    split_jsonl = processed_dir / f"{split}_validated.jsonl"
    model_dir = Path("outputs") / run_name

    logger.info(f"Config:      {args.config}")
    logger.info(f"Run name:    {run_name}")
    logger.info(f"Model dir:   {model_dir}")
    logger.info(f"Processed:   {processed_dir}")
    logger.info(f"Split:       {split} -> {split_jsonl}")
    logger.info(f"Seq len:     {seq_len}")
    logger.info(f"Eval batch:  {eval_batch_size}")

    if not model_dir.is_dir():
        raise FileNotFoundError(f"Cartella modello non trovata: {model_dir}")

    # W&B Setup
    if wandb_cfg.get("enabled", False):
        os.environ["WANDB_PROJECT"] = wandb_cfg.get("project", "mtg-mha")
        logger.info("W&B abilitato via HuggingFace Trainer.")
        report_to = ["wandb"]
    else:
        os.environ["WANDB_MODE"] = "disabled"
        report_to = []
        logger.info("W&B disabilitato da config.")

    # ------------------------------------------------------------------ #
    # 1) Lettura blocchi
    # ------------------------------------------------------------------ #
    blocks = read_validated_blocks(split_jsonl, logger=logger)
    blocks = apply_subset_fraction(
        blocks,
        eval_subset_fraction,
        label=split,
        logger=logger,
    )
    logger.info(f"{split.capitalize()} blocks (used): {len(blocks)}")

    if len(blocks) == 0:
        raise RuntimeError(f"Dataset {split} vuoto dopo il subset.")

    # ------------------------------------------------------------------ #
    # 2) Caricamento modello & tokenizer (LOGICA FIXATA PER QLORA)
    # ------------------------------------------------------------------ #
    logger.info(f"Carico tokenizer da {model_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        # Mistral preferisce padding a sinistra, ma per evaluation batching va bene anche destra
        # se usiamo causal masking. Di default Trainer gestisce bene.

    # Controllo se è un modello PEFT/QLoRA (cerca adapter_config.json)
    is_peft = (model_dir / "adapter_config.json").exists()

    if is_peft:
        logger.info("Rilevato adapter PEFT/QLoRA.")
        if PeftConfig is None:
            raise ImportError("Libreria 'peft' non trovata. Installa con `pip install peft`.")

        # A. Leggiamo il config per sapere qual è il base model
        peft_config = PeftConfig.from_pretrained(str(model_dir))
        base_model_path = peft_config.base_model_name_or_path
        logger.info(f"Modello base identificato: {base_model_path}")

        # B. Carichiamo il modello BASE su CPU
        logger.info("Caricamento modello base...")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            # device_map="auto" NON usato qui per evitare conflitti col resize
        )

        # C. Resize Embeddings (Cruciale per il fix 32771 vs 32768)
        current_vocab_size = base_model.get_input_embeddings().weight.shape[0]
        if len(tokenizer) > current_vocab_size:
            logger.info(f"Resize embeddings: {current_vocab_size} -> {len(tokenizer)}")
            base_model.resize_token_embeddings(len(tokenizer), mean_resizing=False)

        # D. Carichiamo l'adapter
        logger.info(f"Caricamento pesi adapter da {model_dir}...")
        model = PeftModel.from_pretrained(base_model, str(model_dir))

        # E. Merge opzionale (per evaluation veloce), ma tenerlo separato va bene per il Trainer
        # Il Trainer sposterà il modello su GPU automaticamente.

    else:
        logger.info("Caricamento modello standard (Full Finetune o Base)...")
        model = AutoModelForCausalLM.from_pretrained(
            str(model_dir),
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        )
        model.resize_token_embeddings(len(tokenizer))

    # ------------------------------------------------------------------ #
    # 3) Dataset
    # ------------------------------------------------------------------ #
    eval_dataset = CardDataset(blocks, tokenizer, seq_len)

    # ------------------------------------------------------------------ #
    # 4) Evaluation con Trainer
    # ------------------------------------------------------------------ #
    output_dir = model_dir / f"eval_{split}"
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_eval_batch_size=eval_batch_size,
        do_train=False,
        do_eval=True,
        report_to=report_to,
        fp16=torch.cuda.is_available(),
        logging_steps=50,
        # Importante per evitare memory leak durante eval
        eval_accumulation_steps=10, 
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    # ------------------------------------------------------------------ #
    # 5) Calcolo PPL
    # ------------------------------------------------------------------ #
    logger.info(f"Inizio valutazione su split '{split}'...")
    metrics = trainer.evaluate()
    logger.info(f"Metrics raw: {metrics}")

    eval_loss = metrics.get("eval_loss", None)
    if eval_loss is not None:
        ppl = math.exp(eval_loss)
        metrics["perplexity"] = ppl
        logger.info(f"{split.capitalize()} loss: {eval_loss:.6f}")
        logger.info(f"{split.capitalize()} perplexity: {ppl:.4f}")
    else:
        logger.warning("eval_loss non presente nelle metriche.")

    # ------------------------------------------------------------------ #
    # 6) Salvataggio report
    # ------------------------------------------------------------------ #
    report_path = output_dir / f"ppl_{split}.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    logger.info(f"Report PPL salvato in {report_path}")


if __name__ == "__main__":
    main()