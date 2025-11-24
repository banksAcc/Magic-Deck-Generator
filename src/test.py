#!/usr/bin/env python
"""
test.py — Debug minimale: generazione di UNA singola carta

Cosa fa:
  - Carica config + mapping_seed + runtime validator.
  - Costruisce i prompt standard con build_prompts.
  - Prende il PRIMO prompt (prima riga di mapping).
  - Esegue UNA sola chiamata a model.generate sul prompt.
  - Stampa il prompt e l'output completo, più qualche check
    sui token speciali (<|startofcard|>, <|gen_card|>, <|endofcard|>).

Nota:
  - Non usa form precompilati (name/mana_cost/text/pt).
  - Non usa la pipeline di a06_generate_and_validate.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Any, Dict, List

import torch

from .a00_utils import load_config
from .a03_validate import build_validator_runtime
from .a06_generate_and_validate import (
    load_mapping_seed,
    build_prompts,
    load_model_and_tokenizer,
)


# ---------------------------------------------------------------------------
# CLI & logger
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Debug generazione di una singola carta (prompt standard)."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path al file di configurazione YAML.",
    )
    parser.add_argument(
        "--max_new_tokens_debug",
        type=int,
        default=256,
        help="Numero di token NUOVI da generare a partire dal prompt.",
    )
    return parser.parse_args()


def setup_logger() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger("debug_generation")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    logger = setup_logger()

    logger.info(f"Caricamento config da {args.config}")
    cfg: Dict[str, Any] = load_config(args.config)

    mapping_cfg = cfg.get("mapping", {})

    # Runtime validator (token speciali, regex, ecc.)
    runtime = build_validator_runtime(cfg)
    start_token: str = runtime["start_token"]
    gen_token: str = runtime["gen_token"]
    end_token: str = runtime["end_token"]

    logger.info(f"Start token: {start_token}")
    logger.info(f"Gen token:   {gen_token}")
    logger.info(f"End token:   {end_token}")

    # --- Mapping + prompt standard ---
    mapping_path = Path(
        mapping_cfg.get(
            "seed_path",
            os.path.join("src", "mapping", "mapping_seed.csv"),
        )
    ).resolve()

    mapping_rows = load_mapping_seed(mapping_path, logger)
    if not mapping_rows:
        raise RuntimeError(
            "mapping_seed.csv è vuoto, impossibile costruire un prompt di debug."
        )

    prompts: List[str] = build_prompts(mapping_rows, cfg, runtime, logger)
    if not prompts:
        raise RuntimeError(
            "build_prompts ha prodotto 0 prompt (controllare config mapping/generation)."
        )

    debug_prompt = prompts[0]

    logger.info("=== Prompt di DEBUG (prima riga di mapping) ===")
    print("----- PROMPT -----")
    print(debug_prompt)
    print("------------------")

    # --- Modello + tokenizer ---
    model, tokenizer, device, eos_token_id = load_model_and_tokenizer(
        cfg, runtime, logger
    )
    model.eval()

    # Assicuriamoci che il pad_token_id sia definito
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = eos_token_id

    # -----------------------------------------------------------------------
    # RUN UNICA: chiamata diretta a model.generate su UN solo prompt
    # -----------------------------------------------------------------------
    logger.info("=== RUN: model.generate diretto (singolo prompt) ===")

    gen_cfg = cfg.get("generation", {})
    temperature = float(gen_cfg.get("temperature", 0.8))
    top_p = float(gen_cfg.get("top_p", 0.9))
    repetition_penalty = float(gen_cfg.get("repetition_penalty", 1.1))

    # Prepariamo input_ids/attention_mask per una singola sequenza
    inputs = tokenizer(
        [debug_prompt],
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    seq_len = int(input_ids.shape[1])
    max_new_debug = int(args.max_new_tokens_debug)
    max_length = seq_len + max_new_debug

    logger.info(
        "Parametri RUN: temperature=%.3f, top_p=%.3f, repetition_penalty=%.3f, "
        "max_new_tokens_debug=%d -> max_length=%d",
        temperature,
        top_p,
        repetition_penalty,
        max_new_debug,
        max_length,
    )

    gen_kwargs: Dict[str, Any] = dict(
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        max_length=max_length,
        eos_token_id=int(eos_token_id),
        pad_token_id=int(tokenizer.pad_token_id),
    )

    # Chiamata "robusta" con fallback in caso di TypeError (versione transformers vecchia)
    try:
        with torch.no_grad():
            outputs_dbg = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **gen_kwargs,
            )
    except TypeError as e:
        logger.warning(
            "TypeError in model.generate con kwargs %s: %s",
            list(gen_kwargs.keys()),
            e,
        )
        # Fallback super-minimale: solo max_length + eos/pad
        fallback_kwargs = dict(
            max_length=max_length,
            eos_token_id=int(eos_token_id),
            pad_token_id=int(tokenizer.pad_token_id),
        )
        logger.info(
            "Riprovo RUN con kwargs ridotti: %s", list(fallback_kwargs.keys())
        )
        with torch.no_grad():
            outputs_dbg = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **fallback_kwargs,
            )

    out_dbg = tokenizer.decode(outputs_dbg[0], skip_special_tokens=False)

    print("\n===== OUTPUT (debug manuale, singola carta) =====")
    print(out_dbg)
    print("=================================================\n")

    # Check veloci sui token speciali
    print("OUTPUT - contiene '<|startofcard|>'? ", start_token in out_dbg)
    print("OUTPUT - contiene '<|gen_card|>'?    ", gen_token in out_dbg)
    print("OUTPUT - contiene '<|endofcard|>'?  ", end_token in out_dbg)

    logger.info("Debug completato. Ispeziona manualmente la carta generata.")


if __name__ == "__main__":
    main()
