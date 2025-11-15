#!/usr/bin/env python
"""
a02_split.py

Random split (config-driven) del dataset strutturato (all.txt) in
train / val / test.

Policy:
- Config unificato (configs/config.yaml) via a00_utils.load_config.
- Split random semplice, niente stratificazione.
- test_ratio = 1 - train_ratio - val_ratio (da cfg.data.split.*).
- Opzionale dedup 1:1 (cfg.data.split.dedup_exact).
- Opzionale shuffle prima dello split (cfg.data.split.shuffle_before_split).
- Lettura token di inizio carta da cfg.data.special_tokens[0] (via get_special_tokens).
- Salvataggio:
    - data/processed/train.txt
    - data/processed/val.txt
    - data/processed/test.txt
    - data/processed/split_stats.json
- Logging leggero su W&B se cfg.wandb.enabled è True (via init_wandb).

Esecuzione tipica (dal root della repo):

    python -m src.a02_split --config configs/config.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

from .a00_utils import load_config, get_special_tokens, init_wandb  # type: ignore

Block = str


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Random train/val/test split (config-driven) per MTG×MHA."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path al file di configurazione YAML.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Lettura blocchi da all.txt
# ---------------------------------------------------------------------------

def read_blocks(path: Path, start_token: str) -> List[Block]:
    """
    Legge `all.txt` e lo spezza in blocchi, uno per carta, mantenendo
    la struttura originale.

    Assunzione: ogni carta inizia con una riga che contiene SOLO il token
    `<|startofcard|>` (o equivalente da config).
    """
    if not path.is_file():
        raise FileNotFoundError(f"File all.txt non trovato: {path}")

    blocks: List[Block] = []
    current_lines: List[str] = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip() == start_token:
                # chiudi blocco corrente (se esiste) e inizia il prossimo
                if current_lines:
                    blocks.append("".join(current_lines))
                    current_lines = []
            current_lines.append(line)

    if current_lines:
        blocks.append("".join(current_lines))

    return blocks


def normalize_block(block: Block) -> str:
    """
    Normalizza un blocco per confronti di uguaglianza esatta (1:1):
    - trim spazi iniziali/finali
    - rimozione trailing spaces per riga
    """
    lines = [line.rstrip() for line in block.strip().splitlines()]
    return "\n".join(lines)


def deduplicate_blocks(blocks: List[Block]) -> Tuple[List[Block], int]:
    """
    Rimuove duplicati 1:1 in base alla versione normalizzata del blocco.

    Ritorna:
        unique_blocks: lista dei blocchi unici (ordine preservato)
        duplicates_removed: numero di duplicati eliminati
    """
    seen: Dict[str, int] = {}
    unique_blocks: List[Block] = []

    for block in blocks:
        key = normalize_block(block)
        if key in seen:
            seen[key] += 1
        else:
            seen[key] = 1
            unique_blocks.append(block)

    duplicates_removed = len(blocks) - len(unique_blocks)
    return unique_blocks, duplicates_removed


# ---------------------------------------------------------------------------
# Parsing campi header (color/type/rarity)
# ---------------------------------------------------------------------------

def extract_field(block: Block, field_name: str) -> str:
    """
    Estrae il valore di un campo dal blocco.

    Esempi:
        field_name="rarity"  -> riga "rarity: rare"
        field_name="color"   -> riga "color: W R"
        field_name="type"    -> riga "type: Legendary Creature | Human"

    Regole:
      - Per 'color' è ammesso valore vuoto (colorless).
      - Per gli altri campi, valore vuoto -> errore.
    """
    prefix = f"{field_name.lower()}:"
    for line in block.splitlines():
        stripped = line.strip()
        if stripped.lower().startswith(prefix):
            value = stripped.split(":", 1)[1].strip()
            if not value and field_name != "color":
                raise ValueError(f"Campo {field_name} vuoto in blocco:\n{block[:200]}")
            return value
    raise ValueError(f"Campo {field_name} mancante in blocco:\n{block[:200]}")


def compute_distributions(blocks: List[Block]) -> Dict[str, Dict[str, int]]:
    """
    Calcola distribuzioni per rarity, type, color su una lista di blocchi.

    Ritorna un dict del tipo:
        {
          "rarity": { "common": 123, ... },
          "type":   { "Creature | ...": 456, ... },
          "color":  { "W R": 78, "": 12, ... }
        }
    """
    rarity_counter: Counter = Counter()
    type_counter: Counter = Counter()
    color_counter: Counter = Counter()

    for block in blocks:
        rarity = extract_field(block, "rarity")
        type_ = extract_field(block, "type")
        color = extract_field(block, "color")

        rarity_counter[rarity] += 1
        type_counter[type_] += 1
        color_counter[color] += 1

    return {
        "rarity": dict(rarity_counter),
        "type": dict(type_counter),
        "color": dict(color_counter),
    }


# ---------------------------------------------------------------------------
# Split random semplice (no stratificazione)
# ---------------------------------------------------------------------------

def random_split(
    blocks: List[Block],
    train_ratio: float,
    val_ratio: float,
    shuffle_before_split: bool,
) -> Tuple[List[Block], List[Block], List[Block]]:
    """
    Split random semplice (no stratificazione).

    - Se `shuffle_before_split` è True: random.shuffle() in-place.
    - `test_ratio` viene calcolato come 1 - train_ratio - val_ratio.
    """
    if not 0 < train_ratio < 1:
        raise ValueError(f"train_ratio non valido: {train_ratio}")
    if not 0 <= val_ratio < 1:
        raise ValueError(f"val_ratio non valido: {val_ratio}")

    test_ratio = 1.0 - train_ratio - val_ratio
    if test_ratio <= 0:
        raise ValueError(
            f"train_ratio + val_ratio >= 1.0 (train={train_ratio}, val={val_ratio}); "
            f"test_ratio sarebbe {test_ratio}."
        )

    n = len(blocks)
    if shuffle_before_split:
        # seeds disabilitati per progetto (vedi decision log)
        random.shuffle(blocks)

    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))
    n_test = n - n_train - n_val

    if n_test < 0:
        overflow = -n_test
        # riduci prima val, poi train
        if n_val >= overflow:
            n_val -= overflow
        else:
            overflow -= n_val
            n_val = 0
            n_train = max(0, n_train - overflow)
        n_test = n - n_train - n_val

    train_blocks = blocks[:n_train]
    val_blocks = blocks[n_train:n_train + n_val]
    test_blocks = blocks[n_train + n_val:]

    assert len(train_blocks) + len(val_blocks) + len(test_blocks) == n

    return train_blocks, val_blocks, test_blocks


# ---------------------------------------------------------------------------
# Scrittura file + stats JSON + W&B
# ---------------------------------------------------------------------------

def write_blocks(path: Path, blocks: List[Block]) -> None:
    """
    Scrive i blocchi in un file di testo, così come sono, uno dopo l'altro.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for block in blocks:
            f.write(block)
            if not block.endswith("\n"):
                f.write("\n")


def save_stats(
    stats_path: Path,
    *,
    total_blocks: int,
    unique_blocks: int,
    duplicates_removed: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    n_train: int,
    n_val: int,
    n_test: int,
    global_dist: Dict[str, Dict[str, int]],
    train_dist: Dict[str, Dict[str, int]],
    val_dist: Dict[str, Dict[str, int]],
    test_dist: Dict[str, Dict[str, int]],
) -> None:
    """
    Salva uno small JSON con statistiche di split e distribuzioni.
    """
    payload = {
        "total_blocks": total_blocks,
        "unique_blocks": unique_blocks,
        "duplicates_removed": duplicates_removed,
        "ratios": {
            "train": train_ratio,
            "val": val_ratio,
            "test": test_ratio,
        },
        "split_sizes": {
            "train": n_train,
            "val": n_val,
            "test": n_test,
        },
        "global": global_dist,
        "by_split": {
            "train": train_dist,
            "val": val_dist,
            "test": test_dist,
        },
    }

    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def log_wandb(
    cfg: Dict[str, object],
    *,
    total_blocks: int,
    unique_blocks: int,
    duplicates_removed: int,
    n_train: int,
    n_val: int,
    n_test: int,
    global_dist: Dict[str, Dict[str, int]],
    train_dist: Dict[str, Dict[str, int]],
    val_dist: Dict[str, Dict[str, int]],
    test_dist: Dict[str, Dict[str, int]],
) -> None:
    """
    Logging leggero su Weights & Biases (se abilitato).
    Usiamo init_wandb da a00_utils per avere comportamento consistente.
    """
    run = init_wandb(cfg, run_name="split-a02")
    if run is None:
        return

    metrics = {
        "split.total_blocks": total_blocks,
        "split.unique_blocks": unique_blocks,
        "split.duplicates_removed": duplicates_removed,
        "split.size.train": n_train,
        "split.size.val": n_val,
        "split.size.test": n_test,
    }

    # Distribuzione rarity globale
    for rarity, count in global_dist.get("rarity", {}).items():
        metrics[f"split.global.rarity.{rarity}"] = count

    # Distribuzione rarity per split
    for split_name, dist in [
        ("train", train_dist),
        ("val", val_dist),
        ("test", test_dist),
    ]:
        for rarity, count in dist.get("rarity", {}).items():
            metrics[f"split.{split_name}.rarity.{rarity}"] = count

    run.log(metrics)  # type: ignore[attr-defined]
    run.finish()      # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(message)s",
    )
    logger = logging.getLogger("a02_split")

    cfg = load_config(args.config)
    data_cfg = cfg.get("data", {})

    processed_dir = Path(data_cfg.get("processed_dir", "data/processed")).resolve()
    all_path = processed_dir / "all.txt"

    split_cfg = data_cfg.get("split", {})
    train_ratio = float(split_cfg.get("train_ratio", 0.8))
    val_ratio = float(split_cfg.get("val_ratio", 0.1))
    test_ratio = 1.0 - train_ratio - val_ratio

    dedup_exact = bool(split_cfg.get("dedup_exact", True))
    shuffle_before_split = bool(split_cfg.get("shuffle_before_split", True))

    special_tokens = get_special_tokens(cfg)
    start_token = special_tokens[0] if special_tokens else "<|startofcard|>"

    logger.info(f"Config: {args.config}")
    logger.info(f"Processed dir: {processed_dir}")
    logger.info(f"all.txt path: {all_path}")
    logger.info(
        f"Split ratios -> train={train_ratio}, val={val_ratio}, "
        f"test={test_ratio} (calcolato)"
    )
    logger.info(f"Dedup 1:1 abilitato: {dedup_exact}")
    logger.info(f"Shuffle before split: {shuffle_before_split}")
    logger.info(f"Start token: {start_token}")

    # 1) Leggi blocchi grezzi
    blocks = read_blocks(all_path, start_token=start_token)
    total_blocks = len(blocks)
    logger.info(f"Blocchi totali letti: {total_blocks}")

    if total_blocks == 0:
        raise RuntimeError("Nessun blocco trovato in all.txt; controlla a01_preprocessor.")

    # 2) Dedup (se abilitato)
    if dedup_exact:
        blocks, duplicates_removed = deduplicate_blocks(blocks)
        logger.info(f"Dedup 1:1 eseguito: rimossi {duplicates_removed} duplicati.")
    else:
        duplicates_removed = 0
        logger.info("Dedup 1:1 DISABILITATO da config.")

    unique_blocks = len(blocks)
    logger.info(f"Blocchi unici dopo dedup: {unique_blocks}")

    # 3) Distribuzione globale (rarity/type/color)
    logger.info("Calcolo distribuzioni globali (rarity/type/color)...")
    global_dist = compute_distributions(blocks)

    logger.info("Distribuzione globale rarity:")
    for rarity, count in sorted(global_dist["rarity"].items(), key=lambda x: x[0]):
        logger.info(f"  - {rarity}: {count}")

    # 4) Split random semplice
    train_blocks, val_blocks, test_blocks = random_split(
        blocks=blocks,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        shuffle_before_split=shuffle_before_split,
    )

    n_train = len(train_blocks)
    n_val = len(val_blocks)
    n_test = len(test_blocks)

    logger.info("Dimensioni split:")
    logger.info(f"  - train: {n_train}")
    logger.info(f"  - val:   {n_val}")
    logger.info(f"  - test:  {n_test}")

    # 5) Distribuzioni per split
    logger.info("Calcolo distribuzioni per split (rarity/type/color)...")
    train_dist = compute_distributions(train_blocks)
    val_dist = compute_distributions(val_blocks)
    test_dist = compute_distributions(test_blocks)

    for split_name, dist in [
        ("train", train_dist),
        ("val", val_dist),
        ("test", test_dist),
    ]:
        logger.info(f"Distribuzione rarity [{split_name}]:")
        for rarity, count in sorted(dist["rarity"].items(), key=lambda x: x[0]):
            logger.info(f"  - {rarity}: {count}")

    # 6) Scrivi file
    train_path = processed_dir / "train.txt"
    val_path = processed_dir / "val.txt"
    test_path = processed_dir / "test.txt"

    write_blocks(train_path, train_blocks)
    write_blocks(val_path, val_blocks)
    write_blocks(test_path, test_blocks)

    logger.info(f"Scritto train: {train_path}")
    logger.info(f"Scritto val:   {val_path}")
    logger.info(f"Scritto test:  {test_path}")

    # 7) Stats JSON
    stats_path = processed_dir / "split_stats.json"
    save_stats(
        stats_path=stats_path,
        total_blocks=total_blocks,
        unique_blocks=unique_blocks,
        duplicates_removed=duplicates_removed,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        n_train=n_train,
        n_val=n_val,
        n_test=n_test,
        global_dist=global_dist,
        train_dist=train_dist,
        val_dist=val_dist,
        test_dist=test_dist,
    )
    logger.info(f"Statistiche split salvate in: {stats_path}")

    # 8) W&B lite
    log_wandb(
        cfg=cfg,
        total_blocks=total_blocks,
        unique_blocks=unique_blocks,
        duplicates_removed=duplicates_removed,
        n_train=n_train,
        n_val=n_val,
        n_test=n_test,
        global_dist=global_dist,
        train_dist=train_dist,
        val_dist=val_dist,
        test_dist=test_dist,
    )

    logger.info("a02_split: Done.")


if __name__ == "__main__":
    main()
