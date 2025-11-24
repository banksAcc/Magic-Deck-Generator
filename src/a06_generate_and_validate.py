#!/usr/bin/env python
"""
a06_generate_and_validate.py

Generazione condizionata + validazione per blocchi carta MTG×MHA.

Flow:
  1. Legge config.yaml (data, generation, mapping, validator, constants).
  2. Carica mapping.seed_path (seed tematizzazione MHA -> MTG).
  3. Costruisce prompt condizionati nel formato Standard fino a <|gen_card|>.
  4. Carica il modello finetunato (GPT-2 o Mistral + LoRA).
  5. Genera un batch di carte (testo grezzo, uno o più blocchi).
  6. Post-process: un blocco per prompt, tagliato al primo <|endofcard|>.
  7. Valida tutti i blocchi usando a03_validate:
       - in modalità "soft" (di default) per la generazione,
       - eventualmente "hard" se configurato.
  8. Scrive:
       - batch_YYYYMMDD_HHMMSS.txt                    (blocchi grezzi)
       - batch_YYYYMMDD_HHMMSS_validated[_soft].jsonl (risultati validazione)
  9. Logga stats riassuntive + W&B (opzionale).

Config attesa (estratto rilevante):

mapping:
  enabled: true
  seed_path: "src/mapping/mapping_seed.csv"
  keywords_path: "src/mapping/keywords.json"
  default_theme: "Generic"
  default_character: "N/A"

data:
  special_tokens:
    - "<|startofcard|>"
    - "<|gen_card|>"
    - "<|endofcard|>"

generation:
  model_type: "gpt2"        # "mistral" | "gpt2"
  base_model_name: "gpt2"   # o "mistralai/Mistral-7B-Instruct-v0.2"
  checkpoint_path: "outputs/checkpoints/..."  # dir dell’adapter o FT
  device: "cuda"            # "cuda" | "cpu" | "auto"
  batch_size: 8
  samples_per_mapping: 100  # opzionale
  num_samples: 1500
  temperature: 0.8
  top_p: 0.9
  repetition_penalty: 1.1
  max_new_tokens: 160
  eos_token: "<|endofcard|>"
  output_dir: "outputs/generations"  # opzionale, default se assente

validator:
  # modalità di validazione da usare in GENERAZIONE (default: "soft")
  mode_generation: "soft"   # "soft" | "hard"
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Any, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from .a00_utils import (
    load_config,
    get_special_tokens,
    init_wandb,
)
from .a03_validate import (
    build_validator_runtime,
    validate_blocks,
    write_validation_results_jsonl,
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class MappingSeedRow:
    character_or_faction: str
    color_id: str
    default_type: str
    rarity_hint: str
    mechanics_hints: str
    notes: str


Block = str


# ---------------------------------------------------------------------------
# CLI & logger
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="a06 — Generazione + validazione per blocchi carta MTG×MHA."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path al file di configurazione YAML.",
    )
    return parser.parse_args()


def setup_logger() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger("a06_generate_and_validate")


# ---------------------------------------------------------------------------
# Helpers: mapping, prompts, modello
# ---------------------------------------------------------------------------

def load_mapping_seed(path: Path, logger: logging.Logger) -> List[MappingSeedRow]:
    """
    Carica il CSV di mapping MHA -> MTG.

    Colonne richieste:
        character_or_faction,color_id,default_type,rarity_hint,mechanics_hints,notes
    """
    import csv

    if not path.is_file():
        raise FileNotFoundError(f"mapping_seed.csv non trovato in: {path}")

    rows: List[MappingSeedRow] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = set(reader.fieldnames or [])

        required_cols = {
            "character_or_faction",
            "color_id",
            "default_type",
            "rarity_hint",
            "mechanics_hints",
            "notes",
        }

        missing = required_cols - fieldnames
        extra = fieldnames - required_cols

        if missing:
            raise ValueError(
                "mapping_seed.csv manca delle colonne richieste: "
                f"{sorted(missing)} (trovate: {sorted(fieldnames)})"
            )

        if extra:
            logger.warning(
                "mapping_seed.csv contiene colonne aggiuntive non utilizzate: "
                f"{sorted(extra)}"
            )

        for r in reader:
            rows.append(
                MappingSeedRow(
                    character_or_faction=r["character_or_faction"].strip(),
                    color_id=r["color_id"].strip(),
                    default_type=r["default_type"].strip(),
                    rarity_hint=r["rarity_hint"].strip(),
                    mechanics_hints=r.get("mechanics_hints", "").strip(),
                    notes=r.get("notes", "").strip(),
                )
            )

    logger.info(f"Caricate {len(rows)} righe da {path}")
    return rows


def _compute_samples_per_row(
    n_rows: int,
    gen_cfg: Dict[str, Any],
    logger: logging.Logger,
) -> List[int]:
    """
    Decide quanti prompt generare per ciascuna riga del mapping.

    Priorità:
      - se `samples_per_mapping` > 0: proviamo a usarlo
        (totale = n_rows * samples_per_mapping, poi capped da num_samples se presente)
      - altrimenti: distribuiamo `num_samples` in modo (quasi) uniforme.
    """
    if n_rows <= 0:
        raise ValueError("mapping_seed.csv è vuoto (nessuna riga).")

    samples_per_mapping = int(gen_cfg.get("samples_per_mapping", 0))
    num_samples = int(gen_cfg.get("num_samples", 0))

    if samples_per_mapping > 0:
        total = n_rows * samples_per_mapping
        if num_samples > 0 and num_samples < total:
            logger.info(
                f"samples_per_mapping={samples_per_mapping} -> totale={total}, "
                f"ma num_samples={num_samples} => cap a num_samples."
            )
            # ridistribuiamo uniformemente num_samples
            base = num_samples // n_rows
            rem = num_samples % n_rows
            counts = [base + (1 if i < rem else 0) for i in range(n_rows)]
        else:
            counts = [samples_per_mapping] * n_rows
    else:
        # niente samples_per_mapping: usiamo solo num_samples
        if num_samples <= 0:
            # fallback: almeno 1 per riga
            logger.warning(
                "generation.num_samples non impostato o <=0; uso 1 prompt per riga di mapping."
            )
            counts = [1] * n_rows
        else:
            base = num_samples // n_rows
            rem = num_samples % n_rows
            if base == 0:
                # meno sample che righe: alcune righe avranno 1, altre 0
                counts = [1 if i < num_samples else 0 for i in range(n_rows)]
            else:
                counts = [base + (1 if i < rem else 0) for i in range(n_rows)]

    logger.info(
        f"Distribuzione prompt per riga mapping: "
        f"totale={sum(counts)}, n_rows={n_rows}, counts={counts}"
    )
    return counts


def build_prompts(
    mapping_rows: List[MappingSeedRow],
    cfg: Dict[str, Any],
    runtime: Dict[str, Any],
    logger: logging.Logger,
) -> List[str]:
    """
    Costruisce i prompt condizionati nel formato Standard fino a <|gen_card|>.

    Formato:

      <|startofcard|>
      theme: <mapping.default_theme>
      character: <character_or_faction>
      color: <color_id>
      type: <default_type>
      rarity: <rarity_hint>
      <|gen_card|>

    Dopo <|gen_card|> il modello completerà con name/mana_cost/text/pt/.../<|endofcard|>.
    """
    gen_cfg = cfg.get("generation", {})
    mapping_cfg = cfg.get("mapping", {})

    start_token = runtime["start_token"]
    gen_token = runtime["gen_token"]

    # Tema globale per il batch (es. "My Hero Academia" in questo progetto)
    theme = mapping_cfg.get("default_theme", "Generic")

    counts = _compute_samples_per_row(len(mapping_rows), gen_cfg, logger)

    prompts: List[str] = []
    for row, n_for_row in zip(mapping_rows, counts):
        if n_for_row <= 0:
            continue

        header = (
            f"{start_token}\n"
            f"theme: {theme}\n"
            f"character: {row.character_or_faction}\n"
            f"color: {row.color_id}\n"
            f"type: {row.default_type}\n"
            f"rarity: {row.rarity_hint}\n"
            f"{gen_token}\n"
        )

        prompts.extend([header] * n_for_row)

    logger.info(f"Costruiti {len(prompts)} prompt da {len(mapping_rows)} righe di mapping.")
    return prompts


def _resolve_device(device_str: str) -> torch.device:
    device_str = (device_str or "auto").lower()
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_str == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    if device_str == "cpu":
        return torch.device("cpu")
    # fallback robusto
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model_and_tokenizer(
    cfg: Dict[str, Any],
    runtime: Dict[str, Any],
    logger: logging.Logger,
) -> Tuple[Any, Any, torch.device, int]:
    """
    Carica modello + tokenizer per la generazione.

    Config:
      generation.model_type: "gpt2" | "mistral"
      generation.base_model_name
      generation.checkpoint_path
      generation.device
      generation.eos_token

    Usa gli special tokens dal config (get_special_tokens) e l'end_token
    dal runtime validator per derivare l'eos_token_id.
    """
    gen_cfg = cfg.get("generation", {})
    model_type = gen_cfg.get("model_type", "gpt2").lower()
    base_model_name = gen_cfg.get("base_model_name")
    checkpoint_path = gen_cfg.get("checkpoint_path") or base_model_name
    device = _resolve_device(gen_cfg.get("device", "auto"))

    if not base_model_name:
        raise ValueError("generation.base_model_name deve essere impostato in config.yaml")

    # Tokenizer
    logger.info(f"Caricamento tokenizer da base_model_name={base_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=True)

    # special tokens (start/gen/end)
    special_tokens = get_special_tokens(cfg)
    if special_tokens:
        tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})

    # gestiamo pad_token (per GPT-2 spesso coincide con eos)
    if tokenizer.pad_token is None:
        # proviamo a usare eos_token del tokenizer, altrimenti end_token da runtime
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": runtime["end_token"]})

    # Modello
    logger.info(
        f"Caricamento modello (type={model_type}) "
        f"da checkpoint={checkpoint_path} (base={base_model_name}), device={device}"
    )

    if model_type == "gpt2":
        # Assumiamo che checkpoint_path punti a un modello HF già finetunato
        model = AutoModelForCausalLM.from_pretrained(checkpoint_path)
        model.resize_token_embeddings(len(tokenizer))

    elif model_type == "mistral":
        # Caso Mistral + (eventuale) LoRA. Se è stato usato PEFT, carichiamo l'adapter.
        try:
            from peft import PeftModel  # type: ignore
        except ImportError as e:
            raise ImportError(
                "model_type='mistral' richiede il pacchetto 'peft' installato "
                "(pip install peft)."
            ) from e

        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
        )
        model = PeftModel.from_pretrained(base_model, checkpoint_path)
        model.resize_token_embeddings(len(tokenizer))
    else:
        raise ValueError(f"generation.model_type non supportato: {model_type}")

    model.to(device)
    model.eval()

    # EOS token id
    eos_token_str = gen_cfg.get("eos_token", runtime["end_token"])
    eos_token_id = tokenizer.convert_tokens_to_ids(eos_token_str)
    if eos_token_id == tokenizer.unk_token_id:
        logger.warning(
            f"eos_token '{eos_token_str}' non trovato nel vocab "
            f"(id=unk={eos_token_id}). Controllare special_tokens / config."
        )

    logger.info(
        f"Modello caricato su device={device}, eos_token={eos_token_str} (id={eos_token_id})"
    )
    return model, tokenizer, device, eos_token_id


# ---------------------------------------------------------------------------
# Helpers: generazione, post-process, salvataggio
# ---------------------------------------------------------------------------

def generate_with_model(
    model,
    tokenizer,
    device: torch.device,
    eos_token_id: int,
    prompts: List[str],
    cfg: Dict[str, Any],
    logger: logging.Logger,
) -> List[Block]:
    """
    Genera sequenze a partire da una lista di prompt.

    Ritorna una lista di stringhe (grezze), che includono sia il prompt che
    la parte generata.

    Implementazione allineata al test di debug:
      - per ogni batch calcoliamo la lunghezza massima di input_ids
      - max_length = max_input_len + max_new_tokens (da config)
      - chiamiamo model.generate con sampling + eos/pad
      - fallback minimal in caso di TypeError (versioni transformers vecchie)
    """
    gen_cfg = cfg.get("generation", {})

    temperature = float(gen_cfg.get("temperature", 0.8))
    top_p = float(gen_cfg.get("top_p", 0.9))
    repetition_penalty = float(gen_cfg.get("repetition_penalty", 1.1))
    max_new_tokens_cfg = int(gen_cfg.get("max_new_tokens", 160))
    batch_size = int(gen_cfg.get("batch_size", 4))

    logger.info(
        "Parametri di generazione (config): "
        f"temperature={temperature}, top_p={top_p}, "
        f"repetition_penalty={repetition_penalty}, "
        f"max_new_tokens={max_new_tokens_cfg}, batch_size={batch_size}"
    )

    all_outputs: List[Block] = []

    for start in range(0, len(prompts), batch_size):
        batch_prompts = prompts[start: start + batch_size]

        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        # Calcoliamo la lunghezza massima di input nel batch
        seq_len = int(input_ids.shape[1])
        max_length = seq_len + max_new_tokens_cfg

        logger.info(
            "Batch %d: seq_len=%d, max_new_tokens=%d -> max_length=%d",
            (start // batch_size) + 1,
            seq_len,
            max_new_tokens_cfg,
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

        # Chiamata robusta: se alcuni kwargs non sono supportati, facciamo fallback
        try:
            with torch.no_grad():
                outputs = model.generate(
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
            fallback_kwargs = dict(
                max_length=max_length,
                eos_token_id=int(eos_token_id),
                pad_token_id=int(tokenizer.pad_token_id),
            )
            logger.info(
                "Riprovo batch con kwargs ridotti: %s", list(fallback_kwargs.keys())
            )
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    **fallback_kwargs,
                )

        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=False)
        all_outputs.extend(decoded)

        logger.info(
            "Batch generato %d / %d",
            (start // batch_size) + 1,
            (len(prompts) + batch_size - 1) // batch_size,
        )

    logger.info(f"Sequenze generate totali: {len(all_outputs)}")
    return all_outputs


def postprocess_generations(
    raw_generations: List[Block],
    start_token: str,
    end_token: str,
    logger: logging.Logger,
) -> List[Block]:
    """
    Trasforma le sequenze grezze in blocchi "puliti" per il validator:

      - taglia da prima occorrenza di start_token
      - tronca alla prima occorrenza di end_token (inclusa)
      - aggiunge newline finale se mancante

    Se uno dei token non è presente, il blocco viene lasciato com'è
    (il validator lo marchierà invalid).
    """
    blocks: List[Block] = []
    missing_start = 0
    missing_end = 0

    for text in raw_generations:
        t = text

        # 1) da start_token
        idx_start = t.find(start_token)
        if idx_start != -1:
            t = t[idx_start:]
        else:
            missing_start += 1

        # 2) fino a end_token (incluso)
        idx_end = t.find(end_token)
        if idx_end != -1:
            idx_end += len(end_token)
            t = t[:idx_end]
        else:
            missing_end += 1

        # 3) newline finale (per avere <|endofcard|> su una riga)
        if not t.endswith("\n"):
            t = t + "\n"

        blocks.append(t)

    if missing_start > 0:
        logger.warning(
            f"{missing_start} blocchi non contenevano il start_token: "
            "verranno probabilmente invalidati."
        )
    if missing_end > 0:
        logger.warning(
            f"{missing_end} blocchi non contenevano l'end_token: "
            "verranno probabilmente invalidati."
        )

    return blocks


def save_raw_blocks(
    blocks: List[Block],
    cfg: Dict[str, Any],
    logger: logging.Logger,
) -> Path:
    """
    Salva i blocchi grezzi in un file .txt (uno per blocco, separati da riga vuota)
    nella directory di output configurata in generation.output_dir
    (default: outputs/generations).
    """
    gen_cfg = cfg.get("generation", {})
    generations_dir = Path(
        gen_cfg.get("output_dir", os.path.join("outputs", "generations"))
    ).resolve()
    generations_dir.mkdir(parents=True, exist_ok=True)

    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = generations_dir / f"batch_{ts}.txt"

    with out_path.open("w", encoding="utf-8") as f:
        for i, block in enumerate(blocks):
            txt = block.strip()
            f.write(txt)
            if i != len(blocks) - 1:
                f.write("\n\n")

    logger.info(f"Batch di blocchi generati salvato in: {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    logger = setup_logger()

    logger.info(f"Caricamento config da {args.config}")
    cfg = load_config(args.config)

    mapping_cfg = cfg.get("mapping", {})
    gen_cfg = cfg.get("generation", {})
    validator_cfg = cfg.get("validator", {})

    # Modalità di validazione da usare in GENERAZIONE (default: soft)
    gen_validator_mode = validator_cfg.get("mode_generation", "soft")
    if gen_validator_mode not in ("soft", "hard"):
        logger.warning(
            "validator.mode_generation=%r non valido; uso 'soft' come default.",
            gen_validator_mode,
        )
        gen_validator_mode = "soft"

    # Mapping
    mapping_path = Path(
        mapping_cfg.get("seed_path", os.path.join("src", "mapping", "mapping_seed.csv"))
    ).resolve()
    mapping_rows = load_mapping_seed(mapping_path, logger)

    # Runtime validator condiviso (token, regex, ecc.)
    runtime = build_validator_runtime(cfg)
    start_token = runtime["start_token"]
    end_token = runtime["end_token"]

    logger.info(f"Start token: {start_token}")
    logger.info(f"Gen token:   {runtime['gen_token']}")
    logger.info(f"End token:   {end_token}")
    logger.info(f"Theme di default (mapping.default_theme): {mapping_cfg.get('default_theme', 'Generic')}")
    logger.info(f"Modalità validator in generazione: {gen_validator_mode}")

    # Costruzione prompt
    prompts = build_prompts(mapping_rows, cfg, runtime, logger)

    # W&B
    run = init_wandb(cfg, run_name="generate-a06")
    if run is not None:
        run.log(
            {
                "gen.prompts": len(prompts),
                "gen.mapping_rows": len(mapping_rows),
                "gen.validator_mode": gen_validator_mode,
            }
        )

    try:
        # Modello + tokenizer
        model, tokenizer, device, eos_token_id = load_model_and_tokenizer(cfg, runtime, logger)

        # Generazione grezza
        raw_generations = generate_with_model(
            model=model,
            tokenizer=tokenizer,
            device=device,
            eos_token_id=eos_token_id,
            prompts=prompts,
            cfg=cfg,
            logger=logger,
        )

        # Post-process: blocchi "puliti" <start...end>
        blocks = postprocess_generations(raw_generations, start_token, end_token, logger)

        # Salvataggio blocchi grezzi
        txt_path = save_raw_blocks(blocks, cfg, logger)

        # Validazione in memoria usando a03_validate (soft/hard secondo config)
        results, stats = validate_blocks(
            blocks,
            cfg=cfg,
            mode=gen_validator_mode,
            **runtime,  # start_token, gen_token, end_token, regex, rarities, ecc.
        )

        # JSONL validato accanto al .txt
        suffix = "_validated_soft.jsonl" if gen_validator_mode == "soft" else "_validated.jsonl"
        validated_path = txt_path.with_name(txt_path.stem + suffix)
        write_validation_results_jsonl(results, validated_path)

        # Logging stats
        total = stats["total"]
        valid = stats["valid"]
        invalid = stats["invalid"]
        pass_rate = (valid / total) if total > 0 else 0.0

        logger.info("=== Riepilogo generazione+validazione ===")
        logger.info(f"Blocchi totali: {total}")
        logger.info(f"Validi:         {valid}")
        logger.info(f"Invalidi:       {invalid}")
        logger.info(f"Pass-rate:      {pass_rate:.3f}")
        logger.info(f"File TXT:       {txt_path}")
        logger.info(f"File JSONL:     {validated_path}")

        if run is not None:
            metrics = {
                "gen.total_blocks": total,
                "gen.valid": valid,
                "gen.invalid": invalid,
                "gen.pass_rate": pass_rate,
                "gen.validator_mode": gen_validator_mode,
            }
            # logghiamo anche i primi errori, se ci sono
            error_counts = stats.get("error_counts", {})
            for i, (err, count) in enumerate(
                sorted(error_counts.items(), key=lambda kv: kv[1], reverse=True)[:10]
            ):
                metrics[f"gen.error_top_{i}"] = f"{err} :: {count}"
            run.log(metrics)

    finally:
        if run is not None:
            run.finish()
            logger.info("W&B run chiuso.")

    logger.info(
        "Done. Prossimo step della pipeline: usare a07_curate sul JSONL validato."
    )


if __name__ == "__main__":
    main()
