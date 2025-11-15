#!/usr/bin/env python
"""
a03_validate.py

Validator "duro" per blocchi carta in formato Standard:

<|startofcard|>
theme: ...
character: ...          # opzionale (configurabile)
color: ...
type: ...
rarity: ...
<|gen_card|>
name: ...
mana_cost: ...
text: ...
pt: ...                 # SOLO se type contiene "Creature"
<|endofcard|>

Regole principali:
- Ordine dei campi rigido (vedi guida operativa).
- Tutti i campi obbligatori appaiono una sola volta.
- `pt` solo per Creature, obbligatorio se Creature.
- Regex "dure" per `mana_cost` e `pt` (da config.constants.regex).
- `rarity` in set consentito (da config.constants.rarities).
- `type` contiene almeno uno tra i macro-tipi (da config.constants.type_macros).
- Opzionale: regola `color_mana_subset_rule` (color coerente con mana_cost).
- EOS `<|endofcard|>` obbligatorio.
- Output: `*_validated.jsonl` con `{valid, errors, raw, fields}` per blocco.

Esecuzione tipica (dal root della repo):

    python -m src.a03_validate --config configs/config.yaml \
        --input data/processed/train.txt data/processed/val.txt data/processed/test.txt

Se `--input` è omesso, si usano automaticamente train/val/test da `data.processed_dir`.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple, Any

from .a00_utils import (
    load_config,
    get_special_tokens,
    init_wandb,
    compile_regexes,
    is_color_mana_consistent,
)

Block = str


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validator 'duro' per blocchi carta MTG×MHA."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path al file di configurazione YAML.",
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        nargs="*",
        help=(
            "Path dei file .txt da validare. "
            "Se omesso: usa train/val/test da data.processed_dir."
        ),
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Lettura blocchi
# ---------------------------------------------------------------------------

def read_blocks(path: Path, start_token: str) -> List[Block]:
    """
    Legge un file .txt e lo spezza in blocchi, uno per carta,
    usando una riga contenente SOLO `start_token` come delimitatore.

    Il blocco include il `start_token` in testa.
    """
    if not path.is_file():
        raise FileNotFoundError(f"File non trovato: {path}")

    blocks: List[Block] = []
    current_lines: List[str] = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip() == start_token:
                if current_lines:
                    blocks.append("".join(current_lines))
                    current_lines = []
            current_lines.append(line)

    if current_lines:
        blocks.append("".join(current_lines))

    return blocks


# ---------------------------------------------------------------------------
# Validazione di un singolo blocco
# ---------------------------------------------------------------------------

def _parse_fields_in_order(
    block: Block,
    *,
    start_token: str,
    gen_token: str,
    end_token: str,
    require_character: bool,
) -> Tuple[Dict[str, Any], List[str]]:
    """
    Parsing "rigido" del blocco secondo l'ordine di campi Standard.

    Ritorna (fields, errors_struct) dove:
      - fields: dict parziale (solo se riusciamo a parsare i campi)
      - errors_struct: lista di errori strutturali riscontrati
    """
    errors: List[str] = []
    fields: Dict[str, Any] = {}

    # Rimuoviamo solo righe completamente vuote per robustezza
    raw_lines = [ln.rstrip("\n") for ln in block.splitlines()]
    lines = [ln for ln in raw_lines if ln.strip() != ""]

    if not lines:
        return fields, ["blocco vuoto"]

    idx = 0

    # 1) Start token
    if lines[idx].strip() != start_token:
        errors.append(
            f"start token mancante o errato "
            f"(atteso '{start_token}', trovato '{lines[idx]}')."
        )
        # Non forziamo uscita immediata, ma il resto avrà poco senso.
    else:
        idx += 1

    def required_field(name: str) -> bool:
        nonlocal idx
        if idx >= len(lines):
            errors.append(f"campo '{name}' mancante (EOF prematuro).")
            return False
        line = lines[idx].strip()
        lower = line.lower()
        prefix = f"{name}:"
        if not lower.startswith(prefix):
            errors.append(
                f"campo '{name}' mancante o fuori ordine "
                f"(trovato: '{line}')."
            )
            return False
        value = line.split(":", 1)[1].strip()
        fields[name] = value
        idx += 1
        return True

    def optional_field(name: str) -> bool:
        nonlocal idx
        if idx >= len(lines):
            return False
        line = lines[idx].strip()
        lower = line.lower()
        prefix = f"{name}:"
        if not lower.startswith(prefix):
            return False
        value = line.split(":", 1)[1].strip()
        fields[name] = value
        idx += 1
        return True

    # 2) theme (obbligatorio)
    if not required_field("theme"):
        # se manca theme, la struttura è già compromessa
        return fields, errors

    # 3) character (opzionale ma configurabile)
    has_character = optional_field("character")
    if require_character and not has_character:
        errors.append("campo 'character' mancante ma richiesto dal config.")

    # 4) color (obbligatorio, ma valore può essere vuoto per colorless)
    if not required_field("color"):
        return fields, errors

    # 5) type (obbligatorio)
    if not required_field("type"):
        return fields, errors

    # 6) rarity (obbligatorio)
    if not required_field("rarity"):
        return fields, errors

    # 7) token <|gen_card|>
    if idx >= len(lines):
        errors.append("token di generazione mancante (EOF prematuro).")
        return fields, errors

    if lines[idx].strip() != gen_token:
        errors.append(
            f"token di generazione mancante o errato "
            f"(atteso '{gen_token}', trovato '{lines[idx]}')."
        )
        return fields, errors
    idx += 1

    # 8) name (obbligatorio)
    if not required_field("name"):
        return fields, errors

    # 9) mana_cost (obbligatorio come campo, valore può essere vuoto)
    if not required_field("mana_cost"):
        return fields, errors

    # 10) text (obbligatorio, una sola riga)
    if not required_field("text"):
        return fields, errors

    # 11) pt (opzionale; controllo Creature vs non-Creature a valle)
    optional_field("pt")

    # 12) EOS
    if idx >= len(lines):
        errors.append("EOS mancante (EOF prematuro).")
        return fields, errors

    if lines[idx].strip() != end_token:
        errors.append(
            f"EOS mancante o errato "
            f"(atteso '{end_token}', trovato '{lines[idx]}')."
        )
        return fields, errors

    idx += 1

    # 13) eventuale trailing garbage dopo EOS
    if idx < len(lines):
        extra = "; ".join(lines[idx:])
        errors.append(f"contenuto extra dopo EOS: {extra!r}")

    return fields, errors


def validate_block(
    block: Block,
    *,
    cfg: Dict[str, Any],
    start_token: str,
    gen_token: str,
    end_token: str,
    mana_regex,
    pt_regex,
    rarities: List[str],
    macro_types: List[str],
    color_mana_subset_rule: bool,
    require_character: bool,
) -> Dict[str, Any]:
    """
    Valida un singolo blocco.

    Ritorna un dict:
        {
          "valid": bool,
          "errors": [str],
          "raw": <string>,
          "fields": { ... }   # anche se parziale
        }
    """
    errors: List[str] = []

    # 1) parsing strutturale + ordine campi
    fields, struct_errors = _parse_fields_in_order(
        block,
        start_token=start_token,
        gen_token=gen_token,
        end_token=end_token,
        require_character=require_character,
    )
    errors.extend(struct_errors)

    # Se la struttura è già rotta a monte, fermiamoci qui
    if struct_errors:
        return {
            "valid": False,
            "errors": errors,
            "raw": block,
            "fields": fields,
        }

    # 2) Regole sui valori / dominio

    # 2.1) Rarity
    rarity = fields.get("rarity", "")
    if not rarity:
        errors.append("rarity vuota.")
    elif rarities and rarity not in rarities:
        errors.append(
            f"rarity '{rarity}' non è tra quelle consentite: {sorted(rarities)}."
        )

    # 2.2) Type deve contenere almeno un macro-tipo
    type_line = fields.get("type", "")
    if not type_line:
        errors.append("type vuoto.")
    else:
        tl_lower = type_line.lower()
        mt_lower = [t.lower() for t in macro_types] if macro_types else []
        if mt_lower and not any(mt in tl_lower for mt in mt_lower):
            errors.append(
                "type non contiene alcun macro-tipo valido "
                f"(attesi uno tra: {macro_types})."
            )

    # 2.3) Regole su pt vs Creature
    type_is_creature = "creature" in type_line.lower()
    pt_value = fields.get("pt", None)

    if type_is_creature and pt_value is None:
        errors.append("pt obbligatorio per carte Creature.")
    if (not type_is_creature) and pt_value is not None:
        errors.append("pt non permesso per carte non-Creature.")

    if pt_value is not None:
        if not pt_value:
            errors.append("pt vuoto.")
        else:
            if pt_regex is not None and not pt_regex.fullmatch(pt_value):
                errors.append(
                    f"pt '{pt_value}' non rispetta il pattern richiesto."
                )

    # 2.4) Regex su mana_cost (vuoto è consentito)
    mana_value = fields.get("mana_cost", "")
    if mana_value:
        if mana_regex is not None and not mana_regex.fullmatch(mana_value):
            errors.append(
                f"mana_cost '{mana_value}' non rispetta il pattern richiesto."
            )

    # 2.5) Regola opzionale: color coerente con mana_cost
    if color_mana_subset_rule:
        color_value = fields.get("color", "")
        if not is_color_mana_consistent(color_value, mana_value, cfg):
            errors.append(
                f"color '{color_value}' non coerente con i simboli di mana in '{mana_value}'."
            )

    # 2.6) Campi base non vuoti (theme, name, text)
    for key in ("theme", "name", "text"):
        if not fields.get(key, ""):
            errors.append(f"campo '{key}' vuoto.")

    valid = len(errors) == 0

    return {
        "valid": valid,
        "errors": errors,
        "raw": block,
        "fields": fields,
    }


# ---------------------------------------------------------------------------
# Validazione file
# ---------------------------------------------------------------------------

def validate_file(
    path: Path,
    *,
    cfg: Dict[str, Any],
    start_token: str,
    gen_token: str,
    end_token: str,
    mana_regex,
    pt_regex,
    rarities: List[str],
    macro_types: List[str],
    color_mana_subset_rule: bool,
    require_character: bool,
) -> Dict[str, Any]:
    """
    Valida tutti i blocchi presenti in `path` e scrive un file JSONL
    con suffisso `_validated.jsonl`.

    Ritorna uno small dict con statistiche per il file.
    """
    logger = logging.getLogger("a03_validate")

    blocks = read_blocks(path, start_token=start_token)
    n_blocks = len(blocks)
    if n_blocks == 0:
        logger.warning(f"Nessun blocco trovato in {path}.")
        return {
            "file": str(path),
            "total": 0,
            "valid": 0,
            "invalid": 0,
            "error_counts": {},
        }

    out_path = path.with_name(path.stem + "_validated.jsonl")
    logger.info(f"Validazione di {path} -> {out_path} ({n_blocks} blocchi).")

    valid_count = 0
    invalid_count = 0
    error_counter: Counter = Counter()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f_out:
        for block in blocks:
            result = validate_block(
                block,
                cfg=cfg,
                start_token=start_token,
                gen_token=gen_token,
                end_token=end_token,
                mana_regex=mana_regex,
                pt_regex=pt_regex,
                rarities=rarities,
                macro_types=macro_types,
                color_mana_subset_rule=color_mana_subset_rule,
                require_character=require_character,
            )

            if result["valid"]:
                valid_count += 1
            else:
                invalid_count += 1
                for err in result["errors"]:
                    error_counter[err] += 1

            f_out.write(
                json.dumps(result, ensure_ascii=False) + "\n"
            )

    logger.info(
        f"File {path}: valid={valid_count}, invalid={invalid_count}, "
        f"pass_rate={valid_count / n_blocks:.3f}"
    )

    return {
        "file": str(path),
        "total": n_blocks,
        "valid": valid_count,
        "invalid": invalid_count,
        "error_counts": dict(error_counter),
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
    logger = logging.getLogger("a03_validate")

    cfg = load_config(args.config)
    data_cfg = cfg.get("data", {})
    validator_cfg = cfg.get("validator", {})
    constants_cfg = cfg.get("constants", {})

    processed_dir = Path(data_cfg.get("processed_dir", "data/processed")).resolve()

    # Se input non specificato, validiamo train/val/test
    if args.input:
        input_paths = [Path(p) for p in args.input]
    else:
        input_paths = [
            processed_dir / "train.txt",
            processed_dir / "val.txt",
            processed_dir / "test.txt",
        ]

    # Special tokens
    special_tokens = get_special_tokens(cfg)
    if len(special_tokens) >= 3:
        start_token, gen_token, end_token = special_tokens[:3]
    else:
        # fallback ragionevole
        start_token = "<|startofcard|>"
        gen_token = "<|gen_card|>"
        end_token = "<|endofcard|>"

    logger.info(f"Start token: {start_token}")
    logger.info(f"Gen token:   {gen_token}")
    logger.info(f"End token:   {end_token}")

    # Regex mana/pt
    regexes = compile_regexes(cfg)
    mana_regex = regexes.get("mana")
    pt_regex = regexes.get("pt")

    # Rarities e macro-types dal config (niente hard-code)
    rarities = list(constants_cfg.get("rarities", []))
    macro_types = list(constants_cfg.get("type_macros", []))

    logger.info(f"Rarities consentite (da config): {rarities}")
    logger.info(f"Macro-types (da config): {macro_types}")

    # Toggles validator
    color_mana_subset_rule = bool(
        validator_cfg.get("color_mana_subset_rule", False)
    )
    require_character = bool(
        validator_cfg.get("require_character", False)
    )

    logger.info(
        f"Regola color_mana_subset_rule attiva: {color_mana_subset_rule}"
    )
    logger.info(f"'character' richiesto: {require_character}")

    # Validazione dei file
    all_stats: List[Dict[str, Any]] = []
    total_blocks = 0
    total_valid = 0
    total_invalid = 0
    global_error_counter: Counter = Counter()

    for path in input_paths:
        stats = validate_file(
            path,
            cfg=cfg,
            start_token=start_token,
            gen_token=gen_token,
            end_token=end_token,
            mana_regex=mana_regex,
            pt_regex=pt_regex,
            rarities=rarities,
            macro_types=macro_types,
            color_mana_subset_rule=color_mana_subset_rule,
            require_character=require_character,
        )
        all_stats.append(stats)
        total_blocks += stats["total"]
        total_valid += stats["valid"]
        total_invalid += stats["invalid"]
        for err, count in stats["error_counts"].items():
            global_error_counter[err] += count

    logger.info("=== Riepilogo globale validator ===")
    if total_blocks > 0:
        pass_rate = total_valid / total_blocks
    else:
        pass_rate = 0.0
    logger.info(f"Totale blocchi: {total_blocks}")
    logger.info(f"Validi: {total_valid}")
    logger.info(f"Invalidi: {total_invalid}")
    logger.info(f"Pass-rate complessivo: {pass_rate:.3f}")
    if global_error_counter:
        logger.info("Errori più frequenti:")
        for err, count in global_error_counter.most_common(10):
            logger.info(f"  - {err}: {count}")

    # Salviamo anche uno small JSON riassuntivo accanto a processed_dir
    summary_path = processed_dir / "validate_summary.json"
    summary_payload = {
        "files": all_stats,
        "total_blocks": total_blocks,
        "valid": total_valid,
        "invalid": total_invalid,
        "pass_rate": pass_rate,
        "global_error_counts": dict(global_error_counter),
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info(f"Riepilogo globale salvato in: {summary_path}")

    # W&B logging (lite)
    run = init_wandb(cfg, run_name="validate-a03")
    if run is not None:
        metrics = {
            "validate.total_blocks": total_blocks,
            "validate.valid": total_valid,
            "validate.invalid": total_invalid,
            "validate.pass_rate": pass_rate,
        }
        # Logghiamo solo i primi errori più frequenti (se ce ne sono)
        for i, (err, count) in enumerate(
            global_error_counter.most_common(10)
        ):
            metrics[f"validate.error_top_{i}"] = f"{err} :: {count}"
        run.log(metrics)
        run.finish()


if __name__ == "__main__":
    main()
