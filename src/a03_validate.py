#!/usr/bin/env python
"""
a03_validate.py

Validator per blocchi carta in formato Standard:

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

Modalità supportate:
- "hard" (rigida, default):
    - Ordine dei campi rigido (vedi guida operativa).
    - Tutti i campi obbligatori appaiono una sola volta.
    - `pt` solo per Creature, obbligatorio se Creature.
    - Regex "dure" per `mana_cost` e `pt` (da config.constants.regex).
    - `rarity` in set consentito (da config.constants.rarities).
    - `type` contiene almeno uno tra i macro-tipi (da config.constants.type_macros).
    - Opzionale: regola `color_mana_subset_rule` (color coerente con mana_cost).
    - EOS `<|endofcard|>` obbligatorio.

- "soft" (più permissiva, pensata per la generazione):
    - Richiede comunque la presenza dei token speciali.
    - Header (theme/character/color/type/rarity) letto in modo "quasi rigido"
      come nel dataset (dopo <|startofcard|>).
    - Body parsato in modo fuzzy:
        * `name`: accetta "name, Foo", "name Foo", ecc.
        * `mana_cost`: accetta varianti "mana_cost2", "mana_cost/text", ecc.
          Estrae tutti i token {…} e li compone; il resto va nel text.
        * `text`: accetta "text", "text —", "text Ward —", ecc.
          Tutte le linee non riconosciute come altri campi vengono trattate
          come parte del rules text.
        * `pt`: estrae il primo pattern N/M da "pt: 4/5 Wurms 6+ ...".
    - Vincoli minimi:
        * name non vuoto e "sensato",
        * text con lunghezza minima configurabile (validator.soft_min_text_chars),
        * pt presente e decodificabile per Creature,
        * theme/color/type/rarity non vuoti e type con almeno un macro-tipo.

Esecuzione tipica (validator "hard" per dataset, come prima):

    python -m src.a03_validate --config configs/config.yaml \
        --input data/processed/train.txt data/processed/val.txt data/processed/test.txt

Modalità soft (es. per ispezionare i file generati):

    python -m src.a03_validate --config configs/config.yaml --mode soft -i outputs/generations/batch_*.txt

In più, questo modulo espone alcune funzioni riusabili da altri script
(es. a06_generate_and_validate.py):

- build_validator_runtime(cfg) -> dict
- validate_blocks(blocks, cfg=..., mode="hard", **runtime) -> (results, stats)
- write_validation_results_jsonl(results, out_path)
"""

from __future__ import annotations

import argparse
import json
import logging
import re
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
        description="Validator per blocchi carta MTG×MHA (hard/soft)."
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
    parser.add_argument(
        "--mode",
        type=str,
        choices=["hard", "soft"],
        default="hard",
        help=(
            "Modalità di validazione: 'hard' per il dataset (rigida, default), "
            "'soft' per la generazione (più permissiva)."
        ),
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helpers runtime riusabili
# ---------------------------------------------------------------------------

def build_validator_runtime(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Costruisce un piccolo dict con tutti i parametri necessari
    alla validazione (token speciali, regex, rarities, macro-types, toggles).

    Pensato per essere riusato da altri script (es. generate_and_validate).
    """
    validator_cfg = cfg.get("validator", {})
    constants_cfg = cfg.get("constants", {})

    # Special tokens
    special_tokens = get_special_tokens(cfg)
    if len(special_tokens) >= 3:
        start_token, gen_token, end_token = special_tokens[:3]
    else:
        # fallback ragionevole
        start_token = "<|startofcard|>"
        gen_token = "<|gen_card|>"
        end_token = "<|endofcard|>"

    # Regex mana/pt
    regexes = compile_regexes(cfg)
    mana_regex = regexes.get("mana")
    pt_regex = regexes.get("pt")

    # Rarities e macro-types dal config (niente hard-code)
    rarities = list(constants_cfg.get("rarities", []))
    macro_types = list(constants_cfg.get("type_macros", []))

    # Toggles validator
    color_mana_subset_rule = bool(
        validator_cfg.get("color_mana_subset_rule", False)
    )
    require_character = bool(
        validator_cfg.get("require_character", False)
    )

    return {
        "start_token": start_token,
        "gen_token": gen_token,
        "end_token": end_token,
        "mana_regex": mana_regex,
        "pt_regex": pt_regex,
        "rarities": rarities,
        "macro_types": macro_types,
        "color_mana_subset_rule": color_mana_subset_rule,
        "require_character": require_character,
    }


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
# Validazione di un singolo blocco — parsing HARD
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


def _validate_block_hard(
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
    Validator originale "duro" (per il dataset).
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
# Validazione di un singolo blocco — parsing SOFT
# ---------------------------------------------------------------------------

def _validate_block_soft(
    block: Block,
    *,
    cfg: Dict[str, Any],
    start_token: str,
    gen_token: str,
    end_token: str,
    mana_regex,  # unused in soft
    pt_regex,
    rarities: List[str],
    macro_types: List[str],
    color_mana_subset_rule: bool,  # unused in soft
    require_character: bool,
) -> Dict[str, Any]:
    """
    Validator "soft" per blocchi generati.

    Approccio:
      - Richiede la presenza dei token speciali, ma non l'ordine rigido
        di tutti i campi come in hard.
      - Header parsato con logica simile a quella hard (start/theme/character/color/type/rarity).
      - Body parsato in modo fuzzy per name/mana_cost/text/pt.
      - Vincoli minimi su name/text/pt.
    """
    errors: List[str] = []
    fields: Dict[str, Any] = {}

    # Token speciali all'interno del blocco (posizioni string-based)
    start_idx = block.find(start_token)
    gen_idx = block.find(gen_token, start_idx + len(start_token) if start_idx != -1 else 0)
    end_idx = block.find(end_token, gen_idx + len(gen_token) if gen_idx != -1 else 0)

    if start_idx == -1:
        errors.append("start token mancante (soft).")
    if gen_idx == -1:
        errors.append("token di generazione mancante (soft).")
    if end_idx == -1:
        errors.append("EOS mancante (soft).")

    if errors:
        return {
            "valid": False,
            "errors": errors,
            "raw": block,
            "fields": fields,
        }

    # Segmenti header / body
    header_segment = block[start_idx:gen_idx]
    body_segment = block[gen_idx + len(gen_token):end_idx]

    # --- Parsing header (start/theme/(character)/color/type/rarity) ---

    header_lines = [ln.rstrip("\n") for ln in header_segment.splitlines()]
    header_lines = [ln for ln in header_lines if ln.strip() != ""]

    if not header_lines:
        errors.append("header vuoto (soft).")
    idx = 0

    # Start token
    if not header_lines or header_lines[0].strip() != start_token:
        errors.append(
            f"start token mancante o errato (soft) "
            f"(atteso '{start_token}')."
        )
    else:
        idx = 1

    def consume_field(name: str, required: bool = True) -> bool:
        nonlocal idx
        if idx >= len(header_lines):
            if required:
                errors.append(f"campo '{name}' mancante (soft).")
            return False
        line = header_lines[idx].strip()
        lower = line.lower()
        prefix = f"{name}:"
        if not lower.startswith(prefix):
            if required:
                errors.append(
                    f"campo '{name}' mancante o fuori ordine (soft) "
                    f"(trovato: '{line}')."
                )
            return False
        value = line.split(":", 1)[1].strip()
        fields[name] = value
        idx += 1
        return True

    # theme (obbligatorio)
    consume_field("theme", required=True)

    # character (opzionale ma configurabile)
    old_idx = idx
    has_character = consume_field("character", required=False)
    if require_character and not has_character:
        # se non abbiamo avanzato, character manca
        errors.append("campo 'character' mancante ma richiesto dal config (soft).")
    # color (obbligatorio)
    consume_field("color", required=True)
    # type (obbligatorio)
    consume_field("type", required=True)
    # rarity (obbligatorio)
    consume_field("rarity", required=True)

    # --- Parsing body (name/mana_cost/text/pt) fuzzy ---

    body_lines = [ln.rstrip("\n") for ln in body_segment.splitlines()]
    # Manteniamo anche linee vuote: possono separare pezzi di testo, ma non sono essenziali.

    name_value: str | None = None
    mana_value: str = ""
    pt_value: str | None = None
    text_lines: List[str] = []

    for raw_line in body_lines:
        line = raw_line.rstrip("\n")
        stripped = line.strip()
        if not stripped:
            # linea vuota -> separatore di testo
            continue

        lower = stripped.lower()

        # pt: estrae il primo pattern N/M, il resto va nel testo
        if pt_value is None and lower.startswith("pt"):
            # es. "pt: 4/5 Wurms 6+ ..."
            m_rest = re.match(r"^\s*pt\s*[:\-]?\s*(.*)$", stripped, flags=re.IGNORECASE)
            rest = m_rest.group(1).strip() if m_rest else stripped
            m_pt = re.search(r"(\d+)\s*/\s*(\d+)", rest)
            if m_pt:
                p1 = int(m_pt.group(1))
                p2 = int(m_pt.group(2))
                pt_value = f"{p1}/{p2}"
                # parte dopo il pattern N/M va nel text
                tail = rest[m_pt.end():].strip()
                if tail:
                    text_lines.append(tail)
            else:
                # nessun N/M -> linea rumorosa, mettiamola nel testo
                text_lines.append(rest)
            continue

        # name: linee che iniziano con "name"
        if name_value is None and lower.startswith("name"):
            # es. "name, the Unworthy", "name Soratami, the Storm's Champion"
            m_name = re.match(r"^\s*name\b[^\w]*\s*(.*)$", stripped, flags=re.IGNORECASE)
            if m_name:
                raw = m_name.group(1).strip()
                raw = raw.lstrip(",:- \t")
                name_value = raw
            else:
                parts = stripped.split(None, 1)
                name_value = parts[1].strip() if len(parts) > 1 else ""
            continue

        # mana_cost: varianti "mana_cost2", "mana_cost/text", "mana_costly_name", ecc.
        if "mana_cost" in lower or lower.startswith("mana cost"):
            # es. "mana_cost2 {4}{R}", "mana_cost/text {1}{R}: Target..."
            m_mc = re.match(r"^\s*mana[_\s-]*cost\w*[^:]*[:\s]*(.*)$", stripped, flags=re.IGNORECASE)
            rest = m_mc.group(1).strip() if m_mc else stripped
            # estraiamo tutti i token {…}
            tokens = re.findall(r"\{[^}]+\}", rest)
            if tokens:
                mana_value = "".join(tokens)
                # rimuoviamo i token dal resto e mandiamo il resto nel text
                leftover = re.sub(r"\{[^}]+\}", "", rest).strip()
                if leftover:
                    text_lines.append(leftover)
            else:
                # nessun token di mana: trattiamo tutto come testo
                text_lines.append(rest)
            continue

        # text: linee che iniziano con "text"
        if lower.startswith("text"):
            # es. "text When...", "text—{T}: ...", "text Ward — ..."
            m_text = re.match(r"^\s*text\b[^\w-]*[-—:]?\s*(.*)$", stripped, flags=re.IGNORECASE)
            value = m_text.group(1).strip() if m_text else ""
            if value:
                text_lines.append(value)
            continue

        # fallback: tutto il resto viene considerato parte del rules text
        text_lines.append(stripped)

    # Componiamo i campi body
    fields["name"] = name_value or ""
    fields["mana_cost"] = mana_value or ""
    text_value = "\n".join(text_lines).strip()
    fields["text"] = text_value
    if pt_value is not None:
        fields["pt"] = pt_value

    # --- Controlli "soft" sui valori ---

    validator_cfg = cfg.get("validator", {})
    soft_min_text_chars = int(validator_cfg.get("soft_min_text_chars", 20))

    # rarity e macro-types come in hard (rarity / type dal header sono puliti)
    rarity = fields.get("rarity", "")
    if not rarity:
        errors.append("rarity vuota (soft).")
    elif rarities and rarity not in rarities:
        errors.append(
            f"rarity '{rarity}' non è tra quelle consentite (soft): {sorted(rarities)}."
        )

    type_line = fields.get("type", "")
    if not type_line:
        errors.append("type vuoto (soft).")
    else:
        tl_lower = type_line.lower()
        mt_lower = [t.lower() for t in macro_types] if macro_types else []
        if mt_lower and not any(mt in tl_lower for mt in mt_lower):
            errors.append(
                "type non contiene alcun macro-tipo valido "
                f"(soft, attesi uno tra: {macro_types})."
            )

    # Campi base header
    for key in ("theme", "color"):
        if not fields.get(key, ""):
            errors.append(f"campo '{key}' vuoto (soft).")

    # name: non vuoto e con almeno 2 lettere alfabetiche
    name = fields.get("name", "")
    if not name or sum(ch.isalpha() for ch in name) < 2:
        errors.append("campo 'name' mancante o troppo corto (soft).")

    # text: lunghezza minima
    if len(text_value.strip()) < soft_min_text_chars:
        errors.append(
            f"text troppo corto per la modalità soft "
            f"(len={len(text_value.strip())}, min={soft_min_text_chars})."
        )

    # pt vs Creature: pt comunque obbligatorio per Creature
    type_is_creature = "creature" in type_line.lower()
    pt_value = fields.get("pt", None)
    if type_is_creature and not pt_value:
        errors.append("pt obbligatorio per carte Creature (soft).")

    # Se pt c'è, verifichiamo che sia in forma N/M (se abbiamo pt_regex lo riusiamo)
    if pt_value is not None:
        if not pt_value:
            errors.append("pt vuoto (soft).")
        else:
            # Proviamo con il pt_regex se esiste, altrimenti facciamo un check semplice N/M
            if pt_regex is not None:
                if not pt_regex.fullmatch(pt_value):
                    errors.append(
                        f"pt '{pt_value}' non rispetta il pattern richiesto (soft)."
                    )
            else:
                if not re.fullmatch(r"\d+/\d+", pt_value):
                    errors.append(
                        f"pt '{pt_value}' non è nel formato 'N/M' (soft)."
                    )

    valid = len(errors) == 0

    return {
        "valid": valid,
        "errors": errors,
        "raw": block,
        "fields": fields,
    }


# ---------------------------------------------------------------------------
# API pubblica: validate_block (hard/soft)
# ---------------------------------------------------------------------------

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
    mode: str = "hard",
) -> Dict[str, Any]:
    """
    Valida un singolo blocco.

    Parametri:
      - mode: "hard" (rigido, dataset) o "soft" (più permissivo, generazione).

    Ritorna un dict:
        {
          "valid": bool,
          "errors": [str],
          "raw": <string>,
          "fields": { ... }   # anche se parziale
        }
    """
    if mode == "hard":
        return _validate_block_hard(
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
    elif mode == "soft":
        return _validate_block_soft(
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
    else:
        raise ValueError(f"Modalità di validazione non supportata: {mode!r}")


# ---------------------------------------------------------------------------
# Validazione di liste di blocchi (riusabile)
# ---------------------------------------------------------------------------

def validate_blocks(
    blocks: List[Block],
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
    mode: str = "hard",
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Valida una lista di blocchi di testo già in memoria.

    Parametri:
      - mode: "hard" (rigido, dataset) o "soft" (più permissivo, generazione).

    Ritorna:
      - results: lista di dict come `validate_block`
      - stats:   dict con chiavi:
                 {"total", "valid", "invalid", "error_counts"}
    """
    results: List[Dict[str, Any]] = []
    valid_count = 0
    invalid_count = 0
    error_counter: Counter = Counter()

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
            mode=mode,
        )
        results.append(result)

        if result["valid"]:
            valid_count += 1
        else:
            invalid_count += 1
            for err in result["errors"]:
                error_counter[err] += 1

    stats = {
        "total": len(blocks),
        "valid": valid_count,
        "invalid": invalid_count,
        "error_counts": dict(error_counter),
    }
    return results, stats


def write_validation_results_jsonl(
    results: List[Dict[str, Any]],
    out_path: Path,
) -> None:
    """
    Scrive una lista di risultati di validazione (dict come validate_block)
    in formato JSONL nel path specificato.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f_out:
        for result in results:
            f_out.write(json.dumps(result, ensure_ascii=False) + "\n")


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
    mode: str = "hard",
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

    suffix = "_validated_soft.jsonl" if mode == "soft" else "_validated.jsonl"
    out_path = path.with_name(path.stem + suffix)
    logger.info(f"Validazione ({mode}) di {path} -> {out_path} ({n_blocks} blocchi).")

    results, stats = validate_blocks(
        blocks,
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
        mode=mode,
    )

    write_validation_results_jsonl(results, out_path)

    if stats["total"] > 0:
        pass_rate = stats["valid"] / stats["total"]
    else:
        pass_rate = 0.0

    logger.info(
        f"File {path} ({mode}): valid={stats['valid']}, invalid={stats['invalid']}, "
        f"pass_rate={pass_rate:.3f}"
    )

    file_stats = {
        "file": str(path),
        "total": stats["total"],
        "valid": stats["valid"],
        "invalid": stats["invalid"],
        "error_counts": stats["error_counts"],
    }
    return file_stats


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

    mode = args.mode  # "hard" di default

    # Runtime validator condiviso
    runtime = build_validator_runtime(cfg)
    start_token = runtime["start_token"]
    gen_token = runtime["gen_token"]
    end_token = runtime["end_token"]
    mana_regex = runtime["mana_regex"]
    pt_regex = runtime["pt_regex"]
    rarities = runtime["rarities"]
    macro_types = runtime["macro_types"]
    color_mana_subset_rule = runtime["color_mana_subset_rule"]
    require_character = runtime["require_character"]

    logger.info(f"Modalità validator: {mode}")
    logger.info(f"Start token: {start_token}")
    logger.info(f"Gen token:   {gen_token}")
    logger.info(f"End token:   {end_token}")
    logger.info(f"Rarities consentite (da config): {rarities}")
    logger.info(f"Macro-types (da config): {macro_types}")
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
            mode=mode,
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
    summary_name = "validate_summary_soft.json" if mode == "soft" else "validate_summary.json"
    summary_path = processed_dir / summary_name
    summary_payload = {
        "mode": mode,
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
    run = init_wandb(cfg, run_name=f"validate-a03-{mode}")
    if run is not None:
        metrics = {
            "validate.mode": mode,
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
