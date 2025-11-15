#!/usr/bin/env python
"""
a01_preprocessor.py — Scryfall JSON (bulk) -> formato testuale strutturato per training

Obiettivo (step 1):
- Leggere il config unificato (configs/config.yaml).
- Caricare il bulk JSON di Scryfall (default_cards).
- Filtrare: lingua EN, layout "normal" (escludiamo DFC/Adventure/Saga/flip/split).
- Estrarre/normalizzare i campi minimi: color, type, rarity, name, mana_cost, text, pt (se Creature).
- Scrivere un file di output testuale con un blocco per carta:
    <|startofcard|>
    theme: Generic
    character: N/A
    color: W R
    type: Legendary Creature | Human Wizard
    rarity: rare
    <|gen_card|>
    name: Aurelia, Exemplar of Justice
    mana_cost: {2}{R}{W}
    text: Flying, vigilance
    pt: 2/5                     # SOLO se type contiene "Creature"
    <|endofcard|>

- Generare anche data/processed/preprocess_stats.json con conteggi e motivi scarto.

Integrazione W&B (modalità "lite"):
- Se `wandb.enabled` è true nel config e la libreria è disponibile:
  - apre un run con `run_name="preprocess"`,
  - logga scalari chiave (input_total, kept, dropped, drop_reasons.*, rarity.*, type.*),
  - chiude il run a fine script.

Nota: se W&B è disabilitato o mancante, l'esecuzione procede normalmente senza logging.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter
from typing import Any, Dict, List

from src.a00_utils import load_config, get_special_tokens, init_wandb, MhaMapper


# --------------------------
# Helpers di normalizzazione
# --------------------------
def _norm_whitespace(s: str) -> str:
    """Rende spazi/newline puliti: trim e collassa whitespace multipli su testo breve."""
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    # Oracle text in Scryfall può avere \n per separare abilità; per il dataset preferiamo una sola riga.
    s = s.replace("\n", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _norm_type_line(type_line: str) -> str:
    """
    Scryfall usa di solito 'Legendary Creature — Human Wizard'.
    Per coerenza con il nostro schema: rimpiazziamo l'em dash ' — ' con ' | '.
    """
    t = type_line.replace(" — ", " | ").strip()
    # Collassa eventuali doppi spazi
    t = re.sub(r"\s+", " ", t)
    return t


def _color_str(card: Dict[str, Any]) -> str:
    """
    Costruisce la stringa colore "W U" a partire da color_identity (preferito) o colors.
    Se colorless -> stringa vuota.
    """
    ci = card.get("color_identity") or card.get("colors") or []
    # Garantiamo ordine deterministico W U B R G
    order = ["W", "U", "B", "R", "G"]
    syms = [c for c in order if c in ci]
    return " ".join(syms)


def _mana_cost(card: Dict[str, Any]) -> str:
    """
    Scryfall fornisce già mana_cost normalizzato es. '{2}{W}{R}'.
    Se manca (es. Land), restituiamo stringa vuota: 'mana_cost: ' resterà vuoto.
    """
    mc = card.get("mana_cost", "") or ""
    return mc.strip()


def _oracle_text(card: Dict[str, Any]) -> str:
    """
    Usa oracle_text; se assente, prova rules_text su card_faces (ma noi escludiamo quei layout).
    Normalizza whitespace su una riga.
    """
    text = card.get("oracle_text", "") or ""
    return _norm_whitespace(text)


def _pt_line_if_creature(card: Dict[str, Any], type_line_norm: str) -> str:
    """
    Se la carta è una Creature e P/T sono disponibili in forma semplice, restituiamo 'pt: P/T'.
    Evitiamo forme complesse come '*' (per semplicità del dataset "minimal"):
      - Se incontriamo '*' o valori non numerici, omettiamo la riga 'pt:'.
    """
    if "Creature" not in type_line_norm:
        return ""
    power = (card.get("power") or "").strip()
    toughness = (card.get("toughness") or "").strip()
    # Accettiamo numeri o 'X'; scartiamo '*' e stringhe non semplici.
    if re.fullmatch(r"[0-9X]+", power) and re.fullmatch(r"[0-9X]+", toughness):
        return f"pt: {power}/{toughness}"
    return ""


# --------------------------
# Filtri e motivi scarto
# --------------------------
def _is_lang_ok(card: Dict[str, Any], lang: str = "en") -> bool:
    return (card.get("lang") or "").lower() == lang.lower()


def _is_layout_allowed(card: Dict[str, Any], allowed: List[str]) -> bool:
    return (card.get("layout") or "") in set(allowed)


def _has_essentials(card: Dict[str, Any]) -> bool:
    """
    Richiediamo almeno: name, type_line, rarity.
    (mana_cost e oracle_text possono mancare; pt è opzionale.)
    """
    return bool(card.get("name")) and bool(card.get("type_line")) and bool(card.get("rarity"))


# --------------------------
# Blocchi formattati
# --------------------------
def _format_block(
    special_tokens: List[str],
    theme: str,
    character: str,
    color: str,
    type_line: str,
    rarity: str,
    name: str,
    mana_cost: str,
    text: str,
    pt_line: str,
) -> str:
    """
    Compone il blocco testuale con i 3 token speciali.
    "theme" e "character" ora sono parametri, non più hard-coded.
    """
    start_tok, gen_tok, end_tok = special_tokens  # assumiamo i 3 token in quest’ordine
    lines = [
        start_tok,
        f"theme: {theme}",
        f"character: {character}",
        f"color: {color}".rstrip(),
        f"type: {type_line}",
        f"rarity: {rarity}",
        gen_tok,
        f"name: {name}",
        f"mana_cost: {mana_cost}",
        f"text: {text}",
    ]
    if pt_line:
        lines.append(pt_line)
    lines.append(end_tok)
    return "\n".join(lines) + "\n"

# --------------------------
# Main
# --------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess Scryfall bulk JSON into structured text dataset.")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to unified YAML config.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_cfg = cfg.get("data", {})

    # Mapper Magic -> (theme, character) per MHA (può essere disabilitato da config)
    mapper = MhaMapper(cfg)

    allowed_layouts = data_cfg.get("layouts_allowed", ["normal"])
    lang = data_cfg.get("language", "en")

    # Percorsi I/O
    raw_path = data_cfg.get("raw_path", "data/raw/scryfall.json")
    out_dir = data_cfg.get("processed_dir", "data/processed")
    os.makedirs(out_dir, exist_ok=True)
    out_txt = os.path.join(out_dir, "all.txt")
    out_stats = os.path.join(out_dir, "preprocess_stats.json")

    # Token speciali
    special_tokens = get_special_tokens(cfg)
    if len(special_tokens) != 3:
        raise ValueError(
            f"Expected exactly 3 special tokens [start, gen, end]; got {len(special_tokens)}: {special_tokens}"
        )

    # Carica bulk JSON (atteso: array di oggetti carta)
    if not os.path.isfile(raw_path):
        raise FileNotFoundError(f"Scryfall bulk JSON not found at: {raw_path}")
    with open(raw_path, "r", encoding="utf-8") as f:
        try:
            cards = json.load(f)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Invalid JSON in {raw_path}: {e}") from e

    # Statistiche
    total = 0
    kept = 0
    dropped_reasons = Counter()
    rarity_counts = Counter()
    type_counts = Counter()

    # Scrittura progressiva (per memoria)
    with open(out_txt, "w", encoding="utf-8") as fout:
        for card in cards:
            total += 1

            # Filtri base
            if not _is_lang_ok(card, lang=lang):
                dropped_reasons["lang_not_en"] += 1
                continue
            if not _is_layout_allowed(card, allowed_layouts):
                dropped_reasons["layout_excluded"] += 1
                continue
            if not _has_essentials(card):
                dropped_reasons["missing_essentials"] += 1
                continue

            # Estrazione/normalizzazione campi
            color = _color_str(card)
            type_line = _norm_type_line(card["type_line"])
            rarity = (card.get("rarity") or "").strip().lower()
            name = (card.get("name") or "").strip()
            mana_cost = _mana_cost(card)
            text = _oracle_text(card)
            pt_line = _pt_line_if_creature(card, type_line)

            # Mapping Magic -> (theme, character) (se abilitato nel config)
            theme, character = mapper(card)

            # Componi blocco e scrivi
            block = _format_block(
                special_tokens=special_tokens,
                theme=theme,
                character=character,
                color=color,
                type_line=type_line,
                rarity=rarity,
                name=name,
                mana_cost=mana_cost,
                text=text,
                pt_line=pt_line,
            )

            
            fout.write(block)
            kept += 1

            # Aggiorna conteggi per quick EDA
            rarity_counts[rarity] += 1
            # Macro-tipo principale (prima parola di type_line o il macro-type prima della '|')
            macro = type_line.split("|")[0].strip().split(" ")[-1] if "|" in type_line else type_line.split(" ")[-1]
            type_counts[macro] += 1

    # Salva statistiche su file
    stats = {
        "input_total": total,
        "kept": kept,
        "dropped": total - kept,
        "drop_reasons": dict(dropped_reasons),
        "rarity_counts": dict(rarity_counts),
        "type_counts_macro": dict(type_counts),
        "config_used": args.config,
        "special_tokens": special_tokens,
    }
    with open(out_stats, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    # --------------------------
    # W&B logging (modalità "lite")
    # --------------------------
    run = init_wandb(cfg, run_name="preprocess")
    if run is not None:
        # Flatten minimale con prefisso "pre."
        log_payload = {
            "pre.input_total": total,
            "pre.kept": kept,
            "pre.dropped": total - kept,
        }
        # drop reasons
        for k, v in stats["drop_reasons"].items():
            log_payload[f"pre.drop.{k}"] = v
        # rarity
        for k, v in stats["rarity_counts"].items():
            log_payload[f"pre.rarity.{k}"] = v
        # type macro
        for k, v in stats["type_counts_macro"].items():
            log_payload[f"pre.type.{k}"] = v

        run.log(log_payload)
        run.finish()

    print(f"[preprocess] Done. Input: {total}, Kept: {kept}, Dropped: {total - kept}")
    print(f"[preprocess] Wrote: {out_txt}")
    print(f"[preprocess] Stats : {out_stats}")


if __name__ == "__main__":
    main()
