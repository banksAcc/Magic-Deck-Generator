"""
utils.py — helpers collegati al config unificato (configs/config.yaml)

Niente costanti hard-coded: colori, rarità, regex, special tokens, ecc.
sono tutti letti dal file YAML.

Funzioni principali:
  - load_config(path="configs/config.yaml") -> dict
  - get_special_tokens(cfg) -> List[str]
  - init_wandb(cfg, run_name)  # rispetta wandb.enabled
  - compile_regexes(cfg) -> Dict[str, Pattern]
  - parse_color_list(s, cfg) -> Set[str]
  - mana_color_symbols(mana_cost, cfg) -> Set[str]
  - is_color_mana_consistent(color_str, mana_cost, cfg) -> bool
  - read_text / write_text / read_lines
"""

from __future__ import annotations

import os
import re
import csv
import json
from pathlib import Path

from typing import Any, Dict, Optional, Pattern, Set, List

# --- Import opzionale per W&B (non obbligatorio all'avvio) ---
try:
    import wandb  # type: ignore
except Exception:
    wandb = None  # type: ignore


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
def _yaml_load(path: str) -> Dict[str, Any]:
    """Carica YAML con PyYAML."""
    import yaml  # type: ignore

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_config(path: str = "configs/config.yaml") -> Dict[str, Any]:
    """Carica il config unificato come dict semplice e garantisce le sezioni base."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Config not found: {path}")
    cfg = _yaml_load(path)
    for section in ("data", "constants", "validator", "training", "generation", "curation", "wandb"):
        if section not in cfg:
            cfg[section] = {}
    return cfg


def get_special_tokens(cfg: Dict[str, Any]) -> List[str]:
    """Ritorna i token speciali da cfg.data.special_tokens (dedup con ordine stabile)."""
    tokens = cfg.get("data", {}).get("special_tokens", []) or []
    seen: Set[str] = set()
    out: List[str] = []
    for t in tokens:
        if t not in seen:
            out.append(t)
            seen.add(t)
    return out

def load_mapping_seed(path: str):
    path = Path(path)
    seed_by_name = {}
    seed_by_oracle_id = {}

    if not path.exists():
        return seed_by_name, seed_by_oracle_id

    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row.get("card_name")
            oracle_id = row.get("oracle_id")
            theme = row.get("theme")
            character = row.get("character")

            if name:
                seed_by_name[name.lower()] = (theme, character)
            if oracle_id:
                seed_by_oracle_id[oracle_id] = (theme, character)

    return seed_by_name, seed_by_oracle_id


def load_keyword_rules(path: str):
    path = Path(path)
    if not path.exists():
        return []

    with path.open("r", encoding="utf-8") as f:
        rules = json.load(f)
    return rules

# ---------------------------------------------------------------------
# Weights & Biases
# ---------------------------------------------------------------------
def init_wandb(cfg: Dict[str, Any], run_name: str) -> Optional[object]:
    """Inizializza W&B se abilitato nel config; ritorna il run oppure None."""
    wb = cfg.get("wandb", {})
    if not bool(wb.get("enabled", False)) or wandb is None:
        return None
    project = wb.get("project", "mtg-mha")
    entity = wb.get("entity", None)
    tags = wb.get("tags", [])
    run = wandb.init(project=project, entity=entity, name=run_name, tags=tags)  # type: ignore
    return run


# ---------------------------------------------------------------------
# Regex & helper di dominio (guidati dal config)
# ---------------------------------------------------------------------
def compile_regexes(cfg: Dict[str, Any]) -> Dict[str, Pattern]:
    """Compila i regex da cfg.constants.regex.*"""
    regex_cfg = cfg.get("constants", {}).get("regex", {})
    mana_pat = regex_cfg.get("mana_pattern", r"^$")
    pt_pat = regex_cfg.get("pt_pattern", r"^$")
    return {
        "mana": re.compile(mana_pat),
        "pt": re.compile(pt_pat),
    }


def _colors(cfg: Dict[str, Any]) -> Set[str]:
    return set(cfg.get("constants", {}).get("colors", []))


def parse_color_list(s: str, cfg: Dict[str, Any]) -> Set[str]:
    """Da 'W R' o 'U' a {'W','R'}. Accetta anche virgole."""
    colors = _colors(cfg)
    return {t for t in s.replace(",", " ").split() if t in colors}


def mana_color_symbols(mana_cost: str, cfg: Dict[str, Any]) -> Set[str]:
    """Estrae i simboli di colore presenti in '{2}{W}{R}' o '{W/U}'."""
    colors = _colors(cfg)
    syms: Set[str] = set()
    for part in re.findall(r"\{([^}]+)\}", mana_cost):
        for tok in part.split("/"):
            if tok in colors:
                syms.add(tok)
    return syms


def is_color_mana_consistent(color: str, mana_cost: str, cfg: Dict[str, Any]) -> bool:
    """
    Regola minimale: i simboli di mana usati devono essere sottoinsieme dei colori dichiarati.
    'color' è una stringa tipo 'W R', 'U' o '' per colorless.
    """
    declared = parse_color_list(color, cfg)
    used = mana_color_symbols(mana_cost, cfg)
    return used.issubset(declared) or (len(used) == 0 and len(declared) == 0)


# ---------------------------------------------------------------------
# I/O di base
# ---------------------------------------------------------------------
def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def write_text(path: str, s: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(s)


def read_lines(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


# ---------------------------------------------------------------------
# mapper
# ---------------------------------------------------------------------
class MhaMapper:
    def __init__(self, cfg):
        mapping_cfg = cfg.get("mapping", {})
        self.enabled = mapping_cfg.get("enabled", False)
        self.default_theme = mapping_cfg.get("default_theme", "Generic")
        self.default_character = mapping_cfg.get("default_character", "N/A")

        if not self.enabled:
            self.seed_by_name = {}
            self.seed_by_oracle_id = {}
            self.rules = []
            return

        seed_path = mapping_cfg.get("seed_path")
        keywords_path = mapping_cfg.get("keywords_path")

        self.seed_by_name, self.seed_by_oracle_id = load_mapping_seed(seed_path)
        self.rules = load_keyword_rules(keywords_path)

    def __call__(self, card: dict):
        """
        card è l’oggetto Scryfall completo.
        Ritorna (theme, character)
        """
        if not self.enabled:
            return self.default_theme, self.default_character

        # 1) match diretto da mapping_seed (più forte)
        name = card.get("name", "").lower()
        oracle_id = card.get("oracle_id")

        if name in self.seed_by_name:
            return self.seed_by_name[name]

        if oracle_id in self.seed_by_oracle_id:
            return self.seed_by_oracle_id[oracle_id]

        # 2) regole basate su keywords (heuristiche)
        theme, character = self._apply_keyword_rules(card)
        if theme is not None or character is not None:
            return (
                theme if theme is not None else self.default_theme,
                character if character is not None else self.default_character,
            )

        # 3) fallback
        return self.default_theme, self.default_character

    def _apply_keyword_rules(self, card: dict):
        text = (card.get("oracle_text") or "").lower()
        type_line = (card.get("type_line") or "").lower()
        colors = "".join(card.get("color_identity") or [])
        # qui la logica dipende dalla struttura di keywords.json

        for rule in self.rules:
            # es: { "id": "bakugo_rule", "colors": ["R"], "any_text": ["damage", "attack"], "theme": "U.A. Students", "character": "Bakugo Katsuki" }
            needed_colors = set(rule.get("colors", []))
            any_text = [kw.lower() for kw in rule.get("any_text", [])]

            if needed_colors and not needed_colors.issubset(set(colors)):
                continue

            if any_text and not any(kw in text or kw in type_line for kw in any_text):
                continue

            # se la regola è soddisfatta
            return rule.get("theme"), rule.get("character")

        return None, None
