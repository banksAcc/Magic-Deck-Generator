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
