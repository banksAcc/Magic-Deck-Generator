#!/usr/bin/env python
"""
a07_curate.py

Curation finale del set di carte MTG×MHA generate.

Input:
  - Un file JSONL "validated" prodotto da a06_generate_and_validate, del tipo:

    {
      "valid": true,
      "errors": [],
      "raw": "<|startofcard|>\\n...",
      "fields": {
        "theme": "...",
        "character": "...",
        "color": "W R",
        "type": "Legendary Creature | Hero",
        "rarity": "rare",
        "name": "...",
        "mana_cost": "{2}{W}{R}",
        "text": "...",
        "pt": "3/4"
      }
    }

Output:
  - Uno `*_scored.jsonl` con score e rank per ogni carta candidata.
  - Un set finale curato:
      outputs/final_set/final_<N>_<batchstem>.json
      outputs/final_set/final_<N>_<batchstem>_distributions.json

Logica:
  1. Legge il validated.jsonl e filtra solo le carte `valid == True`.
  2. Per ogni carta:
       - calcola `color_ok` usando is_color_mana_consistent(color, mana_cost, cfg)
       - calcola uno score semplice basato su color_ok + rarità (peso leggero)
  3. Ordina le carte per score (decrescente) e salva `_scored.jsonl`.
  4. Seleziona in modo greedy il set finale:
       - rispetta target per rarity (se definiti in config)
       - fa dedup su (name + text) usando distanza normalizzata (SequenceMatcher)
       - si ferma a `curation.final_set_size` carte (default 100)
  5. Esporta il set finale + distribuzioni per rarity/color/type.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter, defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional

from .a00_utils import load_config, init_wandb, is_color_mana_consistent


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class CandidateCard:
    raw: str
    fields: Dict[str, Any]
    syntax_ok: bool
    color_ok: bool
    score: float = 0.0
    rank: int = -1  # verrà popolato dopo l'ordinamento


# ---------------------------------------------------------------------------
# CLI & logger
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="a07 — Curation finale del set di carte MTG×MHA."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path al file di configurazione YAML.",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help=(
            "Path al file *_validated.jsonl da curare. "
            "Se omesso, viene preso l'ultimo *_validated.jsonl "
            "nella cartella generation.output_dir."
        ),
    )
    return parser.parse_args()


def setup_logger() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger("a07_curate")


# ---------------------------------------------------------------------------
# Helpers: file input / output
# ---------------------------------------------------------------------------

def find_latest_validated_file(cfg: Dict[str, Any], logger: logging.Logger) -> Path:
    """
    Cerca l'ultimo file *_validated.jsonl nella cartella generazioni
    (generation.output_dir, default: outputs/generations).
    """
    gen_cfg = cfg.get("generation", {})
    gen_dir = Path(gen_cfg.get("output_dir", "outputs/generations")).resolve()
    if not gen_dir.is_dir():
        raise FileNotFoundError(
            f"Directory delle generazioni non trovata: {gen_dir}. "
            "Specifica --input esplicitamente."
        )

    candidates = list(gen_dir.glob("*_validated.jsonl"))
    if not candidates:
        raise FileNotFoundError(
            f"Nessun file *_validated.jsonl trovato in {gen_dir}. "
            "Specifica --input esplicitamente."
        )

    # scegliamo il più recente per mtime
    latest = max(candidates, key=lambda p: p.stat().st_mtime)
    logger.info(f"Usa ultimo validated trovato: {latest}")
    return latest


def load_validated_cards(path: Path, logger: logging.Logger) -> List[Dict[str, Any]]:
    """
    Carica il JSONL di validated e ritorna una lista di record.
    """
    if not path.is_file():
        raise FileNotFoundError(f"File validated non trovato: {path}")

    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            records.append(rec)

    logger.info(f"Caricate {len(records)} righe da {path}")
    return records


def derive_output_paths(validated_path: Path, cfg: Dict[str, Any]) -> Tuple[Path, Path, Path]:
    """
    A partire da `batch_YYYYMMDD_HHMMSS_validated.jsonl` determina:

      - scored_path   -> stesso dir, suffisso _scored.jsonl
      - final_dir     -> curation.output_dir (default: outputs/final_set)
      - final_prefix  -> nome base per i file finali (final_<N>_<batchstem>)
    """
    # scored accanto al validated
    scored_path = validated_path.with_name(validated_path.stem.replace("_validated", "_scored") + ".jsonl")

    curation_cfg = cfg.get("curation", {})
    final_dir = Path(curation_cfg.get("output_dir", "outputs/final_set")).resolve()
    final_dir.mkdir(parents=True, exist_ok=True)

    # batch stem: es. "batch_20251118_190300" da "batch_20251118_190300_validated"
    stem = validated_path.stem
    if stem.endswith("_validated"):
        batch_stem = stem[: -len("_validated")]
    else:
        batch_stem = stem

    return scored_path, final_dir, batch_stem


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def normalize_rarity(r: str) -> str:
    return (r or "").strip().lower()


def build_rarity_weights(cfg: Dict[str, Any]) -> Dict[str, float]:
    """
    Costruisce un mapping rarity -> peso leggero per lo score.
    Ordine base: common < uncommon < rare < mythic.
    """
    constants_cfg = cfg.get("constants", {})
    rarities_cfg = [r.lower() for r in constants_cfg.get("rarities", [])]

    if not rarities_cfg:
        # fallback standard
        rarities_cfg = ["common", "uncommon", "rare", "mythic"]

    # peso crescente: common=0.0, uncommon=0.1, rare=0.2, mythic=0.3 ...
    weights: Dict[str, float] = {}
    step = 0.1
    for idx, r in enumerate(rarities_cfg):
        weights[r] = idx * step

    return weights


def compute_color_ok(fields: Dict[str, Any], cfg: Dict[str, Any]) -> bool:
    color = fields.get("color", "")
    mana = fields.get("mana_cost", "")
    # Usa la stessa regola del validator; se è disattivata in config,
    # possiamo considerare tutto ok per coerenza
    validator_cfg = cfg.get("validator", {})
    if not validator_cfg.get("color_mana_subset_rule", False):
        return True
    return is_color_mana_consistent(color, mana, cfg)


def build_candidates(
    validated_records: List[Dict[str, Any]],
    cfg: Dict[str, Any],
    logger: logging.Logger,
) -> List[CandidateCard]:
    """
    Trasforma i record del validated in CandidateCard, filtrando invalidi
    e calcolando color_ok.
    """
    candidates: List[CandidateCard] = []
    invalid_count = 0

    for rec in validated_records:
        if not rec.get("valid", False):
            invalid_count += 1
            continue

        fields = rec.get("fields", {}) or {}
        raw = rec.get("raw", "")

        color_ok = compute_color_ok(fields, cfg)

        candidates.append(
            CandidateCard(
                raw=raw,
                fields=fields,
                syntax_ok=True,
                color_ok=color_ok,
            )
        )

    logger.info(
        f"Candidati dopo filtro valid: {len(candidates)} "
        f"(scartati invalidi: {invalid_count})"
    )
    return candidates


def score_candidates(
    candidates: List[CandidateCard],
    cfg: Dict[str, Any],
    logger: logging.Logger,
) -> List[CandidateCard]:
    """
    Assegna uno score semplice a ogni carta:
      score = color_ok * 1.0 + rarity_weight

    dove rarity_weight è un offset leggero basato sulla rarità
    (common < uncommon < rare < mythic).
    """
    rarity_weights = build_rarity_weights(cfg)

    for c in candidates:
        rarity_raw = c.fields.get("rarity", "")
        rarity_norm = normalize_rarity(rarity_raw)
        rarity_weight = rarity_weights.get(rarity_norm, 0.0)

        base = 0.0
        if c.syntax_ok:
            base += 1.0
        if c.color_ok:
            base += 1.0

        # piccola differenza in base alla rarity
        c.score = base + rarity_weight

    # ordiniamo per score decrescente, poi per nome come tie-breaker
    def sort_key(card: CandidateCard):
        name = card.fields.get("name", "")
        return (-card.score, normalize_rarity(card.fields.get("rarity", "")), name)

    candidates_sorted = sorted(candidates, key=sort_key)

    for idx, c in enumerate(candidates_sorted):
        c.rank = idx + 1

    logger.info("Assegnati score e rank ai candidati.")
    return candidates_sorted


def export_scored_jsonl(
    candidates: List[CandidateCard],
    scored_path: Path,
    logger: logging.Logger,
) -> None:
    """
    Esporta tutti i candidati in formato JSONL, con score e rank.
    """
    with scored_path.open("w", encoding="utf-8") as f:
        for c in candidates:
            rec = {
                "raw": c.raw,
                "fields": c.fields,
                "syntax_ok": c.syntax_ok,
                "color_ok": c.color_ok,
                "score": c.score,
                "rank": c.rank,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    logger.info(f"File scored salvato in: {scored_path}")


# ---------------------------------------------------------------------------
# Dedup & selection
# ---------------------------------------------------------------------------

def normalized_distance(a: str, b: str) -> float:
    """
    Distanza normalizzata fra due stringhe basata su SequenceMatcher:
      dist = 1 - ratio
    (0 = identico, 1 = completamente diverso)
    """
    if not a and not b:
        return 0.0
    ratio = SequenceMatcher(None, a, b).ratio()
    return 1.0 - ratio


def is_near_duplicate(
    key_text: str,
    accepted_texts: List[str],
    threshold: float,
) -> bool:
    """
    Ritorna True se key_text è "troppo simile" a uno degli accepted_texts.

    Definizione: se normalized_distance(key_text, t) < threshold per qualche t.
    """
    for t in accepted_texts:
        dist = normalized_distance(key_text, t)
        if dist < threshold:
            return True
    return False

# ... (parte precedente invariata)

def load_curation_params(cfg: Dict[str, Any], logger: logging.Logger) -> Tuple[int, float, Dict[str, int]]:
    """
    Legge dal config i parametri principali di curation.
    
    MODIFICA: Imposta final_set_size a 100 di default (hard limit), 
    indipendentemente dalla somma dei target per rarità.
    """
    curation_cfg = cfg.get("curation", {})
    constants_cfg = cfg.get("constants", {})

    # dedup
    dedup_threshold = float(curation_cfg.get("dedup_threshold", 0.2))

    # rarities note
    rarities_cfg = [r.lower() for r in constants_cfg.get("rarities", [])]
    if not rarities_cfg:
        rarities_cfg = ["common", "uncommon", "rare", "mythic"]

    # target rarity
    target_counts_rarity_cfg = (
        curation_cfg.get("target_counts", {}).get("rarity", {}) or {}
    )
    # normalizziamo a lower
    target_counts_rarity: Dict[str, int] = {}
    for r, v in target_counts_rarity_cfg.items():
        target_counts_rarity[r.lower()] = int(v)

    # MODIFICA QUI:
    # Se final_set_size non è definito nel config, usiamo 100 come default fisso.
    # Non usiamo più "sum(target_counts)" come fallback, perché vogliamo tagliare a 100.
    final_set_size = int(curation_cfg.get("final_set_size", 100))

    # Se target_counts_rarity è vuoto, costruiamo una distribuzione uniforme
    # basata sul final_set_size deciso (100 o quello del config).
    if not target_counts_rarity:
        base = final_set_size // len(rarities_cfg)
        rem = final_set_size % len(rarities_cfg)
        for i, r in enumerate(rarities_cfg):
            target_counts_rarity[r] = base + (1 if i < rem else 0)

    logger.info(f"Final set size (Cutoff): {final_set_size}")
    logger.info(f"Dedup threshold: {dedup_threshold}")
    logger.info(f"Target massimi per rarity: {target_counts_rarity}")

    return final_set_size, dedup_threshold, target_counts_rarity


def select_final_set(
    candidates_sorted: List[CandidateCard],
    final_set_size: int,
    dedup_threshold: float,
    target_counts_rarity: Dict[str, int],
    logger: logging.Logger,
) -> List[CandidateCard]:
    """
    Selezione greedy del set finale:
      - rispetta i target per rarity (se specificati)
      - si ferma ESATTAMENTE a final_set_size carte (es. 100).
    """
    accepted: List[CandidateCard] = []
    accepted_texts: List[str] = []  # key_text di riferimento
    rarity_counts: Dict[str, int] = defaultdict(int)

    for c in candidates_sorted:
        # Controllo prioritario: siamo arrivati al limite totale?
        if len(accepted) >= final_set_size:
            logger.info(f"Raggiunto il limite massimo del set ({final_set_size}). Interruzione selezione.")
            break

        fields = c.fields
        rarity_raw = fields.get("rarity", "")
        rarity = normalize_rarity(rarity_raw)
        if not rarity:
            # se manca la rarity, saltiamo
            continue

        # target per questa rarity
        target_for_r = target_counts_rarity.get(rarity, 0)
        
        # Se abbiamo già riempito il bucket per questa rarità, passiamo alla prossima carta
        # (Nota: se vuoi forzare 100 carte anche sforando le rarità, rimuovi questo if)
        if target_for_r > 0 and rarity_counts[rarity] >= target_for_r:
            continue

        # dedup su name+text
        name = fields.get("name", "").strip()
        text = fields.get("text", "").strip()
        key_text = f"{name} :: {text}"
        if is_near_duplicate(key_text, accepted_texts, dedup_threshold):
            # troppo simile a qualcosa di già accettato
            continue

        # OK, accettiamo
        accepted.append(c)
        accepted_texts.append(key_text)
        rarity_counts[rarity] += 1

    logger.info(f"Carte accettate nel set finale: {len(accepted)}")
    logger.info(f"Distribuzione rarity nel set finale: {dict(rarity_counts)}")
    return accepted

# ... (resto del file invariato)
# ---------------------------------------------------------------------------
# Export finale (set + distribuzioni)
# ---------------------------------------------------------------------------

def compute_distributions(cards: List[CandidateCard], cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calcola distribuzioni per rarity, color e macro-type.

    macro-types: presi da constants.type_macros, match case-insensitive su 'type'.
    """
    constants_cfg = cfg.get("constants", {})
    macro_types_cfg = [m.lower() for m in constants_cfg.get("type_macros", [])]

    rarity_counts: Counter = Counter()
    color_counts: Counter = Counter()
    macro_type_counts: Counter = Counter()

    for c in cards:
        f = c.fields

        rarity = normalize_rarity(f.get("rarity", ""))
        if rarity:
            rarity_counts[rarity] += 1

        color = f.get("color", "").strip()
        if color:
            color_counts[color] += 1

        type_line = f.get("type", "").lower()
        mt_found = False
        for mt in macro_types_cfg:
            if mt and mt in type_line:
                macro_type_counts[mt] += 1
                mt_found = True
        if not mt_found and type_line:
            macro_type_counts["other"] += 1

    return {
        "rarity": dict(rarity_counts),
        "color": dict(color_counts),
        "macro_type": dict(macro_type_counts),
    }


def export_final_set(
    accepted: List[CandidateCard],
    final_dir: Path,
    batch_stem: str,
    final_set_size: int,
    cfg: Dict[str, Any],
    logger: logging.Logger,
) -> Tuple[Path, Path]:
    """
    Esporta il set finale e le distribuzioni in due file JSON.

    Ritorna (path_set, path_distributions).
    """
    final_name = f"final_{final_set_size}_{batch_stem}.json"
    final_path = final_dir / final_name

    payload = []
    for c in accepted:
        payload.append(
            {
                "fields": c.fields,
                "raw": c.raw,
                "score": c.score,
                "rank": c.rank,
            }
        )

    final_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # Distribuzioni
    distributions = compute_distributions(accepted, cfg)
    dist_name = f"final_{final_set_size}_{batch_stem}_distributions.json"
    dist_path = final_dir / dist_name
    dist_path.write_text(
        json.dumps(distributions, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    logger.info(f"Set finale salvato in: {final_path}")
    logger.info(f"Distribuzioni salvate in: {dist_path}")
    return final_path, dist_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    logger = setup_logger()

    logger.info(f"Caricamento config da {args.config}")
    cfg = load_config(args.config)

    # Input file
    if args.input:
        validated_path = Path(args.input).resolve()
    else:
        validated_path = find_latest_validated_file(cfg, logger)

    scored_path, final_dir, batch_stem = derive_output_paths(validated_path, cfg)

    # Carica validated
    validated_records = load_validated_cards(validated_path, logger)

    # Candidati (solo valid=True)
    candidates = build_candidates(validated_records, cfg, logger)

    if not candidates:
        logger.error("Nessun candidato valido disponibile. Esco.")
        return

    # Scoring
    scored_candidates = score_candidates(candidates, cfg, logger)

    # Esporta scored
    export_scored_jsonl(scored_candidates, scored_path, logger)

    # Parametri di curation
    final_set_size, dedup_threshold, target_counts_rarity = load_curation_params(cfg, logger)

    # Selezione finale
    accepted = select_final_set(
        scored_candidates,
        final_set_size=final_set_size,
        dedup_threshold=dedup_threshold,
        target_counts_rarity=target_counts_rarity,
        logger=logger,
    )

    if not accepted:
        logger.warning("Nessuna carta accettata nel set finale. Controllare i target / threshold.")

    # Export finale
    final_set_path, dist_path = export_final_set(
        accepted,
        final_dir=final_dir,
        batch_stem=batch_stem,
        final_set_size=final_set_size,
        cfg=cfg,
        logger=logger,
    )

    # W&B (lite)
    run = init_wandb(cfg, run_name="curate-a07")
    if run is not None:
        metrics = {
            "curate.final_size": len(accepted),
            "curate.target_size": final_set_size,
        }
        # rarity distribution
        dists = compute_distributions(accepted, cfg)
        for r, v in dists.get("rarity", {}).items():
            metrics[f"curate.rarity.{r}"] = v
        for col, v in dists.get("color", {}).items():
            metrics[f"curate.color.{col}"] = v
        run.log(metrics)
        run.finish()

    logger.info("Curation completata.")


if __name__ == "__main__":
    main()
