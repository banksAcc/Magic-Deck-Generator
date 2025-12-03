#!/usr/bin/env python
"""
make_stats_plots.py

Produce i seguenti grafici a partire da un dataset filtrato in formato "Standard":

<|startofcard|>
theme: ...
character: ...
color: W R
type: Legendary Creature | Hero
rarity: rare
<|gen_card|>
name: ...
mana_cost: ...
text: ...
pt: 3/3
<|endofcard|>

Grafici prodotti:
1. Distribuzione dei macro-tipi (Creature, Instant, Sorcery, ...)
2. Distribuzione categorie di colore (monocolore, multicolore, incolore)
3. Distribuzione delle rarità (common, uncommon, rare, mythic)
4. Distribuzione della lunghezza del testo in token (in base al tokenizer HF scelto)

Uso:
    python make_stats_plots.py \\
        --input data/processed/train.txt \\
        --output-dir outputs/figures \\
        --tokenizer gpt2

Se non specifichi nulla:
    --tokenizer gpt2
    --output-dir ./figures
"""

import argparse
import os
from collections import Counter
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
from transformers import AutoTokenizer


MACRO_TYPES = ["Creature", "Instant", "Sorcery",
               "Enchantment", "Artifact", "Planeswalker", "Land"]

RARITY_ORDER = ["common", "uncommon", "rare", "mythic"]


# ---------------------------
# Parsing del file di dataset
# ---------------------------

def parse_cards(path: str) -> List[Dict[str, str]]:
    """
    Parsifica il file testuale in blocchi carta.

    Ritorna una lista di dict, es.:
    {
        "color": "W R",
        "type": "Legendary Creature | Hero",
        "rarity": "rare",
        "text": "Flying ...",
        ...
    }
    """
    cards: List[Dict[str, str]] = []
    current: Dict[str, str] = {}
    in_block = False

    with open(path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.rstrip("\n")

            if line.strip() == "<|startofcard|>":
                in_block = True
                current = {}
                continue

            if line.strip() == "<|endofcard|>":
                if in_block and current:
                    cards.append(current)
                in_block = False
                current = {}
                continue

            if not in_block:
                continue

            # Ignora token speciali senza ":" (es. <|gen_card|>)
            if ":" not in line:
                continue

            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip()
            if key:
                current[key] = value

    return cards


# ---------------------------
# Feature engineering
# ---------------------------

def get_macro_type(type_line: str) -> str:
    """Estrae il macro-tipo principale dalla type line."""
    for macro in MACRO_TYPES:
        if macro in type_line:
            return macro
    return "Other"


def get_color_category(color_field: str) -> str:
    """
    Raggruppa le combinazioni di colore in:
    - "colorless"
    - "monocolor"
    - "multicolor"
    """
    if not color_field:
        return "colorless"

    tokens = [c for c in color_field.split() if c in {"W", "U", "B", "R", "G", "C"}]
    # Nessun colore valido o solo C -> incolore
    if not tokens or tokens == ["C"]:
        return "colorless"

    unique_colors = {c for c in tokens if c in {"W", "U", "B", "R", "G"}}
    if len(unique_colors) == 0:
        return "colorless"
    if len(unique_colors) == 1:
        return "monocolor"
    return "multicolor"


def compute_text_lengths(cards: List[Dict[str, str]],
                         tokenizer_name: str) -> List[int]:
    """
    Calcola la lunghezza in token del campo `text`
    usando un tokenizer HuggingFace (es. 'gpt2', 'mistral-7b-v0.3', ...).
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    lengths: List[int] = []

    for c in cards:
        text = c.get("text", "").strip()
        if not text:
            continue
        enc = tokenizer(text, add_special_tokens=False)
        lengths.append(len(enc["input_ids"]))

    return lengths


# ---------------------------
# Funzioni per i grafici
# ---------------------------

def plot_bar(counter: Counter,
             title: str,
             xlabel: str,
             ylabel: str,
             output_path: str,
             order: Optional[List[str]] = None) -> None:
    """
    Grafico a barre semplice da un Counter.

    Se `order` è fornito, usa quell'ordine di etichette.
    """
    if order is None:
        labels = list(counter.keys())
    else:
        labels = order

    values = [counter.get(l, 0) for l in labels]

    # posizioni esplicite sull’asse x
    x = list(range(len(labels)))

    plt.figure(figsize=(8, 4))          # grafico un po’ più largo
    plt.bar(x, values, width=0.6)       # barre più strette
    plt.xticks(x, labels, rotation=20,  # etichette leggermente ruotate
               ha="right")              # allineate a destra per meno overlap

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.margins(x=0.05)                 # piccolo margine ai lati
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_hist(data: List[int],
              title: str,
              xlabel: str,
              ylabel: str,
              output_path: str,
              bins: int = 30) -> None:
    """Istogramma per distribuzione di lunghezze."""
    plt.figure(figsize=(6, 4))
    plt.hist(data, bins=bins)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


# ---------------------------
# Main
# ---------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Genera grafici di distribuzione per il dataset di carte MTG."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Percorso al file di dataset filtrato (formato blocchi testo)."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="figures",
        help="Directory di output per i grafici (default: ./figures)."
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="gpt2",
        help="Nome del tokenizer HF per calcolare la lunghezza in token del testo (default: gpt2)."
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[INFO] Caricamento dataset da: {args.input}")
    cards = parse_cards(args.input)
    print(f"[INFO] Carte lette: {len(cards)}")

    # 1) Distribuzione macro-tipi
    macro_types = [
        get_macro_type(c.get("type", "")) for c in cards
        if "type" in c
    ]
    type_counter = Counter(macro_types)
    out_types = os.path.join(args.output_dir, "dist_macro_types.png")
    plot_bar(
        type_counter,
        title="Distribuzione dei macro-tipi di carta",
        xlabel="Macro-tipo",
        ylabel="Numero di carte",
        output_path=out_types,
        order=MACRO_TYPES + (["Other"] if "Other" in type_counter else [])
    )
    print(f"[OK] Grafico macro-tipi salvato in: {out_types}")

    # 2) Distribuzione combinazioni di colori (mono/multi/colorless)
    color_categories = [
        get_color_category(c.get("color", ""))
        for c in cards
    ]
    color_cat_counter = Counter(color_categories)
    out_colors = os.path.join(args.output_dir, "dist_color_categories.png")
    plot_bar(
        color_cat_counter,
        title="Distribuzione combinazioni di colori",
        xlabel="Categoria di colore",
        ylabel="Numero di carte",
        output_path=out_colors,
        order=["colorless", "monocolor", "multicolor"]
    )
    print(f"[OK] Grafico categorie di colore salvato in: {out_colors}")

    # 3) Distribuzione rarità
    rarities = [
        c.get("rarity", "").strip().lower()
        for c in cards
        if "rarity" in c
    ]
    rarity_counter = Counter(rarities)
    out_rarity = os.path.join(args.output_dir, "dist_rarity.png")
    plot_bar(
        rarity_counter,
        title="Distribuzione delle rarità",
        xlabel="Rarità",
        ylabel="Numero di carte",
        output_path=out_rarity,
        order=RARITY_ORDER
    )
    print(f"[OK] Grafico rarità salvato in: {out_rarity}")

    # 4) Distribuzione lunghezze del testo (in token)
    print(f"[INFO] Calcolo lunghezze testo in token con tokenizer: {args.tokenizer}")
    text_lengths = compute_text_lengths(cards, tokenizer_name=args.tokenizer)

    if text_lengths:
        out_text_len = os.path.join(args.output_dir, "dist_text_lengths_tokens.png")
        plot_hist(
            text_lengths,
            title="Distribuzione delle lunghezze del testo (in token)",
            xlabel="Lunghezza (token)",
            ylabel="Numero di carte",
            output_path=out_text_len,
            bins=30
        )
        print(f"[OK] Grafico lunghezze testo salvato in: {out_text_len}")
    else:
        print("[WARN] Nessun campo 'text' trovato; grafico delle lunghezze non generato.")


if __name__ == "__main__":
    main()
