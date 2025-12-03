#!/usr/bin/env python
"""
make_ppl_plots.py

Genera due grafici per la relazione:

3. Confronto qualitativo fra le perplexity di GPT-2 small e Mistral 7B QLoRA.
4. Esempio di relazione qualitativa fra perplexity e tasso di carte validate.

Formati attesi dei CSV:

A) Confronto perplexity (ppl_compare.csv)
   colonne:
       model,perplexity

   Esempio:
       model,perplexity
       gpt2-small,34.8
       mistral-7b-qlora,22.5

B) Relazione perplexity vs validation rate (ppl_vs_valid.csv)
   colonne:
       run_name,perplexity,validation_rate

   Esempio:
       run_name,perplexity,validation_rate
       gpt2_run_seed1,40.2,0.44
       mistral_run_seed1,23.8,0.62
       ...

Uso:

    python make_ppl_plots.py \
        --ppl-compare logs/ppl_compare.csv \
        --ppl-vs-valid logs/ppl_vs_valid.csv \
        --output-dir figures_ppl
"""

import argparse
import csv
import os
from typing import List, Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np


# --------------------------------------------------
# Lettura CSV
# --------------------------------------------------

def read_ppl_compare(path: str) -> Dict[str, float]:
    """
    Legge un CSV con colonne:
        model,perplexity

    Restituisce un dict: {model_name: ppl}.
    """
    result: Dict[str, float] = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            model = row["model"]
            ppl = float(row["perplexity"])
            result[model] = ppl
    return result


def read_ppl_vs_valid(path: str) -> Tuple[List[float], List[float], List[str]]:
    """
    Legge un CSV con colonne:
        run_name,perplexity,validation_rate

    Restituisce:
        perplexities, validation_rates, run_names
    """
    perplexities: List[float] = []
    validation_rates: List[float] = []
    run_names: List[str] = []

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            run_names.append(row["run_name"])
            perplexities.append(float(row["perplexity"]))
            validation_rates.append(float(row["validation_rate"]))
    return perplexities, validation_rates, run_names


# --------------------------------------------------
# Plot helper
# --------------------------------------------------

def plot_ppl_compare(ppl_dict: Dict[str, float],
                     title: str,
                     output_path: str) -> None:
    """
    Grafico a barre per confrontare le perplexity dei modelli.
    """
    models = list(ppl_dict.keys())
    values = [ppl_dict[m] for m in models]
    x = np.arange(len(models))

    plt.figure(figsize=(6, 4))
    plt.bar(x, values, width=0.6)
    plt.xticks(x, models, rotation=15, ha="right")
    plt.ylabel("Perplexity")
    plt.title(title)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.margins(x=0.05)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_ppl_vs_valid(perplexities: List[float],
                      validation_rates: List[float],
                      run_names: List[str],
                      title: str,
                      output_path: str) -> None:
    """
    Scatter plot di perplexity vs tasso di carte validate,
    con una retta di regressione lineare (puramente qualitativa).
    """
    x = np.array(perplexities)
    y = np.array(validation_rates)

    plt.figure(figsize=(6, 4))
    plt.scatter(x, y)

    # Annotazioni opzionali con i nomi dei run (solo se pochi punti)
    if len(run_names) <= 10:
        for xi, yi, name in zip(x, y, run_names):
            plt.annotate(name, (xi, yi), textcoords="offset points",
                         xytext=(3, 3), fontsize=8)

    # Fit lineare: y = a*x + b (solo se almeno 2 punti)
    if len(x) >= 2:
        a, b = np.polyfit(x, y, deg=1)
        x_line = np.linspace(x.min(), x.max(), 100)
        y_line = a * x_line + b
        plt.plot(x_line, y_line, linestyle="--")

    plt.xlabel("Perplexity")
    plt.ylabel("Tasso di carte validate")
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Genera grafici di confronto perplexity e relazione con tasso di carte validate."
    )
    parser.add_argument(
        "--ppl-compare",
        type=str,
        required=True,
        help="CSV con confronto di perplexity (model,perplexity)."
    )
    parser.add_argument(
        "--ppl-vs-valid",
        type=str,
        required=True,
        help="CSV con relazione perplexity vs tasso di carte validate "
             "(run_name,perplexity,validation_rate)."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="figures_ppl",
        help="Directory di output per i grafici (default: figures_ppl)."
    )

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # 3) Confronto qualitativo perplexity GPT-2 vs Mistral
    ppl_dict = read_ppl_compare(args.ppl_compare)
    out_ppl_compare = os.path.join(args.output_dir, "ppl_compare_gpt2_vs_mistral.png")
    plot_ppl_compare(
        ppl_dict,
        title="Confronto comparativo delle perplexity",
        output_path=out_ppl_compare,
    )
    print(f"[OK] Grafico confronto perplexity salvato in: {out_ppl_compare}")

    # 4) Relazione qualitativa perplexity vs tasso di carte validate
    perplexities, validation_rates, run_names = read_ppl_vs_valid(args.ppl_vs_valid)
    out_ppl_valid = os.path.join(args.output_dir, "ppl_vs_validation_rate.png")
    plot_ppl_vs_valid(
        perplexities,
        validation_rates,
        run_names,
        title="Relazione qualitativa fra perplexity e tasso di carte validate",
        output_path=out_ppl_valid,
    )
    print(f"[OK] Grafico perplexity vs tasso di carte validate salvato in: {out_ppl_valid}")


if __name__ == "__main__":
    main()
