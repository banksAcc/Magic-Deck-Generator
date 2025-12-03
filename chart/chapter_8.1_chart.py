import matplotlib.pyplot as plt
import pandas as pd
import json
import numpy as np

# Impostazioni di stile
plt.style.use('ggplot')

# ==========================================
# FUNZIONE DI UTILITÀ: CARICAMENTO DATI
# ==========================================

def load_scores_from_jsonl(file_path):
    """Estrae la lista degli score dai file JSONL."""
    scores = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    record = json.loads(line)
                    # Cerchiamo lo score numerico
                    if 'score' in record and isinstance(record['score'], (int, float)):
                        scores.append(record['score'])
                except json.JSONDecodeError:
                    continue
    except FileNotFoundError:
        print(f"File non trovato: {file_path}")
    return scores

# ==========================================
# 1. GRAFICO MEDIA SCORE (JSONL - Tutte le carte)
#    Bar Chart con 2 colonne
# ==========================================

def plot_mean_score_jsonl(path_gpt, path_mistral):
    # 1. Caricamento dati
    scores_gpt = load_scores_from_jsonl(path_gpt)
    scores_mistral = load_scores_from_jsonl(path_mistral)
    
    # 2. Calcolo Medie
    # Gestiamo il caso di liste vuote per evitare errori
    mean_gpt = np.mean(scores_gpt) if scores_gpt else 0
    mean_mistral = np.mean(scores_mistral) if scores_mistral else 0
    
    # 3. Preparazione Dati Plot
    labels = ['GPT-2 (All Generated)', 'Mistral (All Generated)']
    means = [mean_gpt, mean_mistral]
    colors = ['#1f77b4', '#ff7f0e'] # Blu e Arancio

    plt.figure(figsize=(8, 6))
    
    # Creazione Barre
    bars = plt.bar(labels, means, color=colors, width=0.5, alpha=0.9)

    # 4. Aggiunta Etichette sopra le barre
    for bar in bars:
        height = bar.get_height()
        # Offset verticale per staccare il testo dalla barra
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.2f}',
                 ha='center', va='bottom', fontsize=14, fontweight='bold', color='black')

    plt.title('Score Medio: Tutte le carte generate (JSONL)', fontsize=15)
    plt.ylabel('Score Medio')
    
    # Impostiamo il limite Y un po' più alto della barra più alta per estetica
    if means:
        plt.ylim(0, max(means) * 1.2)
        
    plt.grid(axis='x') # Rimuove griglia verticale per pulizia
    plt.tight_layout()
    plt.show()

# ==========================================
# 2. GRAFICO MEDIA PLAYABLE SET (CSV - Carte Filtrate)
#    Bar Chart con 2 colonne
# ==========================================

def plot_playable_average(csv_path_gpt, csv_path_mistral):
    # 1. Caricamento CSV
    try:
        df_gpt = pd.read_csv(csv_path_gpt)
        df_mistral = pd.read_csv(csv_path_mistral)
    except FileNotFoundError:
        print("Errore: Uno dei file CSV non è stato trovato.")
        return

    # 2. Calcolo Medie
    try:
        mean_gpt = df_gpt['score'].mean()
        mean_mistral = df_mistral['score'].mean()
    except KeyError:
        print("Errore: La colonna 'score' non è presente nel CSV.")
        return

    # 3. Preparazione Dati Plot
    labels = ['GPT-2 (Playable)', 'Mistral (Playable)']
    means = [mean_gpt, mean_mistral]
    colors = ['#1f77b4', '#ff7f0e'] # Blu e Arancio

    plt.figure(figsize=(8, 6))
    bars = plt.bar(labels, means, color=colors, width=0.5, alpha=0.9)

    # 4. Aggiunta valore sopra le barre
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.2f}',
                 ha='center', va='bottom', fontsize=14, fontweight='bold', color='black')

    plt.title('Score Medio: Playable Set Filtrato (CSV)', fontsize=15)
    plt.ylabel('Score Medio')
    
    if means:
        plt.ylim(0, max(means) * 1.2)
    
    plt.grid(axis='x')
    plt.tight_layout()
    plt.show()

# ==========================================
# ESECUZIONE
# ==========================================

if __name__ == "__main__":
    # PERCORSI FILE (Aggiornati con i tuoi path)
    
    # Path per Grafico 1 (JSONL)
    path_jsonl_gpt = "outputs/generations/gpt_batch_20251203_183115_scored.jsonl"
    path_jsonl_mistral = "outputs/generations/gpt_batch_20251203_183115_scored.jsonl"
    
    # Path per Grafico 2 (CSV)
    path_csv_gpt = "outputs/final_set/gpt_finel_manual_score.csv"
    path_csv_mistral = "outputs/final_set/mistral_finel_manual_score.csv"

    # Esecuzione funzioni
    print("--- Generazione Grafico 1: Media da JSONL ---")
    plot_mean_score_jsonl(path_jsonl_gpt, path_jsonl_mistral)

    print("\n--- Generazione Grafico 2: Media da CSV ---")
    # Nota: se i file CSV non esistono ancora, vedrai un messaggio di errore gestito.
    plot_playable_average(path_csv_gpt, path_csv_mistral)