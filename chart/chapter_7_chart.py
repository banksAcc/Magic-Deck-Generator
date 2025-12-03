import matplotlib.pyplot as plt
import pandas as pd
import json
import numpy as np

# Impostazioni di stile per i grafici
plt.style.use('ggplot')

# ==========================================
# PARTE 1: Grafico Carte Generate vs Valide
# ==========================================

def plot_pass_rate():
    # Dati forniti
    total_generated = 1500
    gpt2_pass_rate = 92.24  # in %
    mistral_pass_rate = 99.99 # in %

    # Calcolo dei numeri assoluti
    gpt2_valid_count = int(total_generated * (gpt2_pass_rate / 100))
    mistral_valid_count = int(total_generated * (mistral_pass_rate / 100))

    # Definizione delle etichette e dei valori per le 3 colonne
    labels = ['Totale Generate', 'GPT-2 Valide', 'Mistral Valide']
    counts = [total_generated, gpt2_valid_count, mistral_valid_count]
    colors = ['gray', '#1f77b4', '#ff7f0e'] # Grigio, Blu, Arancione

    # Creazione del grafico
    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, counts, color=colors, alpha=0.8)

    # Aggiunta delle etichette sopra le barre
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 20, int(yval), 
                 ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        # Aggiungi la percentuale se non è la colonna del totale
        if yval != total_generated:
            perc = (yval / total_generated) * 100
            plt.text(bar.get_x() + bar.get_width()/2, yval - 100, f"{perc:.2f}%", 
                     ha='center', va='top', color='white', fontsize=11, fontweight='bold')

    plt.title(f'Confronto Carte Generate vs Valide (su {total_generated} carte)', fontsize=15)
    plt.ylabel('Numero di Carte')
    plt.ylim(0, total_generated + 200) # Un po' di margine sopra
    
    plt.tight_layout()
    plt.show()

# ==========================================
# PARTE 2: Analisi degli Errori dai JSONL
# ==========================================

def parse_errors(file_path):
    """
    Legge un file JSONL e conta le tipologie di errori.
    Restituisce un dizionario {tipo_errore: conteggio} e il totale degli errori.
    """
    error_counts = {}
    total_errors_found = 0
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    record = json.loads(line)
                    # Controlliamo se la carta non è valida e ha una lista di errori
                    if not record.get("valid", True) and "errors" in record:
                        for error in record["errors"]:
                            error_counts[error] = error_counts.get(error, 0) + 1
                            total_errors_found += 1
                except json.JSONDecodeError:
                    continue # Salta righe malformate
    except FileNotFoundError:
        print(f"ATTENZIONE: Il file {file_path} non è stato trovato.")
        return {}, 0

    return error_counts, total_errors_found

def plot_common_errors(path_gpt2, path_mistral):
    # 1. Estrazione dati
    gpt2_errors, gpt2_total = parse_errors(path_gpt2)
    mistral_errors, mistral_total = parse_errors(path_mistral)

    if not gpt2_errors and not mistral_errors:
        print("Nessun dato trovato o percorsi file non validi.")
        return

    # 2. Creazione DataFrame per gestire i dati facilmente
    # Uniamo tutti i tipi di errore trovati in entrambi i modelli
    all_error_types = set(gpt2_errors.keys()) | set(mistral_errors.keys())
    
    data = []
    for err in all_error_types:
        # Calcolo percentuale relativa al totale degli errori del rispettivo modello
        # Se vuoi la percentuale sul totale delle carte generate (1500), cambia il divisore qui sotto.
        # Qui uso % rispetto agli errori totali trovati come da prassi per "errori più comuni".
        val_gpt = (gpt2_errors.get(err, 0) / gpt2_total * 100) if gpt2_total > 0 else 0
        val_mistral = (mistral_errors.get(err, 0) / mistral_total * 100) if mistral_total > 0 else 0
        
        data.append({
            'Errore': err,
            'GPT-2 (%)': val_gpt,
            'Mistral (%)': val_mistral,
            'Totale (%)': val_gpt + val_mistral # Usato solo per ordinare
        })

    df = pd.DataFrame(data)
    
    # Ordiniamo per gli errori più frequenti in generale e prendiamo i top 10 per leggibilità
    df = df.sort_values(by='Totale (%)', ascending=False).head(10)
    df = df.sort_values(by='Totale (%)', ascending=True) # Re-sort per il grafico orizzontale o verticale

    # 3. Plotting
    x = np.arange(len(df))
    width = 0.35

    plt.figure(figsize=(12, 8))
    
    # Barre GPT-2
    plt.barh(x - width/2, df['GPT-2 (%)'], width, label='GPT-2', color='#1f77b4')
    # Barre Mistral
    plt.barh(x + width/2, df['Mistral (%)'], width, label='Mistral', color='#ff7f0e')

    plt.xlabel('Percentuale di incidenza dell\'errore (%)')
    plt.title('Top Errori più Comuni: GPT-2 vs Mistral')
    plt.yticks(x, df['Errore'])
    plt.legend()
    
    # Griglia verticale per facilitare la lettura
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.show()

# ==========================================
# ESECUZIONE
# ==========================================

if __name__ == "__main__":
    # 1. Esegui il primo grafico
    print("Generazione grafico pass rate...")
    plot_pass_rate()

    # 2. Configura i percorsi ed esegui il secondo grafico
    # --- MODIFICA QUI SOTTO CON I TUOI PERCORSI ---
    file_path_gpt2 = "outputs/generations/gpt_batch_20251203_183115_validated.jsonl" 
    file_path_mistral = "outputs/generations/mistrlai_batch_20251203_220512_validated.jsonl"
    
    # Nota: Se non hai i file ora, questa funzione stamperà un errore di "File non trovato"
    # ma il codice è corretto per quando avrai i file.
    print("\nGenerazione grafico errori...")
    plot_common_errors(file_path_gpt2, file_path_mistral)