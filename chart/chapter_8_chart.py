import json
import pandas as pd
import matplotlib.pyplot as plt

# Impostazioni grafiche
plt.style.use('ggplot')

# ==========================================
# 1. FUNZIONI DI PULIZIA (NORMALIZZAZIONE)
# ==========================================

def normalize_color(raw_color):
    """
    Incasella l'output del modello nelle categorie standard.
    """
    s = str(raw_color).upper()
    colors_found = set()
    if 'W' in s or 'WHITE' in s: colors_found.add('White')
    if 'U' in s or 'BLUE' in s:  colors_found.add('Blue')
    if 'B' in s or 'BLACK' in s: colors_found.add('Black')
    if 'R' in s or 'RED' in s:   colors_found.add('Red')
    if 'G' in s or 'GREEN' in s: colors_found.add('Green')
    
    count = len(colors_found)
    if count > 1: return 'Multicolor'
    elif count == 1: return list(colors_found)[0]
    else: return 'Colorless'

def normalize_type(raw_type):
    """
    Estrae il tipo principale (Creature, Instant, ecc).
    """
    s = str(raw_type).lower()
    if 'creature' in s or 'summon' in s: return 'Creature'
    if 'land' in s: return 'Land'
    if 'planeswalker' in s: return 'Planeswalker'
    if 'instant' in s: return 'Instant'
    if 'sorcery' in s: return 'Sorcery'
    if 'enchantment' in s: return 'Enchantment'
    if 'artifact' in s: return 'Artifact'
    return 'Other'

def normalize_rarity(raw_rarity):
    """
    Normalizza la rarità gestendo maiuscole/minuscole e sinonimi.
    """
    s = str(raw_rarity).lower().strip()
    
    # L'ordine dei controlli è importante (es. 'mythic' prima di 'rare')
    if 'mythic' in s: return 'Mythic Rare'
    if 'rare' in s: return 'Rare'
    if 'uncommon' in s: return 'Uncommon'
    if 'common' in s: return 'Common'
    if 'basic' in s: return 'Basic Land' # A volte appare come rarità
    
    return 'Other' # Errori o allucinazioni del modello

# ==========================================
# 2. CARICAMENTO DATI
# ==========================================

def load_clean_data(file_path, model_name):
    data_list = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    record = json.loads(line)
                    fields = record.get('fields', {})
                    
                    # Recuperiamo i dati grezzi
                    raw_color = fields.get('color', '')
                    raw_type = fields.get('type', '')
                    raw_rarity = fields.get('rarity', '')
                    
                    # Applichiamo la normalizzazione
                    clean_color = normalize_color(raw_color)
                    clean_type = normalize_type(raw_type)
                    clean_rarity = normalize_rarity(raw_rarity)

                    data_list.append({
                        'clean_color': clean_color,
                        'clean_type': clean_type,
                        'clean_rarity': clean_rarity,
                        'model': model_name
                    })
                except json.JSONDecodeError:
                    continue
    except FileNotFoundError:
        print(f"File non trovato: {file_path}")
        return pd.DataFrame()

    return pd.DataFrame(data_list)

# ==========================================
# 3. PLOTTING GENERICO
# ==========================================

def plot_grouped_bar(df, column, title, xlabel, custom_order=None):
    # Calcolo percentuali
    counts = df.groupby(['model', column])[column].count()
    totals = df.groupby('model')[column].count()
    percentages = counts.div(totals, level='model') * 100
    table = percentages.unstack(level='model').fillna(0)
    
    # Ordinamento
    if custom_order:
        # Mantiene solo quelli presenti nei dati ma nell'ordine specificato
        existing = [x for x in custom_order if x in table.index]
        remaining = [x for x in table.index if x not in existing]
        table = table.reindex(existing + remaining)
    else:
        # Ordina per frequenza decrescente
        table['sort_val'] = table.sum(axis=1)
        table = table.sort_values('sort_val', ascending=False).drop(columns='sort_val')

    # Creazione grafico
    ax = table.plot(kind='bar', figsize=(12, 6), width=0.8, color=['#1f77b4', '#ff7f0e'])
    
    plt.title(title, fontsize=14)
    plt.ylabel('Percentuale (%)')
    plt.xlabel(xlabel)
    plt.xticks(rotation=45)
    plt.legend(title='Modello')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Etichette valori
    for p in ax.patches:
        if p.get_height() > 0:
            ax.annotate(f"{p.get_height():.1f}%", 
                        (p.get_x() + p.get_width() / 2., p.get_height()), 
                        ha='center', va='bottom', fontsize=8, xytext=(0, 2), 
                        textcoords='offset points')
    
    plt.tight_layout()
    plt.show()

# ==========================================
# 4. ESECUZIONE
# ==========================================

if __name__ == "__main__":

    path_gpt2 = "outputs/generations/gpt_batch_20251203_183115_validated.jsonl" 
    path_mistral = "outputs/generations/mistrlai_batch_20251203_220512_validated.jsonl"
    
    print("Elaborazione dati in corso...")
    df_gpt = load_clean_data(path_gpt2, "GPT-2")
    df_mistral = load_clean_data(path_mistral, "Mistral")
    
    if not df_gpt.empty and not df_mistral.empty:
        full_df = pd.concat([df_gpt, df_mistral])
        
        # 1. Grafico Colori Normalizzati
        wubrg_order = ['White', 'Blue', 'Black', 'Red', 'Green', 'Multicolor', 'Colorless']
        plot_grouped_bar(full_df, 'clean_color', 'Distribuzione Colori (Normalizzata)', 'Colore', wubrg_order)
        
        # 2. Grafico Tipi Normalizzati
        plot_grouped_bar(full_df, 'clean_type', 'Distribuzione Tipi di Carta', 'Tipo Principale')
        
        # 3. Grafico Rarità Normalizzate
        # Ordine logico di rarità crescente
        rarity_order = ['Common', 'Uncommon', 'Rare', 'Mythic Rare', 'Basic Land']
        plot_grouped_bar(full_df, 'clean_rarity', 'Distribuzione Rarità', 'Rarità', rarity_order)
        
    else:
        print("Dati mancanti. Controlla i percorsi dei file.")