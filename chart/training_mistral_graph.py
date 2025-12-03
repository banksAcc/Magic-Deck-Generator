import matplotlib.pyplot as plt
import numpy as np

# 1. Configurazione Dati (Simulazione)
np.random.seed(42) # Per riproducibilità
steps = np.arange(1, 701) # Da 1 a 700 step

# Creazione della curva di loss (decadimento esponenziale + base)
# Modifica questi parametri per cambiare la forma della curva
loss_base = 2.0 * (steps ** -0.4) + 0.6 

# Aggiunta del "rumore" realistico (come richiesto, leggero)
noise = np.random.normal(0, 0.005, size=len(steps)) 

# Smoothing leggero per renderlo meno "finto" ma mantenere le imperfezioni
loss_final = loss_base + noise

# 2. Configurazione Stile Grafico (Stile WandB/Tensorboard pulito)
plt.figure(figsize=(12, 7), dpi=100)
plt.rcParams['font.family'] = 'sans-serif'

# Plot della linea
# Colore blu tipico delle dashboard di training
plt.plot(steps, loss_final, color='#5c96d6', linewidth=1.5, label='700: 0.8136 test_kaggle-mistral-v0.3-qlora train/loss')

# 3. Personalizzazione Assi e Griglia
ax = plt.gca()

# Griglia leggera
ax.grid(True, which='major', axis='y', linestyle='-', alpha=0.2, color='#dddddd')
ax.grid(True, which='major', axis='x', linestyle='-', alpha=0.2, color='#dddddd')

# Rimozione bordi superflui (spines) per il look "pulito"
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('#dddddd')
ax.spines['bottom'].set_color('#dddddd')

# Titoli e Etichette
plt.title("train/loss", fontsize=12, pad=20, color='#333333', fontweight='bold')
plt.xlabel("train/global_step", fontsize=10, color='#666666', loc='right')
plt.ylabel("", fontsize=10) # Label Y spesso vuota in queste dashboard

# Tick (numeri sugli assi)
ax.tick_params(axis='both', colors='#666666')

# Legenda (Stile box in alto a destra)
legend = plt.legend(loc='upper right', frameon=True, fontsize=9)
frame = legend.get_frame()
frame.set_facecolor('white')
frame.set_edgecolor('#dddddd')
frame.set_boxstyle('round,pad=0.5')

# Limiti assi (simili alla tua immagine)
plt.xlim(0, 710)
plt.ylim(min(loss_final) - 0.1, max(loss_final) + 0.1)

# 4. Salvataggio in SVG
plt.tight_layout()
plt.savefig("training_loss_graph.svg", format='svg')
plt.show()

print("Grafico salvato come 'training_loss_graph.svg'")