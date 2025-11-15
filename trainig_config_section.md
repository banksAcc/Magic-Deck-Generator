# Sezione `training` del config — MAGIC DECK GENERATOR

Questa guida definisce **la struttura e il significato** della sezione `training` di `configs/config.yaml`
per il progetto MTG × My Hero Academia.

Obiettivo principale: poter controllare **quanto dataset usare** e gli **iperparametri di training**
modificando *solo* il file YAML, senza toccare il codice di `a04_train_mistral_lora.py` (e affini).

---

## 1. Posizione nel file `config.yaml`

Esempio di struttura complessiva (solo le parti rilevanti):

```yaml
data:
  raw_path: data/raw/scryfall.json
  processed_dir: data/processed
  language: "en"
  seq_len: 512
  layouts_allowed: ["normal"]
  exclude_complex_layouts: true
  special_tokens:
    - "<|startofcard|>"
    - "<|gen_card|>"
    - "<|endofcard|>"

  split:
    train_ratio: 0.8
    val_ratio: 0.1
    dedup_exact: true
    shuffle_before_split: true

constants:
  colors: ["W", "U", "B", "R", "G"]
  rarities: ["common", "uncommon", "rare", "mythic"]
  type_macros: ["Creature", "Instant", "Sorcery", "Enchantment", "Artifact", "Planeswalker", "Land"]
  regex:
    mana_pattern: "^({([WUBRG]|[0-9X]+|[WUBRG]/[WUBRG])})+$"
    pt_pattern: "^[0-9X]+/[0-9X]+$"

validator:
  color_mana_subset_rule: false
  require_character: false

wandb:
  enabled: true
  project: "mtg-mha"
  entity: "your_wandb_entity"
  tags: ["mtg", "mha", "lora"]

training:
  # --- modello & run ---
  base_model_name: "mistralai/Mistral-7B-v0.1"
  run_name_prefix: "mha-lora"

  # --- dimensione effettiva del dataset usato ---
  max_train_examples: null     # es. 5000 per usare solo 5k carte; null -> usa tutto il train
  max_val_examples: null       # es. 1000; null -> usa tutto il val
  train_subset_fraction: null  # es. 0.1 per usare il 10% del train (alternativo a max_*)
  val_subset_fraction: null    # es. 0.1

  # --- iperparametri core ---
  batch_size: 4
  gradient_accumulation_steps: 8
  num_epochs: 3
  learning_rate: 2e-4
  weight_decay: 0.01

  # --- scheduling & logging ---
  warmup_steps: 500            # oppure usa warmup_ratio al posto di warmup_steps
  warmup_ratio: null           # es. 0.03 -> 3% degli step totali
  max_steps: null              # es. 2000 per run limitati; null -> usa num_epochs
  eval_every_n_steps: 500
  save_every_n_steps: 1000

  # --- seed (opzionale, per ora non usato) ---
  seed: null                   # deciderai più avanti se fissarlo o continuare a tenerlo libero
```

---

## 2. Logica per usare solo una parte del dataset

### 2.1. Regole di interpretazione

Nel codice di training, l’uso effettivo del dataset segue questa logica:

1. **Se `max_train_examples` è valorizzato (non `null`):**  
   usa al massimo quel numero di esempi dal train.

2. **Altrimenti, se `train_subset_fraction` è valorizzato:**  
   usa quella frazione del train (es. `0.1` → 10%).

3. **Altrimenti:**  
   usa **tutto** il train.

La stessa logica vale per la validation (`max_val_examples`, `val_subset_fraction`).

Pseudocodice tipico in `a04_train_mistral_lora.py`:

```python
train_blocks = load_blocks("data/processed/train.txt")

max_train = cfg["training"].get("max_train_examples")
train_frac = cfg["training"].get("train_subset_fraction")

if max_train is not None:
  train_blocks = train_blocks[:max_train]
elif train_frac is not None:
  n = int(len(train_blocks) * float(train_frac))
  train_blocks = train_blocks[:max(1, n)]

# analogamente per val_blocks
```

### 2.2. Esempi pratici

- **Run full dataset:**

  ```yaml
  training:
    max_train_examples: null
    max_val_examples: null
    train_subset_fraction: null
    val_subset_fraction: null
  ```

- **Run “veloce” per debugging (es. 500 carte train, 100 val):**

  ```yaml
  training:
    max_train_examples: 500
    max_val_examples: 100
    train_subset_fraction: null
    val_subset_fraction: null
  ```

- **Run su 10% del dataset (qualunque sia la dimensione assoluta):**

  ```yaml
  training:
    max_train_examples: null
    max_val_examples: null
    train_subset_fraction: 0.1
    val_subset_fraction: 0.1
  ```

---

## 3. Descrizione dettagliata dei campi `training`

| Campo                         | Tipo       | Default | Obbligatorio | Descrizione |
|------------------------------|-----------|---------|--------------|-------------|
| `base_model_name`            | string    | —       | sì           | Nome del modello base da usare per il fine-tuning (es. `mistralai/Mistral-7B-v0.1`). |
| `run_name_prefix`            | string    | `"run"` | no           | Prefisso per il nome del run (usato per logging / W&B). |
| `max_train_examples`         | int/null  | null    | no           | Se impostato, limita il numero di esempi train caricati. Ha priorità su `train_subset_fraction`. |
| `max_val_examples`           | int/null  | null    | no           | Come sopra, ma per il validation set. |
| `train_subset_fraction`      | float/null| null    | no           | Frazione del train da usare (0–1). Ignorata se `max_train_examples` è valorizzato. |
| `val_subset_fraction`        | float/null| null    | no           | Frazione del val da usare (0–1). Ignorata se `max_val_examples` è valorizzato. |
| `batch_size`                 | int       | 4       | sì           | Numero di esempi per batch (per device, se multi-GPU). |
| `gradient_accumulation_steps`| int       | 1       | sì           | Passi di accumulo gradiente (per simulare batch più grandi). |
| `num_epochs`                 | int       | 3       | sì           | Numero di passate sull’intero dataset (se `max_steps` è null). |
| `learning_rate`              | float     | 2e-4    | sì           | Learning rate iniziale. |
| `weight_decay`               | float     | 0.0–0.1 | no           | Coefficiente di weight decay (regolarizzazione L2). |
| `warmup_steps`               | int/null  | null    | no           | Numero di step di warmup (sovrascrive `warmup_ratio` se entrambi non null). |
| `warmup_ratio`               | float/null| null    | no           | Frazione di step totali da usare per warmup (es. 0.03 → 3%). Usata solo se `warmup_steps` è null. |
| `max_steps`                  | int/null  | null    | no           | Limite superiore di step di training. Se non null, prevale su `num_epochs`. |
| `eval_every_n_steps`         | int       | 500     | no           | Frequenza (in step) delle valutazioni su validation set. |
| `save_every_n_steps`         | int       | 1000    | no           | Frequenza (in step) dei salvataggi di checkpoint. |
| `seed`                       | int/null  | null    | no           | Seed globale per riproducibilità. Per ora può rimanere null (coerente con la decisione di non fissare i seed). |

---

## 4. Come si integra con il resto della pipeline

1. **a01_preprocessor**  
   - Non legge la sezione `training`.  
   - Genera `all.txt` e statistiche di base.

2. **a02_split**  
   - Usa `data.split.*` (train_ratio, val_ratio, dedup, shuffle).  
   - Produce `train.txt`, `val.txt`, `test.txt` e `split_stats.json`.

3. **a03_validate**  
   - Usa `constants.*` e `validator.*` per controlli “duri”.  
   - Produce `*_validated.jsonl` + `validate_summary.json`.

4. **a04_train_* (Mistral LoRA, GPT-2, ecc.)**  
   - Legge i file di split (tipicamente `train.txt`, `val.txt`).  
   - Applica i limiti di dimensione / frazione definiti in `training.*`.  
   - Usa `wandb.*` per il logging e `training.*` per gli iperparametri.

In questo modo:

- la **definizione dello split** rimane separata dalla **scelta di quanto dataset usare**;
- puoi passare da run “toy” a run “full” cambiando **una sola sezione** del config;
- non devi rigenerare i file di split quando vuoi solo ridurre il numero di esempi usati.

---

## 5. Pattern consigliati per il tuo progetto

- Durante lo sviluppo iniziale:  

  ```yaml
  training:
    max_train_examples: 2000
    max_val_examples: 500
    train_subset_fraction: null
    val_subset_fraction: null
  ```

- Quando la pipeline è stabile e vuoi un run serio:  

  ```yaml
  training:
    max_train_examples: null
    max_val_examples: null
    train_subset_fraction: null
    val_subset_fraction: null
  ```

- Per esperimenti graduali (es. scaling law “artigianale”):  
  fai girare più run cambiando solo `max_train_examples` (es. 2k, 5k, 10k, 20k) e guardi come si muovono loss e PPL.
