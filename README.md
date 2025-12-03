# Magic-Deck-Generator

Pipeline sperimentale per generare carte in stile *Magic: The Gathering* condizionate sul tema *My Hero Academia*. Tutti i passaggi
sono guidati dal file di configurazione `configs/config.yaml` (token speciali, percorsi dati, regex, hyperparameter, W&B, ecc.) e
si eseguono come moduli Python dal root del repository.

## Dipendenze
Installa i requisiti minimi (PyYAML, Hugging Face `transformers`/`torch`, W&B opzionale):

```bash
pip install -r requirements.txt
```

## Dati attesi
- **Sorgente**: bulk JSON di Scryfall in `data/raw/scryfall.json`.
- **Token speciali**: definiti in `configs/config.yaml` (`<|startofcard|>`, `<|gen_card|>`, `<|endofcard|>`).
- **Output preprocess**: `data/processed/all.txt` con un blocco strutturato per carta più `preprocess_stats.json`.

## Flusso passo-passo (file per file)
### `src/a01_preprocess.py`
*Input*: `data/raw/scryfall.json` (layout "normal", lingua EN).  
*Cosa fa*: normalizza testo e type line, filtra carte prive di campi essenziali, applica il mapper tematico (config `mapping.*`),
costruisce blocchi nel formato standard e calcola statistiche di filtraggio.  
*Output*: `data/processed/all.txt` + `data/processed/preprocess_stats.json`.  
*Esegui*:
```bash
python -m src.a01_preprocess --config configs/config.yaml
```

### `src/a02_split.py`
*Input*: `data/processed/all.txt`.  
*Config chiave*: `data.split.train_ratio`, `data.split.val_ratio`, `data.split.dedup_exact`, `data.split.shuffle_before_split`.  
*Cosa fa*: separa i blocchi in train/val/test (split semplice, dedup 1:1 opzionale), calcola distribuzioni di rarity/type/color e
salva statistiche.  
*Output*: `data/processed/train.txt`, `val.txt`, `test.txt`, `split_stats.json`.  
*Esegui*:
```bash
python -m src.a02_split --config configs/config.yaml
```

### `src/a03_validate.py`
*Input*: uno o più file `.txt` (per default usa train/val/test).  
*Config chiave*: sezione `validator` (ordine rigido dei campi, regex mana/pt, coerenza color↔mana, obbligatorietà di `character`).  
*Mechanica*: parser "duro" che verifica ordine, presenza singola dei campi, macro-tipi validi, `pt` solo per Creature, EOS
obbligatorio.  
*Output*: `*_validated.jsonl` con `{valid, errors, raw, fields}` per ogni blocco.  
*Esegui*:
```bash
python -m src.a03_validate --config configs/config.yaml \
  --input data/processed/train.txt data/processed/val.txt data/processed/test.txt
```

### `src/a04_train_gpt2.py`
*Stato*: implementato per la baseline GPT-2 (full FT o LoRA).  
*Input*: `data/processed/train_validated.jsonl`, `val_validated.jsonl` (solo record `valid=True`).  
*Config chiave*: `training.gpt2.*` (model_name, lr, epochs, batch, gradient accumulation) e `training.subset_fraction`.  
*Output*: checkpoint Hugging Face in `outputs/checkpoints/<run_name>/` e log di training.  
*Esegui* (dopo validazione):
```bash
python -m src.a04_train_gpt2 --config configs/config.yaml
```

### `src/a04_train_mistral_lora.py`
*Stato*: file placeholder per futuro training Mistral 7B QLoRA (non ancora implementato).

### `src/a05_eval_ppl.py`
*Input*: modello CausalLM (es. checkpoint GPT-2) e uno split validato (`*_validated.jsonl`).  
*Config chiave*: sezione `ppl_eval` (`run_name`, split da valutare, `subset_fraction`, `batch_size`).  
*Output*: metrica `eval_loss` e `perplexity` salvate in `outputs/<run_name>/eval_<split>/ppl_<split>.json`.  
*Esegui*:
```bash
python -m src.a05_eval_ppl --config configs/config.yaml
```

### `src/a06_generate_and_validate.py`
*Input*: modello fine-tunato, mapping seed/keywords (config `mapping.*`) e config di decoding `generation.*`.  
*Flow*: costruisce prompt condizionati (fino a `<|gen_card|>`), genera batch di carte, tronca al primo `<|endofcard|>`, valida i
blocchi con la stessa logica di `a03_validate`.  
*Output*: `outputs/generations/batch_<timestamp>.txt` e `batch_<timestamp>_validated.jsonl` con errori/pass-rate.  
*Esegui*:
```bash
python -m src.a06_generate_and_validate --config configs/config.yaml
```

### `src/a07_curate.py`
*Stato*: scheletro per ranking/bilanciamento finale (non implementato).  
*Obiettivo previsto*: combinare score sintattici/diversità con target di distribuzione (`curation.*` nel config) per ottenere il
set finale da ~100 carte.

## Configurazione e logging
- **Configurazione unica**: `configs/config.yaml` centralizza percorsi, token, regex, mapping, parametri di training/generazione.
  Evitare costanti duplicate nel codice.
- **Weights & Biases**: se `wandb.enabled` è `true` e l’SDK è installato, gli script loggano run basilari (`pre.*`, `split.*`,
  metriche di training). Imposta `WANDB_API_KEY` o effettua `wandb login` prima di eseguire.

## Suggerimenti rapidi
1. Esegui sempre dal root della repo (`Magic-Deck-Generator/`).
2. Mantieni il formato dei blocchi invariato: ordine dei campi e token speciali sono contratti rigidi per l’intera pipeline.
3. Dopo ogni step controlla i file di statistiche (`preprocess_stats.json`, `split_stats.json`, report di validazione) per assicurarti
   che i filtri e le distribuzioni siano sensati.


python -m src.a01_preprocess --> fa il pre-process dei dati, elimina carte errate/non utili e crea un file unico detto all.txt con tutto il dataset (in base ai flag di attivazione nel config fa più o meno cose sui dati)

python -m src.a02_split --> splitta i dati iniziali (del dataset precedente) e li inserisce in 3 file seprarati. Fa anche una eliminazione di record duplicati, se attiva nel file di configurazione

python -m src.a03_validate --> valida che i file creati per i 3 task del modello siano coerenti alla struttura stabilita e decisa per il sistema. Va a creare i file .jsonl per ogniuno dei 3 split creati

python -m src.a04_train_gpt2 --> fai il training del modello selezionato (quindi vedere in config, nella sezione modelli sotto gpt2), con il set presente in training, test validated.

python -m src.a04_train_mistral_lora

python -m src.a05_eval_ppl 

python -m src.a06_generate_and_validate --> misura quanto “bene” il modello ha imparato il nostro formato di carte, usando la metrica di Perplexity (PPL)


python chart/chaper_5_chart.py  --input data/processed/train.txt --output-dir outputs/figures_train --tokenizer gpt2

python chart/chaper_6_chart.py --ppl-compare logs/ppl_compare_example.csv --ppl-vs-valid logs/ppl_vs_valid_example.csv --output-dir outputs/figures_train

