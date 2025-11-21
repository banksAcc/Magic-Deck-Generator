# Magic-Deck-Generator
This repo contains a fine-tuned AI card generator for the Magic: The Gathering Set.  Several balanced card decks are available, suitable for printing and play.


mtg-mha/
├─ README.md
├─ requirements.txt
├─ .gitignore
│
├─ data/
│  ├─ raw/                  # input (bulk Scryfall)
│  │  └─ .gitkeep
│  └─ processed/            # output preprocess/split
│     └─ .gitkeep
│
├─ src/
│  ├─ preprocess.py
│  ├─ split.py
│  ├─ train_gpt2.py
│  ├─ train_mistral_lora.py
│  ├─ eval_ppl.py
│  ├─ generate.py
│  ├─ validate.py
│  ├─ curate.py
│  ├─ utils.py
│  └─ mapping/
│     ├─ mapping_seed.csv
│     └─ keywords.json
│
├─ configs/
│  └─ config.yaml #general file of config with all important configuration info 
│
├─ outputs/
│  ├─ checkpoints/
│  │  └─ .gitkeep
│  ├─ generations/
│  │  └─ .gitkeep
│  └─ final_set/
│     └─ .gitkeep
│
└─  scripts/
    ├─ run_train_gpt2.sh
    ├─ run_train_mistral.sh
    ├─ run_generate.sh
    └─ run_validate_curate.sh



python -m src.a01_preprocess --> fa il pre-process dei dati, elimina carte errate/non utili e crea un file unico detto all.txt con tutto il dataset (in base ai flag di attivazione nel config fa più o meno cose sui dati)

python -m src.a02_split --> splitta i dati iniziali (del dataset precedente) e li inserisce in 3 file seprarati. Fa anche una eliminazione di record duplicati, se attiva nel file di configurazione

python -m src.a03_validate --> valida che i file creati per i 3 task del modello siano coerenti alla struttura stabilita e decisa per il sistema. Va a creare i file .jsonl per ogniuno dei 3 split creati


python -m src.a04_train_gpt2 --> fai il training del modello selezionato (quindi vedere in config, nella sezione modelli sotto gpt2), con il set presente in training, test validated.

python -m src.a06_generate_and_validate --> misura quanto “bene” il modello ha imparato il nostro formato di carte, usando la metrica di Perplexity (PPL)