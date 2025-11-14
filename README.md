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
│  ├─ data.yaml
│  ├─ gpt2.yaml
│  ├─ mistral.yaml
│  ├─ gen.yaml
│  ├─ curate.yaml
│  └─ wandb.yaml
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

