# Regicide

The card game Regicide playable via a CLI interface.

The games rules can be found here: https://www.regicidegame.com/site_files/33132/upload_files/RegicideRulesA4.pdf


## Overview

Regicide is a cooperative card game where players work together to defeat the royal enemies (Jacks, Queens, and Kings) using number cards, animals (Aces), and jesters.

## Installation

```bash
# Clone the repository
git clone https://github.com/fbec76/regicide.git
cd regicide

python -m venv .venv
# unix  
source .venv/bin/activate 
# OR windows
.venv/Scripts/activate

# Install dependencies
pip install -r requirements.txt
```

## Running the Game

```bash
python src/main.py
```