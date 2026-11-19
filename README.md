# Chess Opening ML

Machine Learning I group project for CSCI 3052U.

## Project Goal

This project investigates whether machine learning can be used to identify which chess openings perform best for players at different Elo rating ranges.

The project will use historical chess game data to train and evaluate supervised machine learning models for predicting game outcomes based on features such as:

- Player Elo ratings
- Elo difference
- Opening / ECO code
- Time control
- Game result
- Potentially derived play-style features

The long-term goal is to use these results to support personalized opening recommendations for chess players.

## Opening recommendation CLI

From the repository root, install the project dependencies and run:

```powershell
python -m pip install -r requirements.txt
python -m src.recommend_cli --elo 1500 --time-control rapid --style solid --color white --top-k 5
```

The CLI accepts `bullet`, `blitz`, `rapid`, and `classical` time-control aliases,
or a Lichess control such as `300+5`. Use `--input` to select a cleaned CSV or
Parquet dataset and `--min-games` to change the minimum training-game count for
an opening candidate. Run `python -m src.recommend_cli --help` for all options.
Recommendations are ranked from the model's training split; the CLI does not
evaluate the reserved test split.

## Team

- Luke McMann
- Shawn Heer
- Nesta Lacroix
- Brantley Cooper
- Cooper MacDonald

## Status

Milestone 1 — Project proposal and planning.