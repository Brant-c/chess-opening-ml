# Data

The project uses the public-domain (CC0) August 2016 Lichess standard-rated
games. The Milestone 2 notebook's processed sample is expected at
`data/processed/lichess_clean_2016_08.parquet`; it contains the cleaned first
100,000 parsed games (99,397 retained after filtering).

Milestone 3 uses `Result`, `AverageElo`, `EloDiff`, `ECO`, and `TimeControl`.
Raw moves and termination are excluded from model features because they are
not available before a game is played. Do not include raw PGN downloads or
generated environment files in the submission.