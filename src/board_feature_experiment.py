"""Compare the current LightGBM model against engine-free board features.

Run from the repository root after installing the project's dependencies
and python-chess: python -m src.board_feature_experiment
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import chess
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, log_loss, recall_score

if __package__:
    from . import baseline
else:
    import baseline  # VS Code can launch this file directly from src/.


MOVE_COLUMNS = tuple(f"Move{i}" for i in range(1, 21))
BOARD_COLUMNS = (
    "MaterialDiff20",
    "WhiteCastled20",
    "BlackCastled20",
    "WhiteDevelopedMinorPieces20",
    "BlackDevelopedMinorPieces20",
)
PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
}
MINOR_START_SQUARES = {
    chess.B1, chess.C1, chess.F1, chess.G1,
    chess.B8, chess.C8, chess.F8, chess.G8,
}
DRAW = "1/2-1/2"
CLOSE_ELO_GAP = 100


def extract_board_features(moves: tuple[str, ...]) -> tuple[int, int, int, int, int]:
    """Describe only the board and moves through ply 20, from a standard start."""
    if len(moves) != len(MOVE_COLUMNS):
        raise ValueError("Expected exactly 20 UCI plies.")

    board = chess.Board()
    castled = {chess.WHITE: 0, chess.BLACK: 0}
    developed = {chess.WHITE: 0, chess.BLACK: 0}
    unmoved_minors = MINOR_START_SQUARES.copy()

    for uci in moves:
        move = board.parse_uci(uci)
        if not move:
            raise ValueError("Null moves are not allowed in a standard game.")
        color = board.turn
        if board.is_castling(move):
            castled[color] = 1
        # Count a starting knight or bishop the first time it moves, even if
        # it later returns to its starting square. A capture does not count.
        if move.from_square in unmoved_minors:
            developed[color] += 1
            unmoved_minors.remove(move.from_square)
        unmoved_minors.discard(move.to_square)
        board.push(move)

    material_diff = sum(
        value * (len(board.pieces(piece, chess.WHITE)) - len(board.pieces(piece, chess.BLACK)))
        for piece, value in PIECE_VALUES.items()
    )
    return (
        material_diff,
        castled[chess.WHITE], castled[chess.BLACK],
        developed[chess.WHITE], developed[chess.BLACK],
    )


def add_board_features(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    move_rows = frame.loc[:, MOVE_COLUMNS].itertuples(index=False, name=None)
    for game_index, moves in zip(frame.index, move_rows):
        try:
            rows.append(extract_board_features(moves))
        except ValueError as exc:
            raise ValueError(f"Cannot replay game at dataset row {game_index}: {exc}") from exc

    extras = pd.DataFrame(rows, index=frame.index, columns=BOARD_COLUMNS)
    return pd.concat([frame, extras], axis=1)


def evaluate(X_train, X_val, y_train, y_val, threads: int) -> pd.DataFrame:
    base_columns = list(X_train.columns.drop(list(BOARD_COLUMNS)))
    variants = [
        ("Current LightGBM", base_columns),
        ("Current + material", base_columns + [BOARD_COLUMNS[0]]),
        ("Current + all board features", base_columns + list(BOARD_COLUMNS)),
    ]
    labels = list(baseline.CLASS_LABELS)
    close_elo = X_val["EloDiff"].abs() <= CLOSE_ELO_GAP
    rows = []
    for name, columns in variants:
        print(f"Training {name}", flush=True)
        model = baseline.build_models()["LightGBM (Main)"]
        model.set_params(model__n_jobs=threads)
        started = perf_counter()
        model.fit(X_train.loc[:, columns], y_train)
        fit_seconds = perf_counter() - started

        for group, games in (("all", slice(None)), ("elo_gap_le_100", close_elo)):
            truth = y_val.loc[games]
            if truth.empty:
                continue
            features = X_val.loc[games, columns]
            predictions = model.predict(features)
            probabilities = baseline._probabilities_in_label_order(model, features)
            rows.append({
                "model": name,
                "validation_group": group,
                "n": len(truth),
                "accuracy": accuracy_score(truth, predictions),
                "macro_f1": f1_score(truth, predictions, labels=labels,
                                     average="macro", zero_division=0),
                "draw_recall": recall_score(truth, predictions, labels=[DRAW],
                                             average="macro", zero_division=0)
                if (truth == DRAW).any() else float("nan"),
                "log_loss": log_loss(truth, probabilities, labels=labels),
                "fit_seconds": fit_seconds,
            })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path,
                        default=Path("data/processed/lichess_clean_2016_08.parquet"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("reports/board_feature_experiment"))
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be at least 1.")
    if tuple(getattr(baseline, "MOVE_FEATURES", ())) != MOVE_COLUMNS or DRAW not in baseline.CLASS_LABELS:
        parser.error("This experiment needs the 20-ply, three-class src/baseline.py.")

    X, y = baseline.prepare_dataset(baseline.load_dataset(args.input))
    X_train, X_val, X_test, y_train, y_val, _ = baseline.split_dataset(X, y)
    started = perf_counter()
    print(f"Replaying the first 20 plies of {len(X_train):,} training games "
          f"and {len(X_val):,} validation games...", flush=True)
    X_train = add_board_features(X_train)
    X_val = add_board_features(X_val)
    feature_seconds = perf_counter() - started
    results = evaluate(X_train, X_val, y_train, y_val, args.threads)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.output_dir / "validation_comparison.csv"
    results.to_csv(results_path, index=False, float_format="%.6f")
    (args.output_dir / "run_metadata.json").write_text(
        json.dumps({
            "input": str(args.input),
            "seed": baseline.RANDOM_STATE,
            "training_rows": len(X_train),
            "validation_rows": len(X_val),
            "test_rows_reserved": len(X_test),
            "test_evaluated": False,
            "position_ply": len(MOVE_COLUMNS),
            "added_features": BOARD_COLUMNS,
            "feature_seconds": round(feature_seconds, 3),
            "chess_engine_used": False,
        }, indent=2) + "\n",
        encoding="utf-8",
    )
    print(results.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"Saved validation results to {results_path}")


if __name__ == "__main__":
    main()
