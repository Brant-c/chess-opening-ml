"""Recommend chess openings from the project's cleaned game dataset."""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Any, Callable, Sequence

import pandas as pd

from src import baseline

DEFAULT_INPUT = Path("data/processed/lichess_clean_2016_08.parquet")
TIME_CONTROL_ALIASES = {
    "bullet": "60+0",
    "blitz": "180+0",
    "rapid": "600+0",
    "classical": "900+15",
}
STYLE_CHOICES = ("balanced", "solid", "aggressive")
COLOR_CHOICES = ("white", "black")


def parse_elo(value: str) -> float:
    """Parse a positive, finite Elo rating for argparse."""
    try:
        elo = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("Elo must be a number.") from error
    if not math.isfinite(elo) or elo <= 0:
        raise argparse.ArgumentTypeError("Elo must be a positive, finite number.")
    return elo


def parse_time_control(value: str) -> str:
    """Accept common time-class names or a Lichess base+increment value."""
    normalized = value.strip().lower()
    if normalized in TIME_CONTROL_ALIASES:
        return TIME_CONTROL_ALIASES[normalized]
    if re.fullmatch(r"\d+\+\d+", normalized):
        return normalized
    choices = ", ".join((*TIME_CONTROL_ALIASES, "or a value like 600+0"))
    raise argparse.ArgumentTypeError(f"Choose a time control from {choices}.")


def positive_int(value: str) -> int:
    """Parse a positive integer for argparse."""
    try:
        number = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("Value must be a positive integer.") from error
    if number <= 0:
        raise argparse.ArgumentTypeError("Value must be a positive integer.")
    return number


def build_candidate_openings(
    frame: pd.DataFrame,
    train_index: pd.Index,
    min_games: int = 40,
) -> pd.DataFrame:
    """Build training-only candidates with a common move sequence per opening."""
    if min_games <= 0:
        raise ValueError("min_games must be a positive integer.")
    required = {"ECO", "Opening", "Moves"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(missing)}")

    training = frame.loc[train_index, ["ECO", "Opening", "Moves"]].copy()
    training = training.dropna(subset=["ECO", "Opening"])
    moves = training["Moves"].fillna("").astype(str)
    training = training[moves.str.split().str.len() > 20].copy()
    moves = training["Moves"].fillna("").astype(str).str.split()
    for index, column in enumerate(baseline.MOVE_FEATURES):
        training[column] = moves.str[index].fillna("NONE")

    opening_columns = ["ECO", "Opening"]
    sequence_columns = [*opening_columns, *baseline.MOVE_FEATURES]
    sequence_counts = (
        training.groupby(sequence_columns, dropna=False)
        .size()
        .reset_index(name="sequence_count")
    )
    opening_counts = (
        sequence_counts.groupby(opening_columns, dropna=False)["sequence_count"]
        .sum()
        .reset_index(name="train_count")
    )
    representatives = (
        sequence_counts.sort_values(
            "sequence_count", ascending=False, kind="stable"
        )
        .drop_duplicates(opening_columns)
        .drop(columns="sequence_count")
    )
    candidates = representatives.merge(opening_counts, on=opening_columns)
    return (
        candidates[candidates["train_count"] >= min_games]
        .reset_index(drop=True)
    )


def score_candidates(
    candidates: pd.DataFrame,
    model: Any,
    user_elo: float,
    time_control: str,
    color: str,
    style: str,
) -> pd.DataFrame:
    """Score and rank candidate openings with a fitted outcome model."""
    if candidates.empty:
        raise ValueError(
            "No opening candidates meet the minimum game count. "
            "Try a smaller --min-games value."
        )
    if color not in COLOR_CHOICES:
        raise ValueError(f"color must be one of: {', '.join(COLOR_CHOICES)}")
    if style not in STYLE_CHOICES:
        raise ValueError(f"style must be one of: {', '.join(STYLE_CHOICES)}")
    if not math.isfinite(user_elo) or user_elo <= 0:
        raise ValueError("user_elo must be a positive, finite number.")

    query = pd.DataFrame(index=candidates.index)
    query["AverageElo"] = float(user_elo)
    query["EloDiff"] = 0.0
    query["TimeControl"] = time_control
    for column in baseline.MOVE_FEATURES:
        query[column] = candidates[column]
    query["AbsEloDiff"] = 0.0
    query["WhiteElo"] = float(user_elo)
    query["BlackElo"] = float(user_elo)

    feature_columns = [
        *baseline.FEATURES,
        *baseline.MOVE_FEATURES,
        "AbsEloDiff",
        "WhiteElo",
        "BlackElo",
    ]
    probabilities = baseline._probabilities_in_label_order(
        model, query.loc[:, feature_columns]
    )
    class_indices = {label: index for index, label in enumerate(baseline.CLASS_LABELS)}
    p_black = probabilities[:, class_indices["0-1"]]
    p_white = probabilities[:, class_indices["1-0"]]
    p_draw = probabilities[:, class_indices["1/2-1/2"]]
    win_probability = p_white if color == "white" else p_black
    expected_score = win_probability + 0.5 * p_draw

    if style == "solid":
        rank_score = expected_score + 0.15 * p_draw
    elif style == "aggressive":
        rank_score = win_probability + 0.10 * (1.0 - p_draw)
    else:
        rank_score = expected_score

    display_columns = [
        "ECO",
        "Opening",
        *baseline.MOVE_FEATURES[:4],
        "train_count",
    ]
    scored = candidates.loc[:, display_columns].copy()
    scored["p_win"] = win_probability
    scored["p_draw"] = p_draw
    scored["expected_score"] = expected_score
    scored["rank_score"] = rank_score
    return scored.sort_values(
        "rank_score", ascending=False, kind="stable"
    ).reset_index(drop=True)


def fit_recommendation_model(
    frame: pd.DataFrame,
    min_games: int = 40,
    model_factory: Callable[[], Any] | None = None,
) -> tuple[pd.DataFrame, Any]:
    """Split the dataset, build training-only candidates, and fit the model."""
    features, target = baseline.prepare_dataset(frame)
    X_train, _, _, y_train, _, _ = baseline.split_dataset(features, target)
    candidates = build_candidate_openings(frame, X_train.index, min_games=min_games)
    if candidates.empty:
        raise ValueError(
            "No opening candidates meet the minimum game count. "
            "Try a smaller --min-games value."
        )

    if model_factory is None:
        model_factory = lambda: baseline.build_models()["LightGBM (Main)"]
    model = model_factory()
    model.fit(X_train, y_train)
    return candidates, model


def recommend_openings(
    frame: pd.DataFrame,
    user_elo: float,
    time_control: str,
    color: str = "white",
    style: str = "balanced",
    min_games: int = 40,
    model_factory: Callable[[], Any] | None = None,
) -> pd.DataFrame:
    """Fit the recommendation model and return ranked opening candidates."""
    candidates, model = fit_recommendation_model(
        frame, min_games=min_games, model_factory=model_factory
    )
    return score_candidates(
        candidates,
        model,
        user_elo=user_elo,
        time_control=time_control,
        color=color,
        style=style,
    )


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser independently for easy testing."""
    parser = argparse.ArgumentParser(
        description="Rank chess openings for a player's rating and preferences."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Cleaned M2 Parquet or CSV dataset.",
    )
    parser.add_argument("--elo", type=parse_elo, required=True, help="Your Elo rating.")
    parser.add_argument(
        "--time-control",
        type=parse_time_control,
        default="rapid",
        metavar="CLASS|BASE+INCREMENT",
        help="bullet, blitz, rapid, classical, or a control like 600+0 (default: rapid).",
    )
    parser.add_argument(
        "--style",
        type=str.lower,
        choices=STYLE_CHOICES,
        default="balanced",
        help="Ranking preference: balanced, solid, or aggressive.",
    )
    parser.add_argument(
        "--color",
        type=str.lower,
        choices=COLOR_CHOICES,
        default="white",
        help="Recommend for white or black.",
    )
    parser.add_argument(
        "--min-games",
        type=positive_int,
        default=40,
        help="Minimum training games for an opening candidate (default: 40).",
    )
    parser.add_argument(
        "--top-k",
        type=positive_int,
        default=5,
        help="Number of recommendations to print (default: 5).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Run the recommendation CLI; argv can be supplied by tests or callers."""
    args = build_parser().parse_args(argv)
    frame = baseline.load_dataset(args.input)
    ranked = recommend_openings(
        frame,
        user_elo=args.elo,
        time_control=args.time_control,
        color=args.color,
        style=args.style,
        min_games=args.min_games,
    )
    columns = [
        "ECO",
        "Opening",
        *baseline.MOVE_FEATURES[:4],
        "p_win",
        "p_draw",
        "expected_score",
        "train_count",
    ]
    output = ranked.loc[:, columns].head(args.top_k).copy()
    output["p_win"] *= 100
    output["p_draw"] *= 100
    output["expected_score"] *= 100
    output = output.rename(
        columns={
            "p_win": "win%",
            "p_draw": "draw%",
            "expected_score": "expected%",
        }
    )
    print(
        f"\nTop {len(output)} openings for {args.color.upper()} | "
        f"Elo={args.elo:.0f} | Time control={args.time_control} | "
        f"Style={args.style}\n"
    )
    print(output.to_string(index=False, float_format=lambda value: f"{value:.1f}"))


if __name__ == "__main__":
    main()
