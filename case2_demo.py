
from __future__ import annotations

import argparse
from math import sqrt
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

DEFAULT_DATA_PATH = Path("researchfinal.csv")
DEFAULT_MODEL_PATH = Path("lstm_4_layer_96_15.h5")
N_INPUT = 12  # matches the notebook window length


def load_monthly_max(path: Path) -> pd.DataFrame:
    """Load the CSV, clean it, and aggregate to monthly mean max temps."""
    data = pd.read_csv(path)
    data = data.drop(columns=["AW", "RF", "SSH", "INDEX"], errors="ignore")
    data = data.dropna().copy()
    data["Date"] = pd.to_datetime(data["Date"], format="%d-%m-%Y")
    data = data.set_index("Date")
    monthly = data.resample("M").mean().ffill().bfill()
    monthly = monthly.drop(columns=[col for col in monthly.columns if col != "MAX"])
    return monthly


def fit_scaler(monthly: pd.DataFrame) -> MinMaxScaler:
    history = monthly.loc["1996-01-01":"2017-01-01"].copy()
    train = history.iloc[:-12]
    scaler = MinMaxScaler()
    scaler.fit(train)
    return scaler


def run_case2_demo(
    monthly: pd.DataFrame,
    scaler: MinMaxScaler,
    model_path: Path,
    base_year: int,
    forecast_years: int,
) -> Tuple[pd.DataFrame, List[Dict[str, float]]]:
    model = load_model(model_path, compile=False)

    seed_start = f"{base_year}-01-01"
    seed_end = f"{base_year + 1}-01-01"
    target_start = f"{base_year + 1}-01-01"
    target_end = f"{base_year + forecast_years + 1}-01-01"

    seed = monthly.loc[seed_start:seed_end].copy()
    target = monthly.loc[target_start:target_end].copy()

    if len(seed) < N_INPUT or len(target) < forecast_years * 12:
        raise ValueError("Not enough data to perform the requested demo.")

    current_batch = scaler.transform(seed)[-N_INPUT:].reshape(1, N_INPUT, 1)
    predictions = []
    metrics: List[Dict[str, float]] = []
    eval_year = base_year + 1

    for window_start in range(0, forecast_years * 12, 12):
        scaled_preds = []
        for _ in range(12):
            next_scaled = model.predict(current_batch, verbose=0)[0]
            scaled_preds.append(next_scaled)
            current_batch = np.append(
                current_batch[:, 1:, :],
                [[next_scaled]],
                axis=1,
            )

        year_preds = scaler.inverse_transform(np.array(scaled_preds).reshape(-1, 1)).ravel()
        predictions.extend(year_preds)

        actual_slice = target.iloc[window_start:window_start + 12].copy()
        actual_slice["Predictions"] = year_preds

        rmse = sqrt(mean_squared_error(actual_slice["MAX"], actual_slice["Predictions"]))
        r2 = r2_score(actual_slice["MAX"], actual_slice["Predictions"])
        metrics.append({"year": eval_year, "r2": r2, "rmse": rmse})
        eval_year += 1

    target = target.assign(Predictions=predictions[: len(target)])
    return target, metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Case 2 LSTM demo.")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="Path to researchfinal.csv",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Path to the trained LSTM weights (H5 file).",
    )
    parser.add_argument(
        "--base-year",
        type=int,
        default=2016,
        help="Year used as the seed window for predictions.",
    )
    parser.add_argument(
        "--forecast-years",
        type=int,
        default=5,
        help="Number of years (sets of 12 months) to predict.",
    )
    parser.add_argument(
        "--plot-path",
        type=Path,
        help="If provided, save an actual vs predicted plot to this file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    monthly = load_monthly_max(args.data_path)
    scaler = fit_scaler(monthly)
    forecast_df, metrics = run_case2_demo(
        monthly=monthly,
        scaler=scaler,
        model_path=args.model_path,
        base_year=args.base_year,
        forecast_years=args.forecast_years,
    )

    print("Case 2 demo metrics:")
    for item in metrics:
        print(f"Year {item['year']}: R2={item['r2']:.3f}, RMSE={item['rmse']:.3f}")

    print("\nPreview of predictions:")
    print(forecast_df.head(15))

    if args.plot_path:
        plt.figure(figsize=(14, 5))
        plt.plot(forecast_df.index, forecast_df["MAX"], label="Actual MAX", linewidth=1.5)
        plt.plot(
            forecast_df.index,
            forecast_df["Predictions"],
            label="Predicted MAX",
            linewidth=1.5,
        )
        plt.title(f"Case 2 Forecast (seed {args.base_year})")
        plt.xlabel("Date")
        plt.ylabel("Temperature (°C)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(args.plot_path, dpi=300)
        plt.close()
        print(f"\nSaved plot to {args.plot_path}")


if __name__ == "__main__":
    main()

