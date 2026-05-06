# src/sigbandit/tests/temp_sensor_exp/load_data.py
import numpy as np
import pandas as pd

def load_intel_temp_matrix(txt_path: str, freq="5min"):
    """
    Returns:
      temp: (T, K) float array, K=54, indexed by uniform 5-min bins
      times: pandas.DatetimeIndex length T
      sensor_ids: sorted unique mote ids used as columns
    """
    cols = ["date", "time", "epoch", "moteid", "temperature", "humidity", "light", "voltage"]
    df = pd.read_csv(txt_path, sep=r"\s+", header=None, names=cols)

    # Combine date+time into timestamp
    df["ts"] = pd.to_datetime(df["date"] + " " + df["time"], errors="coerce")
    df = df.dropna(subset=["ts", "moteid", "temperature"])
    df["moteid"] = df["moteid"].astype(int)

    # restrict time frame
    start = pd.Timestamp("2004-02-28 00:00:00")
    end   = pd.Timestamp("2004-03-08 00:00:59")
    df = df[(df["ts"] >= start) & (df["ts"] <= end)].copy()

    # resample to 5-min bins per sensor (mean)
    df = df.set_index("ts")
    temp_bin = (
      df.groupby("moteid")["temperature"]
        .resample(freq)
        .mean()
        .reset_index()
    )

    # remove sensors with little data
    min_coverage = 0.7
    pivot_raw = temp_bin.pivot(index="ts", columns="moteid", values="temperature").sort_index()
    coverage = pivot_raw.notna().mean(axis=0)
    sensor_ids = coverage[coverage >= min_coverage].index
    pivot = pivot_raw[sensor_ids]

    # Fill missing values: time interpolation then ffill/bfill
    pivot = pivot.interpolate(limit_direction="both")
    pivot = pivot.ffill().bfill()


    # add time-of-day features as extra columns
    idx = pivot.index
    hour = idx.hour + idx.minute / 60.0
    ang = 2 * np.pi * hour / 24.0
    pivot["tod_sin"] = np.sin(ang)
    pivot["tod_cos"] = np.cos(ang)

    temp = pivot.to_numpy(dtype=float)
    times = pivot.index
    col_ids = list(pivot.columns)

    return temp, times, col_ids


# test
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--txt", type=str, default="tests/temp_sensor_exp/temp_sensor.txt")
    parser.add_argument("--freq", type=str, default="5min")
    parser.add_argument("--n", type=int, default=13)
    args = parser.parse_args()

    temp, times, sensor_ids = load_intel_temp_matrix(args.txt, freq=args.freq)

    print(f"Loaded matrix: temp shape={temp.shape} (T,K), K={len(sensor_ids)}")
    print(f"Time range: {times[0]} -> {times[-1]}")

    # columns to show: anchors 28,32,0 plus tod_sin/tod_cos (last two)
    idxs = [28, 32, 0, -2, -1]  # -2 = tod_sin, -1 = tod_cos

    # make column names safely (sensor_ids is a list)
    colnames = [sensor_ids[i] for i in idxs]
    head_df = pd.DataFrame(temp[:args.n, idxs],
                           index=times[:args.n],
                           columns=colnames)
    print("\nHEAD (first rows, first cols):")
    print(head_df)
