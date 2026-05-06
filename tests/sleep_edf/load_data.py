# src/sigbandit/tests/sleep_edf/load_data.py
from __future__ import annotations

import os
import glob
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd
import mne
from tqdm import tqdm
import warnings


_STAGE_MAP = {
    "Sleep stage W": "W",
    "Sleep stage 1": "N1",
    "Sleep stage 2": "N2",
    "Sleep stage 3": "N3",
    "Sleep stage 4": "N3",
    "Sleep stage R": "REM",
    "Sleep stage ?": "UNK",
    "Movement time": "MT",
}

STAGE_ORDER = ["W", "N1", "N2", "N3", "REM", "MT", "UNK"]


@dataclass
class SleepBanditRecord:
    record_id: str
    fs: float
    X: np.ndarray
    epoch_len: int
    channels: List[str]
    P: np.ndarray              # (n_epochs, n_stages) overlap fractions
    stage_order: List[str]     # = STAGE_ORDER


def _pair_sleep_edf_files(cassette_dir: str) -> List[Dict[str, str]]:
    cassette_dir = os.path.abspath(cassette_dir)
    psgs = sorted(glob.glob(os.path.join(cassette_dir, "*-PSG.edf")))
    pairs: List[Dict[str, str]] = []

    for psg in psgs:
        stem = os.path.basename(psg).replace("-PSG.edf", "")  # e.g. SC4011E0
        prefix = stem[:-1]  # e.g. SC4011E
        hyp_matches = sorted(glob.glob(os.path.join(cassette_dir, f"{prefix}*-Hypnogram.edf")))
        if not hyp_matches:
            continue
        pairs.append({"record_id": stem, "psg": psg, "hyp": hyp_matches[0]})

    return pairs


def _intersect_hyp_with_psg(stages_df: pd.DataFrame, n_samp: int, fs: float) -> pd.DataFrame:
    T_sig = (n_samp - 1) / fs
    st = stages_df.copy()

    st = st[st["onset_sec"] < T_sig].copy()

    end_sec = (st["onset_sec"] + st["duration_sec"]).clip(upper=T_sig)
    st["duration_sec"] = end_sec - st["onset_sec"]

    st["start_samp"] = st["start_samp"].clip(lower=0, upper=n_samp)
    st["end_samp"] = st["end_samp"].clip(lower=0, upper=n_samp)

    st = st[(st["duration_sec"] > 0) & (st["end_samp"] > st["start_samp"])].copy()
    return st.reset_index(drop=True)


def _trim_leading_awake_to_sleep_onset(
    X: np.ndarray,                 # (n_samples, d)
    stages_df: pd.DataFrame,
    fs: float,
    *,
    sleep_stages: Optional[set] = None,
    min_sleep_run_sec: float = 60.0,
) -> tuple[np.ndarray, pd.DataFrame, int]:
    """
    Trim X and stages to start at sleep onset. Returns (X_trim, stages_trim, cut_start_samp).
    """
    if sleep_stages is None:
        sleep_stages = {"N1", "N2", "N3", "REM"}

    st = stages_df.reset_index(drop=True)

    cut_start_samp = 0
    onset_idx = None

    # sustained sleep run
    for i in range(len(st)):
        if st.loc[i, "stage"] in sleep_stages:
            dur = 0.0
            j = i
            while j < len(st) and st.loc[j, "stage"] in sleep_stages and dur < min_sleep_run_sec:
                dur += float(st.loc[j, "duration_sec"])
                j += 1
            if dur >= min_sleep_run_sec:
                onset_idx = i
                break

    # fallback: first non-W
    if onset_idx is None:
        for i in range(len(st)):
            if st.loc[i, "stage"] != "W":
                onset_idx = i
                break

    if onset_idx is not None:
        cut_start_samp = int(st.loc[onset_idx, "start_samp"])

    if cut_start_samp <= 0:
        return X, stages_df, 0

    X2 = X[cut_start_samp:, :]

    st2 = stages_df[stages_df["end_samp"] > cut_start_samp].copy()
    st2["start_samp"] = (st2["start_samp"] - cut_start_samp).clip(lower=0)
    st2["end_samp"] = (st2["end_samp"] - cut_start_samp).clip(lower=0)

    shift_sec = cut_start_samp / fs
    st2["onset_sec"] = (st2["onset_sec"] - shift_sec).clip(lower=0)
    st2 = st2[st2["end_samp"] > st2["start_samp"]].reset_index(drop=True)

    return X2, st2, cut_start_samp


def _downsample(X: np.ndarray, fs: float, target_fs: Optional[float]) -> tuple[np.ndarray, float]:
    if target_fs is None or target_fs >= fs:
        return X, fs
    step = int(round(fs / target_fs))
    step = max(step, 1)
    X2 = X[::step, :]
    fs2 = fs / step
    return X2, fs2


def _epoch_stage_distribution_matrix(
    stages_df: pd.DataFrame,
    n_samp: int,
    fs: float,
    window_sec: float,
    stage_order: List[str],
) -> tuple[np.ndarray, int]:
    epoch_len = int(round(window_sec * fs))
    n_epochs = n_samp // epoch_len
    n_stages = len(stage_order)
    idx = {s: i for i, s in enumerate(stage_order)}

    P = np.zeros((n_epochs, n_stages), dtype=np.float32)
    st = stages_df.reset_index(drop=True)

    for e in range(n_epochs):
        a = e * epoch_len
        b = a + epoch_len

        # accumulate overlaps
        for _, row in st.iterrows():
            sa, sb = int(row["start_samp"]), int(row["end_samp"])
            if sb <= a:
                continue
            if sa >= b:
                break
            ov = max(0, min(sb, b) - max(sa, a))
            if ov > 0:
                j = idx.get(row["stage"], idx["UNK"])
                P[e, j] += ov

        if P[e].sum() > 0:
            P[e] /= epoch_len
        else:
            P[e, idx["UNK"]] = 1.0

    return P, epoch_len


def load_sleep_cassette(
    txt_path: str,
    *,
    pick_channels: Optional[List[str]] = None,
    max_records: Optional[int] = None,
    preload: bool = True,
    intersect_time: bool = True,
    trim_leading_w: bool = True,
    min_sleep_run_sec: float = 60.0,
    window_sec: float = 30.0,
    target_fs: Optional[float] = 25.0,
    dtype: np.dtype = np.float32,
) -> List[SleepBanditRecord]:
    txt_path = os.path.abspath(txt_path)
    pairs = _pair_sleep_edf_files(txt_path)
    if max_records is not None:
        pairs = pairs[:max_records]

    out: List[SleepBanditRecord] = []

    for pr in tqdm(pairs, desc="Loading sleep-cassette", unit="rec", dynamic_ncols=True):
        rid, psg_path, hyp_path = pr["record_id"], pr["psg"], pr["hyp"]

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)

            raw = mne.io.read_raw_edf(psg_path, preload=preload, verbose="ERROR")

        if pick_channels is not None:
            keep = [ch for ch in pick_channels if ch in raw.ch_names]
            raw.pick(keep)

        fs = float(raw.info["sfreq"])
        X = raw.get_data().T.astype(dtype, copy=False)  # (n_samples, d)
        channels = list(raw.ch_names)

        # hypnogram annotations -> stages_df
        anns = mne.read_annotations(hyp_path)
        rows = []
        for onset, duration, desc in zip(anns.onset, anns.duration, anns.description):
            stage = _STAGE_MAP.get(desc, desc)
            rows.append(
                dict(
                    onset_sec=float(onset),
                    duration_sec=float(duration),
                    stage=stage,
                    start_samp=int(round(onset * fs)),
                    end_samp=int(round((onset + duration) * fs)),
                )
            )
        stages_df = pd.DataFrame(rows)

        if intersect_time:
            stages_df = _intersect_hyp_with_psg(stages_df, n_samp=X.shape[0], fs=fs)

        if trim_leading_w:
            X, stages_df, _ = _trim_leading_awake_to_sleep_onset(
                X, stages_df, fs, min_sleep_run_sec=min_sleep_run_sec
            )

        # optional downsample AFTER trimming (cheaper)
        X, fs = _downsample(X, fs, target_fs)

        # recompute stages sample indices if we downsampled by integer step
        # (our _downsample uses integer step, so we can scale indices safely)
        if target_fs is not None and fs != float(raw.info["sfreq"]):
            step = int(round(float(raw.info["sfreq"]) / fs))
            stages_df = stages_df.copy()
            stages_df["start_samp"] = (stages_df["start_samp"] / step).astype(int)
            stages_df["end_samp"] = (stages_df["end_samp"] / step).astype(int)

            # also re-intersect after scaling, just to be safe
            stages_df = _intersect_hyp_with_psg(stages_df, n_samp=X.shape[0], fs=fs)
        
        P, epoch_len = _epoch_stage_distribution_matrix(
        stages_df, n_samp=X.shape[0], fs=fs, window_sec=window_sec, stage_order=STAGE_ORDER
        )

        # truncate everything to n_epochs
        n_epochs = P.shape[0]
        X = X[: n_epochs * epoch_len, :]
        P = P[:n_epochs, :]

        # ---------- drop epochs with any non-finite samples ----------
        # reshape to (n_epochs, epoch_len, d)
        X_ep = X.reshape(n_epochs, epoch_len, X.shape[1])

        # mark epochs that are fully finite
        good = np.isfinite(X_ep).all(axis=(1, 2))

        n_bad = int((~good).sum())
        if n_bad > 0:
            # optional: print once per record (or keep silent)
            # print(f"[{rid}] dropping {n_bad}/{n_epochs} epochs due to non-finite values in X")

            X_ep = X_ep[good]
            X = X_ep.reshape(int(good.sum()) * epoch_len, X.shape[1])
            P = P[good]
            n_epochs = int(good.sum())

            # safety: if too much got dropped, skip record
            if n_epochs < 10:
                # skip unusable record
                continue

        out.append(
            SleepBanditRecord(
                record_id=rid,
                fs=fs,
                X=X,
                epoch_len=epoch_len,
                channels=channels,
                P=P,
                stage_order=STAGE_ORDER,
            )
        )

    return out


if __name__ == "__main__":
    cassette_dir = os.path.join(os.path.dirname(__file__), "sleep-cassette")

    recs = load_sleep_cassette(
        cassette_dir,
        pick_channels=["EEG Fpz-Cz", "EEG Pz-Oz", "EOG horizontal"],
        max_records=10,
        intersect_time=True,
        trim_leading_w=True,
        min_sleep_run_sec=60.0,
        window_sec=30.0,
        target_fs=25.0,
    )

    r0 = recs[1]
    print("record:", r0.record_id)
    print("fs:", r0.fs)
    print("X:", r0.X.shape, "channels:", r0.channels)
    print("epoch_len:", r0.epoch_len, "n_epochs:", r0.P.shape[0])
    print("P head:", r0.P[:10])
    print("label counts:", dict(pd.Series(r0.y).value_counts()))