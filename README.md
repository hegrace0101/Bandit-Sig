# Bandit-Sig

Signature-based contextual bandits. The context at each round is a path (window of a multivariate time series) rather than a single feature vector, and the policy uses path signatures as features for UCB-style action selection.

## What's here

- **`sigUCB`** — linear UCB on truncated path-signature features (with optional time augmentation, time-channel pruning, terminal-state concatenation, and intercept).
- **`ammSigUCB`** — `sigUCB` with the tuned exploration coefficient `gamma` replaced by the closed-form AMM (Analytic Martingale Mixtures) confidence radius from Flynn, Reeb, Kandemir & Peters, NeurIPS 2023 (arXiv:2309.14298).
- **`linUCB`** — baseline linear UCB; takes a `feat_fn` so it can run on any fixed feature map. The real-data experiments (`temp_sensor_exp`, `sleep_edf`, `hospital_staff`) each sweep three variants over the same window (see `utils/features.py`):
  - `linUCB_stats` — per-channel temporal stats (mean/std/min/max/first/last/slope).
  - `linUCB_poly` — degree-2 polynomial features on the time-averaged window.
  - `linUCB_rff` — random Fourier features approximating an RBF kernel.
- **`KernelUCB`** — per-arm kernel ridge regression UCB (RBF or polynomial kernel), with incremental block-inverse updates and optional memory cap.

All algorithms expose the same `select(context)` / `update(context, action, reward)` API.

## Layout

```
src/sigbandit/
  algos/
    ucb.py            # sigUCB, ammSigUCB, linUCB
    kernel_ucb.py     # KernelUCB, SigKernelUCB
  envs/
    bandit.py         # generic Bandit driver
    context.py        # ContextualProcess + synthetic generators (BM, GBM, OU, AR1, ARMA, regime-switching, trend+seasonal)
    reward.py         # min/max, linear, nested-extrema, path-mix, roughness rewards
    sleep_bandit.py     # Sleep-EDF environment
    temp_sensor_bandit.py  # Intel temp-sensor environment
    hospital_bandit.py     # NYC hospital staffing (newsvendor) environment
  utils/
    signature.py      # signature / log-signature features, time augmentation, time-channel pruning
    features.py        # non-signature feature transforms (flatten, temporal stats, RFF) for linUCB/KernelUCB baselines

tests/
  test_ucb_sig.py     # synthetic-process sweep: tunes hyperparameters on held-out seeds, evaluates regret across processes/rewards
  test_regret_vs_d.py # regret as a function of path dimension
  test_regret_vs_k.py # regret as a function of number of arms
  check_diversity.py
  check_diversity_nested_mc.py  # nested Monte Carlo estimate of the action-conditional feature second moment (Assumption 5.1(ii))
  temp_sensor_exp/    # Intel Berkeley lab temperature-sensor experiment
  sleep_edf/          # Sleep-EDF stage-prediction experiment
    plot_sleep_data.py  # standalone plot of raw EEG/EOG + hypnogram for one cassette record
  hospital_staff/     # NYC hospital staffing experiment

experiments/figs/     # generated plots
notebooks/            # exploratory notebooks
```

## Install

Requires Python 3.10+.

```bash
pip install numpy pandas matplotlib tqdm iisignature
```

Then add `src/` to your `PYTHONPATH` (or `pip install -e .` if you add a `pyproject.toml`):

```bash
export PYTHONPATH="$PWD/src:$PYTHONPATH"
```

Real-data experiments live under `tests/{temp_sensor_exp,sleep_edf,hospital_staff}/`. Each has a `load_data.py` and a `main.py`. All commands below assume the repo root as the working directory (paths are relative to it).

### 1. Intel temperature-sensor experiment

Predicts which lab sensor will record the next-hour max/avg temperature, using a window of recent readings as the path context.

Data: download the Intel Berkeley Lab sensor data and place the raw `data.txt` at `tests/temp_sensor_exp/temp_sensor.txt`. Source: <https://db.csail.mit.edu/labdata/labdata.html>. The loader keeps the `2004-02-28` to `2004-03-08` window, resamples to 5-minute bins, drops sensors with <70% coverage, and appends sin/cos time-of-day features.

```bash
python -m tests.temp_sensor_exp.main
```

`main()` parameters in [main.py:464](tests/temp_sensor_exp/main.py#L464):

| param | default | meaning |
|---|---|---|
| `txt_path` | `tests/temp_sensor_exp/temp_sensor.txt` | raw Intel-lab data file |
| `freq` | `"5min"` | resampling bin width |
| `reward_mode` | `"max"` | `"max"` (next-hour max temp) or `"avg"` |
| `sig_type` | `"sig"` | `"sig"` or `"logsig"` |
| `sig_depth` | `3` | signature truncation depth `N` |
| `d0` | `3` | number of anchor sensors used as path channels |
| `window_L` | `12` | window length (12 × 5min = 1 hour) |
| `lam` | `1.0` | ridge regularization |
| `gamma` | `1.0` | UCB exploration coefficient |
| `n_runs` | `20` | seeds per configuration |
| `seed0` | `123` | base seed (`seeds = seed0 + arange(n_runs)`) |
| `out_dir` | `experiments/figs/temp_sensor` | where plots are written |

### 2. Sleep-EDF experiment

Predicts the sleep stage of the next 30-second epoch from the previous epoch's EEG window. Reward is the overlap fraction between the chosen stage and the ground-truth annotations.

Data: download the Sleep-EDF *Sleep Cassette* PSG + Hypnogram `.edf` files into `tests/sleep_edf/sleep-cassette/`. Source: <https://physionet.org/content/sleep-edfx/1.0.0/>. Requires `mne` for EDF parsing:

```bash
pip install mne
python -m tests.sleep_edf.main
```

`main()` parameters in [main.py:447](tests/sleep_edf/main.py#L447):

| param | default | meaning |
|---|---|---|
| `cassette_dir` | `tests/sleep_edf/sleep-cassette` | directory of `*-PSG.edf` / `*-Hypnogram.edf` pairs |
| `pick_channels` | `("EEG Fpz-Cz", "EEG Pz-Oz", "EOG horizontal")` | EDF channels kept as path dimensions |
| `max_records` | `None` | cap on number of cassettes loaded (`None` = all) |
| `sig_type` | `"sig"` | `"sig"` or `"logsig"` |
| `sig_depth` | `3` | signature truncation depth `N` |
| `warmup_epochs` | `200` | epochs of full-info warmup before bandit play |
| `seed0` | `123` | base seed (one run per cassette, seeded `seed0 + i`) |
| `out_dir` | `experiments/figs/sleep_edf` | where plots are written |

Cassettes shorter than `T_CUT + warmup_epochs` (1000 + warmup) epochs are skipped.

### 3. NYC hospital-staffing experiment

Weekly newsvendor staffing decisions driven by daily vomiting-related ED visit counts. Context is the last `window_L_days` of daily counts; reward is the negative newsvendor loss against the realized weekly demand.

Data: the CSV is committed at `tests/hospital_staff/nyc_vomit_count.csv` (NYC EpiQuery export, Citywide / all age groups), so no extra download is needed.

```bash
python -m tests.hospital_staff.main
```

`main()` parameters in [main.py:454](tests/hospital_staff/main.py#L454):

| param | default | meaning |
|---|---|---|
| `csv_path` | `tests/hospital_staff/nyc_vomit_count.csv` | NYC EpiQuery export |
| `sig_type` | `"sig"` | `"sig"` or `"logsig"` |
| `sig_depth` | `3` | signature truncation depth `N` |
| `window_L_days` | `7` | days of daily counts in each context window |
| `lam` | `1.0` | ridge regularization |
| `gamma` | `1.0` | UCB exploration coefficient |
| `warmup_weeks` | `52` | weeks of full-info warmup before bandit play |
| `n_runs` | `50` | seeds per configuration |
| `seed0` | `123` | base seed (`seeds = seed0 + arange(n_runs)`) |
| `out_dir` | `experiments/figs/hospital_staff` | where plots are written |
