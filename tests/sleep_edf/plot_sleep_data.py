"""Plot raw EEG/EOG signals stacked above a hypnogram for one sleep-cassette record.

Usage:
    python -m tests.sleep_edf.plot_sleep_data                        # first record, 1 hour
    python -m tests.sleep_edf.plot_sleep_data --record 3 --hours 6    # 4th record, 6 hours
    python -m tests.sleep_edf.plot_sleep_data --hours 0.5             # short window (nice morphology)
"""
import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.patches import Patch, ConnectionPatch, Rectangle

from tests.sleep_edf.load_data import load_sleep_cassette, STAGE_ORDER


# Standard hypnogram vertical order: Wake on top, deep sleep on the bottom, REM
# sitting between Wake and N1 (its usual EEG-activity-based placement).
DISPLAY_ORDER = ["W", "REM", "N1", "N2", "N3", "MT", "UNK"]
_DISPLAY_INDEX = {s: i for i, s in enumerate(DISPLAY_ORDER)}
_STAGE_TO_DISPLAY = np.array(
    [_DISPLAY_INDEX[s] for s in STAGE_ORDER], dtype=int
)

# Colors for stage-shaded background bands (matches common sleep-scoring conventions).
STAGE_COLORS = {
    "W":   "#e8887f",   # salmon           (wake)
    "REM": "#e8b04b",   # amber            (REM)
    "N1":  "#cfe0f2",   # very light blue  (light sleep)
    "N2":  "#8fb8dd",   # medium blue      (deeper light sleep)
    "N3":  "#3f6d9e",   # dark blue        (deep sleep / slow-wave)
    "MT":  "#bdbdbd",   # gray             (movement)
    "UNK": None,        # not drawn
}

# Background-band opacity: strong enough to read on the hypnogram, whisper-faint
# behind the signal traces so the waveform stays the foreground.
_BAND_ALPHA_HYPNO = 0.55
_BAND_ALPHA_SIGNAL = 0.16

# One ink for the signal, whichever way it gets drawn. The envelope band is given a
# little alpha only because a solid 2000-bin slab reads much heavier than a 0.5pt line
# at the same color; the hue must not change between the two paths.
_SIGNAL_INK = "#1b1b1b"
_ENVELOPE_ALPHA = 0.9

# Zoom-callout accent. Blue, per the sketch: it sits on top of the amber REM band, so it
# reads clearly there despite the blue stage bands elsewhere in the hypnogram.
_CALLOUT_COLOR = "#1f5fbf"


def _envelope(signal_1d: np.ndarray, n_bins: int = 2000):
    """Bin a dense 1D signal into a per-bin (min, max, mean) envelope for display.

    Returns (bin_centers_idx, y_min, y_max, y_mean) in sample-index units. Drawing the
    min/max pair as a filled band (rather than as an alternating zig-zag line) shows the
    amplitude envelope honestly: at these time scales individual waves are far narrower
    than a pixel, so a "trace" would be a solid ink blob pretending to be morphology.
    Returns None if the signal is sparse enough to plot as a real waveform.
    """
    n = signal_1d.shape[0]
    if n <= 4 * n_bins:
        return None

    n_bins = min(n_bins, n)
    edges = np.linspace(0, n, n_bins + 1, dtype=int)
    # Ragged bins differ by at most 1 sample; pad to a rectangle so we can reduce at once.
    starts, stops = edges[:-1], edges[1:]
    width = int(np.max(stops - starts))
    idx = starts[:, None] + np.arange(width)[None, :]
    valid = idx < stops[:, None]
    idx = np.clip(idx, 0, n - 1)
    block = np.where(valid, signal_1d[idx], np.nan)

    centers = 0.5 * (starts + stops - 1)
    return (
        centers,
        np.nanmin(block, axis=1),
        np.nanmax(block, axis=1),
        np.nanmean(block, axis=1),
    )


def _stage_runs(dominant_stage_names):
    """Collapse a per-epoch stage list into (start_epoch, stop_epoch, stage) runs.

    Drawing one span per epoch makes adjacent same-stage spans overlap at their edges,
    which shows up as faint vertical seams across every panel. One span per run fixes it.
    """
    runs = []
    start = 0
    for e in range(1, len(dominant_stage_names) + 1):
        if e == len(dominant_stage_names) or dominant_stage_names[e] != dominant_stage_names[start]:
            runs.append((start, e, dominant_stage_names[start]))
            start = e
    return runs


def _add_stage_bands(ax, t_epoch_min, dominant_stage_names, epoch_sec_min, alpha):
    """Fill the background of `ax` with one colored band per contiguous stage run."""
    y0, y1 = ax.get_ylim()
    for a, b, stg in _stage_runs(dominant_stage_names):
        color = STAGE_COLORS.get(stg)
        if color is None:
            continue
        ax.axvspan(t_epoch_min[a], t_epoch_min[b - 1] + epoch_sec_min,
                   ymin=0, ymax=1, facecolor=color, alpha=alpha,
                   edgecolor="none", linewidth=0, zorder=0)
    # Preserve original y-limits (axvspan can shift them subtly).
    ax.set_ylim(y0, y1)


def plot_signal_with_hypnogram(
    cassette_dir: str = "tests/sleep_edf/sleep-cassette",
    record_index: int = 0,
    hours: float = 1.0,
    out_dir: str = "experiments/figs/sleep_edf",
    pick_channels=("EEG Fpz-Cz", "EEG Pz-Oz", "EOG horizontal"),
    target_fs: float = 25.0,
):
    # Load only enough records to reach record_index (avoids reading all 150 cassettes).
    recs = load_sleep_cassette(
        cassette_dir,
        pick_channels=list(pick_channels),
        max_records=record_index + 1,
        intersect_time=True,
        trim_leading_w=True,
        min_sleep_run_sec=60.0,
        window_sec=30.0,
        target_fs=target_fs,
    )
    if record_index >= len(recs):
        raise IndexError(
            f"Requested record_index={record_index} but only {len(recs)} records were loaded."
        )
    rec = recs[record_index]

    # How many epochs / samples cover the requested time window.
    epoch_sec = 30.0
    epoch_sec_min = epoch_sec / 60.0
    n_epochs_avail = rec.P.shape[0]
    n_epochs_plot = int(min(hours * 3600 / epoch_sec, n_epochs_avail))
    n_samples_plot = n_epochs_plot * rec.epoch_len

    t_epoch_min = np.arange(n_epochs_plot) * epoch_sec_min

    # Dominant stage per epoch (argmax over the overlap-fraction row).
    dominant_stage_idx = np.argmax(rec.P[:n_epochs_plot], axis=1)
    dominant_stage_name = [STAGE_ORDER[i] for i in dominant_stage_idx]
    display_y = _STAGE_TO_DISPLAY[dominant_stage_idx]

    # Only label stages this window actually contains (dead MT/UNK rows otherwise eat a
    # third of the hypnogram's height and flatten the trace into an unreadable ribbon).
    present = set(dominant_stage_name)
    shown_stages = [s for s in DISPLAY_ORDER if s in present or s in ("W", "N1", "N2", "N3")]
    y_lo = min(_DISPLAY_INDEX[s] for s in shown_stages)
    y_hi = max(_DISPLAY_INDEX[s] for s in shown_stages)

    # Layout: hypnogram row + one row per channel + a slim row that holds the legend.
    # Giving the legend its own axes is what keeps it off the x-axis label.
    n_channels = len(rec.channels)
    fig = plt.figure(figsize=(13, 2.4 + 1.4 * n_channels))
    gs = gridspec.GridSpec(
        n_channels + 2, 1,
        height_ratios=[1.4] + [1.0] * n_channels + [0.5],
        hspace=0.18,
    )

    # ----- Hypnogram (top) -----
    ax_h = fig.add_subplot(gs[0])
    # step-post so each epoch's stage extends from its start to the next epoch's start.
    x_step = np.concatenate([t_epoch_min, [t_epoch_min[-1] + epoch_sec_min]])
    y_step = np.concatenate([display_y, [display_y[-1]]])
    ax_h.step(x_step, y_step, where="post", color="#14203a", linewidth=1.6, zorder=3,
              solid_joinstyle="miter")
    ax_h.set_yticks([_DISPLAY_INDEX[s] for s in shown_stages])
    ax_h.set_yticklabels(shown_stages)
    ax_h.set_ylim(y_hi + 0.45, y_lo - 0.45)   # inverted: Wake on top
    ax_h.set_ylabel("Sleep stage")
    ax_h.set_title(
        f"Sleep-EDF cassette {rec.record_id} — first {hours:g} hour(s) after sleep onset\n"
        f"channels: {', '.join(rec.channels)}   fs = {rec.fs:g} Hz"
    )
    ax_h.grid(True, axis="y", alpha=0.25, linewidth=0.6, zorder=1)
    ax_h.tick_params(labelbottom=False)
    _add_stage_bands(ax_h, t_epoch_min, dominant_stage_name, epoch_sec_min,
                     alpha=_BAND_ALPHA_HYPNO)

    # ----- Signal channels -----
    for i, ch_name in enumerate(rec.channels):
        ax = fig.add_subplot(gs[i + 1], sharex=ax_h)

        # Convert V → µV for display.
        sig_uv = rec.X[:n_samples_plot, i] * 1e6
        env = _envelope(sig_uv, n_bins=2000)

        if env is None:
            t_min = np.arange(sig_uv.shape[0]) / rec.fs / 60.0
            ax.plot(t_min, sig_uv, linewidth=0.5, color=_SIGNAL_INK, zorder=2)
        else:
            centers, y_lo_env, y_hi_env, y_mean = env
            t_min = centers / rec.fs / 60.0
            ax.fill_between(t_min, y_lo_env, y_hi_env, color=_SIGNAL_INK,
                            alpha=_ENVELOPE_ALPHA, linewidth=0, zorder=2)

        ax.set_ylabel(f"{ch_name}\n(µV)", fontsize=9)
        ax.grid(True, axis="y", alpha=0.25, linewidth=0.6, zorder=1)
        ax.axhline(0, color="#8a8a8a", linewidth=0.6, alpha=0.5, zorder=1)
        _add_stage_bands(ax, t_epoch_min, dominant_stage_name, epoch_sec_min,
                         alpha=_BAND_ALPHA_SIGNAL)
        if i < n_channels - 1:
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xlabel("Time (minutes since sleep onset)")

    ax_h.set_xlim(0, t_epoch_min[-1] + epoch_sec_min)

    # ----- Legend row (its own axes, so it can never overlap the x-label) -----
    ax_leg = fig.add_subplot(gs[-1])
    ax_leg.axis("off")
    legend_stages = [s for s in ["W", "REM", "N1", "N2", "N3", "MT"] if s in present]
    handles = [Patch(facecolor=STAGE_COLORS[s], alpha=_BAND_ALPHA_HYPNO,
                     edgecolor="none", label=s) for s in legend_stages]
    ax_leg.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False,
                  handlelength=1.6, handleheight=0.9, columnspacing=1.8,
                  borderpad=0.0, fontsize=9)

    os.makedirs(out_dir, exist_ok=True)
    hours_tag = f"{hours:g}".replace(".", "p")
    fname = f"sleep_signal_hypnogram_{rec.record_id}_{hours_tag}h.png"
    out_path = os.path.join(out_dir, fname)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to {out_path}")


def plot_paper_figure(
    cassette_dir: str = "tests/sleep_edf/sleep-cassette",
    record_index: int = 0,
    hours: float = 4.0,
    zoom_epoch: int | None = None,
    zoom_stage: str = "REM",
    box_center_min: float = 100.0,
    box_pad_min: float = 3.0,
    out_dir: str = "experiments/figs/sleep_edf",
    pick_channels=("EEG Fpz-Cz", "EEG Pz-Oz", "EOG horizontal"),
    target_fs: float = 25.0,
    show_title: bool = False,
    show_question: bool = True,
):
    """Two-panel figure for the paper: task horizon on top, one context window below.

    The top panel is the hypnogram over the full episode the bandit plays, which gives
    the horizon and the sleep architecture. The bottom panels magnify a SINGLE 30-s epoch
    at full sample rate — that window is exactly the path the signature is computed on,
    so the reader can see that each round's context has shape, not just an amplitude.
    """
    recs = load_sleep_cassette(
        cassette_dir,
        pick_channels=list(pick_channels),
        max_records=record_index + 1,
        intersect_time=True,
        trim_leading_w=True,
        min_sleep_run_sec=60.0,
        window_sec=30.0,
        target_fs=target_fs,
    )
    if record_index >= len(recs):
        raise IndexError(
            f"Requested record_index={record_index} but only {len(recs)} records were loaded."
        )
    rec = recs[record_index]

    epoch_sec = 30.0
    epoch_sec_min = epoch_sec / 60.0
    n_epochs_plot = int(min(hours * 3600 / epoch_sec, rec.P.shape[0]))
    t_epoch_min = np.arange(n_epochs_plot) * epoch_sec_min

    dominant_stage_idx = np.argmax(rec.P[:n_epochs_plot], axis=1)
    dominant_stage_name = [STAGE_ORDER[i] for i in dominant_stage_idx]
    display_y = _STAGE_TO_DISPLAY[dominant_stage_idx]

    # Find the `zoom_stage` block nearest `box_center_min`; the box outlines that whole
    # block and the magnified window is taken from its middle, so the box and the traces
    # below always refer to the same place.
    stage_runs = [(a, b) for a, b, s in _stage_runs(dominant_stage_name) if s == zoom_stage]
    if stage_runs:
        box_a, box_b = min(
            stage_runs,
            key=lambda ab: abs(0.5 * (ab[0] + ab[1]) * epoch_sec_min - box_center_min),
        )
    else:
        box_a, box_b = 0, n_epochs_plot
    if zoom_epoch is None:
        zoom_epoch = (box_a + box_b) // 2
    zoom_epoch = int(np.clip(zoom_epoch, 0, n_epochs_plot - 1))
    zoom_stage_name = dominant_stage_name[zoom_epoch]

    present = set(dominant_stage_name)
    shown_stages = [s for s in DISPLAY_ORDER if s in present or s in ("W", "N1", "N2", "N3")]
    y_lo = min(_DISPLAY_INDEX[s] for s in shown_stages)
    y_hi = max(_DISPLAY_INDEX[s] for s in shown_stages)

    # Nested gridspecs: the hypnogram spans the full width, the middle row splits into
    # the boxed signal stack (left) and the question (right), and the channel stack wants
    # a much tighter hspace than the gap that has to hold the callout arrow.
    n_channels = len(rec.channels)
    fig = plt.figure(figsize=(11, 2.4 + 1.05 * n_channels))
    gs = gridspec.GridSpec(
        2, 1,
        height_ratios=[1.25, 0.85 * n_channels],
        hspace=0.6,   # the gap carries the x-label, the legend and the callout arrow
    )
    gs_mid = gridspec.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=gs[1], width_ratios=[1.0, 0.8], wspace=0.08)
    gs_zoom = gridspec.GridSpecFromSubplotSpec(
        n_channels, 1, subplot_spec=gs_mid[0], hspace=0.12)

    # ----- Top: hypnogram over the whole episode -----
    ax_h = fig.add_subplot(gs[0])
    x_step = np.concatenate([t_epoch_min, [t_epoch_min[-1] + epoch_sec_min]])
    y_step = np.concatenate([display_y, [display_y[-1]]])
    ax_h.step(x_step, y_step, where="post", color="#14203a", linewidth=1.4, zorder=3)
    ax_h.set_yticks([_DISPLAY_INDEX[s] for s in shown_stages])
    ax_h.set_yticklabels(shown_stages)
    ax_h.set_ylim(y_hi + 0.45, y_lo - 0.45)
    ax_h.set_ylabel("Sleep stage")
    ax_h.set_xlabel("Time since sleep onset (min)", fontsize=9, labelpad=2)
    ax_h.grid(True, axis="y", alpha=0.25, linewidth=0.6, zorder=1)
    _add_stage_bands(ax_h, t_epoch_min, dominant_stage_name, epoch_sec_min,
                     alpha=_BAND_ALPHA_HYPNO)
    ax_h.set_xlim(0, t_epoch_min[-1] + epoch_sec_min)
    if show_title:
        ax_h.set_title(
            f"Sleep-EDF {rec.record_id}: {n_epochs_plot} bandit rounds "
            f"(one 30 s epoch each, {rec.fs:g} Hz)", fontsize=10, pad=6)

    # Outline the whole stage block the magnified window is drawn from. A single epoch is
    # ~1/500 of the axis width here, so boxing the block is what the reader can actually
    # see; the amber band underneath already marks it as REM. The box is padded past the
    # block on all four sides and drawn unclipped, so it reads as a callout sitting on top
    # of the hypnogram rather than as another stage band flush inside it.
    t_box_a = t_epoch_min[box_a]
    t_box_b = t_epoch_min[box_b - 1] + epoch_sec_min
    y_top, y_bot = ax_h.get_ylim()
    y_span = abs(y_top - y_bot)
    pad_y = 0.09 * y_span
    ax_h.add_patch(Rectangle(
        (t_box_a - box_pad_min, min(y_top, y_bot) - pad_y),
        (t_box_b - t_box_a) + 2 * box_pad_min, y_span + 2 * pad_y,
        facecolor="none", edgecolor=_CALLOUT_COLOR, linewidth=1.8,
        zorder=6, clip_on=False))

    # ----- Bottom: one context window at full resolution -----
    L = rec.epoch_len
    sl = slice(zoom_epoch * L, (zoom_epoch + 1) * L)
    t_sec = np.arange(L) / rec.fs
    zoom_axes = []
    for i, ch_name in enumerate(rec.channels):
        ax = fig.add_subplot(gs_zoom[i], sharex=zoom_axes[0] if zoom_axes else None)
        zoom_axes.append(ax)

        sig_uv = rec.X[sl, i] * 1e6
        ax.plot(t_sec, sig_uv, linewidth=0.5, color=_SIGNAL_INK, zorder=3)
        ax.set_facecolor(STAGE_COLORS[zoom_stage_name])
        ax.patch.set_alpha(_BAND_ALPHA_SIGNAL)
        # Bare traces: no ticks, labels, grid or spines. The single outer box supplies the
        # frame and the caption supplies the units; scale is carried by the caption text,
        # not by axis furniture.
        ax.set_xticks([])
        ax.set_yticks([])
        for side in ax.spines.values():
            side.set_visible(False)
    zoom_axes[0].set_xlim(0, L / rec.fs)

    # Extent of the channel stack in figure coords — the caption, the arrow target and
    # the "information" label are all anchored to it. Positions come straight from the
    # gridspec (no tight/constrained layout in play), so they need no draw pass.
    boxes = [ax.get_position() for ax in zoom_axes]
    pad = 0.012
    x0 = min(b.x0 for b in boxes) - pad
    x1 = max(b.x1 for b in boxes) + pad
    y0 = min(b.y0 for b in boxes) - pad
    y1 = max(b.y1 for b in boxes) + pad
    fig.text(0.5 * (x0 + x1), y0 - 0.03, "EEG, EOG time series",
             ha="center", va="top", fontsize=11)

    # Callout arrow, routed down-LEFT as in the sketch. Aiming at the box's near corner
    # instead would drag the arrow through the hypnogram's x-label; going left keeps the
    # centre of that gap clear.
    pos_h = ax_h.get_position()
    x_from = pos_h.x0 + pos_h.width * (
        0.5 * (t_box_a + t_box_b) / (t_epoch_min[-1] + epoch_sec_min))
    x_to, y_to = 0.5 * (x0 + x1), y1
    # Start at the box's own bottom edge, which hangs pad_y (9% of the y-range) below the
    # axes because the box is drawn unclipped.
    y_from = pos_h.y0 - 0.09 * pos_h.height
    fig.add_artist(ConnectionPatch(
        xyA=(x_from, y_from), coordsA=fig.transFigure,
        xyB=(x_to, y_to), coordsB=fig.transFigure,
        arrowstyle="-|>", mutation_scale=14, shrinkA=2, shrinkB=5,
        color=_CALLOUT_COLOR, linewidth=1.3, zorder=10))
    # Label on the left, clear of the arrow: the arrow runs from the box down to the
    # centre of the stack, so the left edge is the one part of the gap it never crosses.
    fig.text(x0 + 0.02, y_to + 0.008, "information", ha="left", va="bottom",
             fontsize=11, color=_CALLOUT_COLOR)

    # ----- Right: the question this task poses -----
    if show_question:
        ax_q = fig.add_subplot(gs_mid[1])
        ax_q.axis("off")
        ax_q.text(0.5, 0.5,
                  "Q: Based on the time series,\nwhat is the sleep stage?",
                  ha="center", va="center", fontsize=18, linespacing=1.6)

    # ----- Legend, tucked directly under the hypnogram's x-label -----
    # Centred on the hypnogram, which keeps it clear of the callout arrow on the left.
    legend_stages = [s for s in ["W", "REM", "N1", "N2", "N3", "MT"] if s in present]
    handles = [Patch(facecolor=STAGE_COLORS[s], alpha=_BAND_ALPHA_HYPNO,
                     edgecolor="none", label=s) for s in legend_stages]
    fig.legend(handles=handles, loc="upper center",
               bbox_to_anchor=(pos_h.x0 + 0.5 * pos_h.width, pos_h.y0 - 0.075),
               ncol=len(handles), frameon=False, handlelength=1.6, handleheight=0.9,
               columnspacing=1.8, borderpad=0.0, fontsize=9)

    os.makedirs(out_dir, exist_ok=True)
    hours_tag = f"{hours:g}".replace(".", "p")
    stem = f"sleep_task_and_context_{rec.record_id}_{hours_tag}h_ep{zoom_epoch}"
    out_paths = []
    for ext, kw in (("pdf", {}), ("png", {"dpi": 300})):
        out_path = os.path.join(out_dir, f"{stem}.{ext}")
        fig.savefig(out_path, bbox_inches="tight", **kw)
        out_paths.append(out_path)
    plt.close(fig)
    for p_ in out_paths:
        print(f"Saved figure to {p_}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cassette-dir", type=str,
                   default="tests/sleep_edf/sleep-cassette")
    p.add_argument("--record", type=int, default=0,
                   help="Which cassette to plot (0-based).")
    p.add_argument("--hours", type=float, default=None,
                   help="How many hours (after sleep onset) to display. "
                        "Default: 1 for the explorer view, 4 for --paper.")
    p.add_argument("--out-dir", type=str, default="experiments/figs/sleep_edf")
    p.add_argument("--paper", action="store_true",
                   help="Render the two-panel paper figure (episode hypnogram + one "
                        "magnified context window) as both PDF and PNG.")
    p.add_argument("--zoom-epoch", type=int, default=None,
                   help="[--paper] Which epoch/round to magnify. Default: mid-REM.")
    p.add_argument("--zoom-stage", type=str, default="REM",
                   help="[--paper] Stage the outlined block is picked from.")
    p.add_argument("--box-center-min", type=float, default=100.0,
                   help="[--paper] Outline the --zoom-stage block nearest this minute.")
    p.add_argument("--title", action="store_true",
                   help="[--paper] Add an in-figure title (off by default; use a caption).")
    p.add_argument("--no-question", action="store_true",
                   help="[--paper] Drop the 'what is the sleep stage?' text panel.")
    args = p.parse_args()

    if args.paper:
        plot_paper_figure(
            cassette_dir=args.cassette_dir,
            record_index=args.record,
            hours=4.0 if args.hours is None else args.hours,
            zoom_epoch=args.zoom_epoch,
            zoom_stage=args.zoom_stage,
            box_center_min=args.box_center_min,
            out_dir=args.out_dir,
            show_title=args.title,
            show_question=not args.no_question,
        )
    else:
        plot_signal_with_hypnogram(
            cassette_dir=args.cassette_dir,
            record_index=args.record,
            hours=1.0 if args.hours is None else args.hours,
            out_dir=args.out_dir,
        )


if __name__ == "__main__":
    main()
