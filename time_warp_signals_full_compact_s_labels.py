# %%
# ============================================================
# Compact time-warp comparison figure
# 12 signals
# Layout: 6 rows x 2 columns
# Signal titles: S1 to S12
# ============================================================

import matplotlib.pyplot as plt
import numpy as np


# ------------------------------------------------------------
# Figure typography
# ------------------------------------------------------------

TITLE_FONT_SIZE = 10
TICK_FONT_SIZE = 9
AXIS_FONT_SIZE = 10

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": [
        "Arial",
        "Liberation Sans",
        "DejaVu Sans",
    ],

    "font.size": TICK_FONT_SIZE,
    "axes.titlesize": TITLE_FONT_SIZE,
    "axes.labelsize": AXIS_FONT_SIZE,
    "xtick.labelsize": TICK_FONT_SIZE,
    "ytick.labelsize": TICK_FONT_SIZE,

    "text.color": "#000000",
    "axes.labelcolor": "#000000",
    "axes.titlecolor": "#000000",
    "xtick.color": "#000000",
    "ytick.color": "#000000",

    "font.weight": "bold",
    "axes.titleweight": "bold",
    "axes.labelweight": "bold",

    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# ------------------------------------------------------------
# Signal label
# ------------------------------------------------------------

def clean_warp_signal_label(signal_name):

    label = signal_name

    replacements = {
        "MACHINE_": "",
        "_Max_Torque_[%]": " torque",
        "_Movement_[mm]": " movement",
        "_Angle_[\u00b0]": " angle",
        "_": " ",
    }

    for old_text, new_text in replacements.items():
        label = label.replace(
            old_text,
            new_text,
        )

    return label


# ------------------------------------------------------------
# Use all retained signals
# ------------------------------------------------------------

selected_signals = (
    example_signal_names.copy()
)

if len(selected_signals) != 12:
    raise ValueError(
        f"Expected 12 signals, "
        f"but found {len(selected_signals)}."
    )

signal_label_map = {
    f"S{index}": signal_name
    for index, signal_name in enumerate(
        selected_signals,
        start=1,
    )
}


# ------------------------------------------------------------
# Create 6 x 2 layout
# ------------------------------------------------------------

fig, axes = plt.subplots(
    nrows=6,
    ncols=2,
    figsize=(11.5, 9.5),
)

axes = np.asarray(
    axes
).reshape(-1)


# ------------------------------------------------------------
# Time values
# ------------------------------------------------------------

time_values = (
    original_example[TIME_COL]
    .to_numpy(dtype=float)
)


# ------------------------------------------------------------
# Plot signals
# ------------------------------------------------------------

for ax, (short_label, signal_name) in zip(
    axes,
    signal_label_map.items(),
):

    original_values = (
        original_example[signal_name]
        .to_numpy(dtype=float)
    )

    warped_values = (
        warped_example[signal_name]
        .to_numpy(dtype=float)
    )

    valid = (
        np.isfinite(time_values)
        & np.isfinite(original_values)
        & np.isfinite(warped_values)
    )

    if valid.sum() < 2:
        raise ValueError(
            f"Insufficient finite values for "
            f"{signal_name}."
        )

    # --------------------------------------------------------
    # Original signal
    # --------------------------------------------------------

    ax.plot(
        time_values[valid],
        original_values[valid],
        color="black",
        linewidth=1.25,
        zorder=2,
    )

    # --------------------------------------------------------
    # Time-warped signal
    # --------------------------------------------------------

    ax.plot(
        time_values[valid],
        warped_values[valid],
        color="#4C78A8",
        linewidth=0.90,
        alpha=0.90,
        zorder=3,
    )

    # --------------------------------------------------------
    # Signal short name
    # --------------------------------------------------------

    ax.set_title(
        short_label,
        fontsize=TITLE_FONT_SIZE,
        fontweight="bold",
        color="#000000",
        pad=3,
    )

    # --------------------------------------------------------
    # X-axis
    # --------------------------------------------------------

    ax.set_xlim(
        0,
        85,
    )

    ax.set_xticks(
        [0, 20, 40, 60, 80]
    )

    ax.set_xlabel("")

    # --------------------------------------------------------
    # No y-axis label
    # --------------------------------------------------------

    ax.set_ylabel("")

    # --------------------------------------------------------
    # Tick appearance
    # --------------------------------------------------------

    ax.tick_params(
        axis="both",
        which="both",
        labelsize=TICK_FONT_SIZE,
        labelcolor="#000000",
        color="#000000",
    )

    for tick_label in ax.get_xticklabels():
        tick_label.set_color("#000000")
        tick_label.set_fontweight("bold")

    for tick_label in ax.get_yticklabels():
        tick_label.set_color("#000000")
        tick_label.set_fontweight("bold")

    # --------------------------------------------------------
    # Grid
    # --------------------------------------------------------

    ax.grid(
        alpha=0.16,
        linewidth=0.6,
    )


# ------------------------------------------------------------
# Time labels only on the bottom row
# ------------------------------------------------------------

axes[-2].set_xlabel(
    "Time [s]",
    fontsize=AXIS_FONT_SIZE,
    fontweight="bold",
    color="#000000",
)

axes[-1].set_xlabel(
    "Time [s]",
    fontsize=AXIS_FONT_SIZE,
    fontweight="bold",
    color="#000000",
)


# ------------------------------------------------------------
# Final explicit typography pass
# ------------------------------------------------------------

for ax in axes:

    ax.title.set_color("#000000")
    ax.title.set_fontweight("bold")

    ax.xaxis.label.set_color("#000000")
    ax.xaxis.label.set_fontweight("bold")

    ax.yaxis.label.set_color("#000000")
    ax.yaxis.label.set_fontweight("bold")

    for text in ax.get_xticklabels():
        text.set_color("#000000")
        text.set_fontweight("bold")

    for text in ax.get_yticklabels():
        text.set_color("#000000")
        text.set_fontweight("bold")

for text in fig.findobj(match=plt.Text):
    text.set_color("#000000")
    text.set_fontweight("bold")


# ------------------------------------------------------------
# Layout
# ------------------------------------------------------------

fig.subplots_adjust(
    left=0.08,
    right=0.98,
    top=0.975,
    bottom=0.055,
    hspace=0.52,
    wspace=0.22,
)


# ------------------------------------------------------------
# Save
# Legend intentionally omitted
# ------------------------------------------------------------

output_path = (
    FIGURE_DIR
    / "time_warp_signals_full_compact.pdf"
)

fig.savefig(
    output_path,
    bbox_inches="tight",
)

plt.show()

print(
    "Figure saved to:",
    output_path,
)

print("\nS1 to S12 signal list:")
for short_label, signal_name in signal_label_map.items():
    print(
        f"{short_label}: {signal_name} "
        f"({clean_warp_signal_label(signal_name)})"
    )
