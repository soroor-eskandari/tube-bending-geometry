import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==========================================================
# PAGE CONFIG
# ==========================================================
st.set_page_config(page_title="Real vs Synthetic Curves", layout="wide")
st.title("Real vs Synthetic Experiment Comparison")

# ==========================================================
# LOAD DATA
# ==========================================================
file_path = "/Users/soroureskandari/Master Thesis /tube-bending-geometry/data/rf_augmented/final_geometry.csv"

@st.cache_data
def load_data(path):
    return pd.read_csv(path)

df = load_data(file_path)

# ==========================================================
# PREPARE DATA
# ==========================================================
angle_col = "Angle[degree]ORDistance[mm]"
main_col  = "Main-axis [mm]"
sec_col   = "Secondary-axis [mm]"
id_col    = "Experiment_ID"
group_col = "group_id"

df[angle_col] = pd.to_numeric(df[angle_col], errors="coerce")

# ==========================================================
# SIDEBAR
# ==========================================================
st.sidebar.header("Controls")

group_ids = sorted(df[group_col].dropna().unique())

selected_group = st.sidebar.selectbox(
    "Select Group ID",
    group_ids
)

metric = st.sidebar.selectbox(
    "Axis",
    ["Both", "Main", "Secondary"]
)

# ==========================================================
# FILTER BY GROUP
# ==========================================================
df_group = df[df[group_col] == selected_group]

df_real_group = df_group[df_group["Synthetic"] == False]
df_fake_group = df_group[df_group["Synthetic"] == True]

real_ids = df_real_group[id_col].dropna().unique()
fake_ids = df_fake_group[id_col].dropna().unique()

# ==========================================================
# DISPLAY IDS
# ==========================================================
st.write(f"### Showing Group {selected_group}")
st.write("Real Experiment IDs:", real_ids)
st.write("Synthetic Experiment IDs:", fake_ids)

# ==========================================================
# PLOT FUNCTION
# ==========================================================
def plot_axis(value_col, title):

    fig, ax = plt.subplots(figsize=(13,6))

    # REAL SIGNALS (solid)
    for rid in real_ids:
        temp = df_real_group[df_real_group[id_col] == rid].sort_values(angle_col)

        ax.plot(
            temp[angle_col],
            temp[value_col],
            linewidth=2,
            alpha=0.9,
            label=f"Real {rid}"
        )

    # SYNTHETIC SIGNALS (dashed)
    for fid in fake_ids:
        temp = df_fake_group[df_fake_group[id_col] == fid].sort_values(angle_col)

        ax.plot(
            temp[angle_col],
            temp[value_col],
            linestyle="--",
            linewidth=2,
            alpha=0.9,
            label=f"Synthetic {fid}"
        )

    ax.set_title(f"{title} (Group {selected_group})", fontsize=14)
    ax.set_xlabel("Angle")
    ax.set_ylabel(value_col)
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2, fontsize=8)

    st.pyplot(fig)

# ==========================================================
# DISPLAY PLOTS
# ==========================================================
if len(real_ids) == 0 and len(fake_ids) == 0:
    st.warning("No data found for this group.")
else:
    if metric == "Main":
        plot_axis(main_col, "Main Axis")

    elif metric == "Secondary":
        plot_axis(sec_col, "Secondary Axis")

    else:
        plot_axis(main_col, "Main Axis")
        plot_axis(sec_col, "Secondary Axis")