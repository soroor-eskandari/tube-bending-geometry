import streamlit as st
import pandas as pd
import plotly.express as px

# ---------------------------
# Load data
# ---------------------------
@st.cache_data
def load_data():
    machine = pd.read_csv("/Users/soroureskandari/Master Thesis /tube-bending-geometry/data/processed/machine.csv")
    movement = pd.read_csv("/Users/soroureskandari/Master Thesis /tube-bending-geometry/data/processed/movement.csv")
    return machine, movement

machine_df, movement_df = load_data()

# ---------------------------
# UI
# ---------------------------
st.title("Tube Bending Experiment Dashboard")

# Get unique experiment IDs
experiment_ids = sorted(machine_df["Experiment_ID"].unique())

selected_exp = st.selectbox("Select Experiment ID", experiment_ids)

# ---------------------------
# Filter data
# ---------------------------
machine_filtered = machine_df[machine_df["Experiment_ID"] == selected_exp]
movement_filtered = movement_df[movement_df["Experiment_ID"] == selected_exp]

# ---------------------------
# MACHINE PLOT
# ---------------------------
st.subheader("Machine Signals")

machine_melted = machine_filtered.melt(
    id_vars=["Time_[s]"],
    value_vars=[col for col in machine_filtered.columns if col not in ["Experiment_ID", "Time_[s]"]],
    var_name="Signal",
    value_name="Value"
)

fig_machine = px.line(
    machine_melted,
    x="Time_[s]",
    y="Value",
    color="Signal",
    title=f"Machine Signals - Experiment {selected_exp}"
)

st.plotly_chart(fig_machine, use_container_width=True)

# ---------------------------
# MOVEMENT PLOT
# ---------------------------
st.subheader("Movement Signals")

movement_melted = movement_filtered.melt(
    id_vars=["Time_[s]"],
    value_vars=[col for col in movement_filtered.columns if col not in ["Experiment_ID", "Time_[s]"]],
    var_name="Signal",
    value_name="Value"
)

fig_movement = px.line(
    movement_melted,
    x="Time_[s]",
    y="Value",
    color="Signal",
    title=f"Movement Signals - Experiment {selected_exp}"
)

st.plotly_chart(fig_movement, use_container_width=True)