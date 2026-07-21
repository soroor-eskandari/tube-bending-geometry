import pandas as pd

file_path = "/Users/soroureskandari/Master Thesis /tube-bending-geometry/data/raw/unique_bending_setups.csv"

df = pd.read_csv(file_path)

df["Group_ID"] = range(1, len(df) + 1)

df.to_csv(file_path, index=False)

print("Group_ID column added and file saved.")
print(df.head())