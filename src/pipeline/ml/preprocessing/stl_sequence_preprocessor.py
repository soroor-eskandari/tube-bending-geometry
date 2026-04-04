import pandas as pd

class STLSequencePreprocessor:
    """
    Preprocess STL point sequences for machine learning.

    - Groups by Experiment_ID
    - Pads with zeros or truncates to fixed length
    - Ensures uniform sequence length per experiment
    """

    def __init__(self):
        self.max_lengths = {
            "arc":  23832,
            "lin1": 27612,
            "lin2": 13582,
        }

    def _normalize_length(self, df: pd.DataFrame, target_len: int) -> pd.DataFrame:
        parts = []

        for exp_id, g in df.groupby("Experiment_ID", sort=False):
            n = len(g)

            if n > target_len:
                g = g.iloc[:target_len]

            elif n < target_len:
                missing = target_len - n
                pad = pd.DataFrame(
                    0.0,
                    index=range(missing),
                    columns=["X [mm]", "Y [mm]", "Z [mm]"]
                )
                pad.insert(0, "Experiment_ID", exp_id)
                g = pd.concat([g, pad], ignore_index=True)

            parts.append(g)

        return pd.concat(parts, ignore_index=True)

    def process_arc(self, df: pd.DataFrame) -> pd.DataFrame:
        return self._normalize_length(df, self.max_lengths["arc"])

    def process_lin1(self, df: pd.DataFrame) -> pd.DataFrame:
        return self._normalize_length(df, self.max_lengths["lin1"])

    def process_lin2(self, df: pd.DataFrame) -> pd.DataFrame:
        return self._normalize_length(df, self.max_lengths["lin2"])
