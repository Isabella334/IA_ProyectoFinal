import pandas as pd
import numpy as np
from ml.config import FEATURE_COLS, TARGET_COL

class DataCleaner:
    def __init__(self, df: pd.DataFrame) -> None:
        self._df = df.copy()
        self._log: list[str] = []

    def clean(self):
        self._drop_duplicates()
        self._replace_inf()
        self._select_columns()
        self._print_log()
        return self._df

    def _drop_duplicates(self) -> None:
        before = len(self._df)
        self._df.drop_duplicates(inplace=True)
        dropped = before - len(self._df)
        self._log.append(f"Duplicates dropped: {dropped} rows")

    def _select_columns(self) -> None:
        keep = FEATURE_COLS + [TARGET_COL]
        self._df = self._df[keep]
        self._log.append(f"Columns kept: {keep}")

    def _replace_inf(self):
        finite_max = self._df["player_distance"].replace([np.inf, -np.inf], np.nan).max()
        self._df["player_distance"] = self._df["player_distance"].replace([np.inf, -np.inf], finite_max)
        self._log.append(f"Inf in player_distance replaced with: {finite_max:.4f}")

    def _print_log(self) -> None:
        for entry in self._log:
            print(f"{entry}")
