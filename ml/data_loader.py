import pandas as pd
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from ml.data_cleaner import DataCleaner
from ml.config import FEATURE_COLS, TARGET_COL, TEST_SIZE, RANDOM_STATE

@dataclass
class SplitData:
    X_train: pd.DataFrame
    X_test:  pd.DataFrame
    y_train: pd.Series
    y_test:  pd.Series

class DataLoader:
    def __init__(self, path: str) -> None:
        self.path = path
        self._raw = None
        self._clean = None

    def load(self) -> SplitData:
        self._raw   = self._read()
        self._clean = self._clean_data(self._raw)
        return self._split(self._clean)

    def clean_df(self):
        if self._clean is None:
            raise RuntimeError("Call load() before accessing clean_df.")
        return self._clean

    def raw_df(self):
        if self._raw is None:
            raise RuntimeError("Call load() before accessing raw_df.")
        return self._raw

    def _read(self) -> pd.DataFrame:
        df = pd.read_csv(self.path)
        self._validate_columns(df)
        return df

    def _validate_columns(self, df: pd.DataFrame) -> None:
        expected = set(FEATURE_COLS + [TARGET_COL])
        missing  = expected - set(df.columns)
        if missing:
            raise ValueError(f"CSV is missing columns: {missing}")

    def _clean_data(self, df):
        cleaner = DataCleaner(df)
        return cleaner.clean()

    def _split(self, df) -> SplitData:
        X = df[FEATURE_COLS]
        y = df[TARGET_COL]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=y,
        )

        return SplitData(
            X_train=X_train.reset_index(drop=True),
            X_test=X_test.reset_index(drop=True),
            y_train=y_train.reset_index(drop=True),
            y_test=y_test.reset_index(drop=True),
        )
