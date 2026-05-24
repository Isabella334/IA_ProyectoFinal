import pandas as pd
import numpy as np

DATA_PATH = "./data/training_data.csv"
TARGET_COL = "action"
SEP = "─" * 55

# ── Helpers ───────────────────────────────────────────────────────────────────
def section(title: str) -> None:
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)


def load_raw(path: str) -> pd.DataFrame:
    """Load CSV replacing inf strings with actual np.inf."""
    df = pd.read_csv(path)
    df.replace([float("inf"), "inf", "Inf", "INF"], np.inf, inplace=True)
    df.replace([float("-inf"), "-inf"], -np.inf, inplace=True)
    return df


# ── Report ────────────────────────────────────────────────────────────────────
def report_shape(df: pd.DataFrame) -> None:
    section("1. SHAPE & DTYPES")
    print(f"  Rows    : {df.shape[0]}")
    print(f"  Columns : {df.shape[1]}")
    print()
    print(df.dtypes.to_string())


def report_missing(df: pd.DataFrame) -> None:
    section("2. MISSING VALUES (NaN / None)")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    summary = pd.DataFrame({"count": missing, "pct": missing_pct})
    has_missing = summary[summary["count"] > 0]

    if has_missing.empty:
        print("  ✓ No NaN/None values found")
    else:
        print(has_missing.to_string())


def report_inf(df: pd.DataFrame) -> None:
    section("3. INFINITE VALUES")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    inf_counts = {}

    for col in numeric_cols:
        n_pos = np.isinf(df[col]).sum()
        n_neg = np.isneginf(df[col]).sum()
        if n_pos > 0 or n_neg > 0:
            inf_counts[col] = {"inf": int(n_pos), "-inf": int(n_neg)}

    if not inf_counts:
        print("  ✓ No infinite values found")
    else:
        for col, counts in inf_counts.items():
            print(f"  {col}: +inf={counts['inf']}  -inf={counts['-inf']}")


def report_target(df: pd.DataFrame) -> None:
    section(f"4. TARGET DISTRIBUTION  →  '{TARGET_COL}'")
    if TARGET_COL not in df.columns:
        print(f"  ✗ Column '{TARGET_COL}' not found!")
        return

    counts = df[TARGET_COL].value_counts()
    pcts = (counts / len(df) * 100).round(2)

    for label in counts.index:
        bar = "█" * int(pcts[label] / 2)
        print(f"  {label:<15} {counts[label]:>4}  ({pcts[label]:>5.1f}%)  {bar}")

    if counts.min() / counts.max() < 0.5:
        print("\n  ⚠ Class imbalance detected — consider oversampling or class_weight")
    else:
        print("\n  ✓ Classes are reasonably balanced")


def report_stats(df: pd.DataFrame) -> None:
    section("5. FEATURE STATISTICS")
    numeric = df.select_dtypes(include=[np.number])
    # Describe ignores inf, so we compute on finite values only
    finite_df = numeric.replace([np.inf, -np.inf], np.nan)
    print(finite_df.describe().round(4).to_string())


def report_ranges(df: pd.DataFrame) -> None:
    section("6. FEATURE RANGES  (finite values only)")
    numeric = df.select_dtypes(include=[np.number])
    for col in numeric.columns:
        finite = numeric[col].replace([np.inf, -np.inf], np.nan).dropna()
        if finite.empty:
            print(f"  {col:<35} all inf")
            continue
        lo, hi = finite.min(), finite.max()
        has_inf = np.isinf(df[col]).any()
        flag = "  ⚠ has inf" if has_inf else ""
        print(f"  {col:<35} [{lo:.4f}, {hi:.4f}]{flag}")


def report_duplicates(df: pd.DataFrame) -> None:
    section("7. DUPLICATE ROWS")
    n_dupes = df.duplicated().sum()
    if n_dupes == 0:
        print("  ✓ No duplicate rows found")
    else:
        print(f"  ⚠ {n_dupes} duplicate row(s) found")
        print(df[df.duplicated(keep=False)].to_string())


def report_outliers(df: pd.DataFrame) -> None:
    section("8. OUTLIERS  (IQR method, finite values only)")
    numeric = df.select_dtypes(include=[np.number])
    found_any = False

    for col in numeric.columns:
        finite = numeric[col].replace([np.inf, -np.inf], np.nan).dropna()
        if finite.empty:
            continue
        Q1, Q3 = finite.quantile(0.25), finite.quantile(0.75)
        IQR = Q3 - Q1
        lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        outliers = finite[(finite < lower) | (finite > upper)]
        if not outliers.empty:
            found_any = True
            print(f"  {col}: {len(outliers)} outlier(s)  bounds=[{lower:.4f}, {upper:.4f}]")

    if not found_any:
        print("  ✓ No outliers detected")


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    print(SEP)
    print("  NPC DATA EXPLORATION REPORT")
    print(SEP)
    print(f"  Source: {DATA_PATH}")

    df = load_raw(DATA_PATH)

    report_shape(df)
    report_missing(df)
    report_inf(df)
    report_target(df)
    report_stats(df)
    report_ranges(df)
    report_duplicates(df)
    report_outliers(df)

    section("DONE")
    print("  Review warnings above before building DataLoader.\n")


main()
