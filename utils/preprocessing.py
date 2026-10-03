"""Data loading, validation and feature preparation for NeuroSentinel.

Design notes
------------
* ``activity_id`` is an identifier and is never used as an ML feature.
* The supplied CSV is already standardized (z-scores). Instead of assuming
  this, :func:`check_standardization` verifies it from the data, and
  :func:`prepare_dataset` only applies a ``StandardScaler`` if the check fails.
* Problems found in the data are *reported*, never silently hidden.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

ID_COLUMN = "activity_id"

EXPECTED_FEATURES = [
    "duration_sec",
    "src_bytes",
    "dst_bytes",
    "packets",
    "connections",
    "request_frequency",
    "upload_ratio",
    "night_activity",
    "unique_destinations",
    "protocol_HTTP",
    "protocol_HTTPS",
    "protocol_TCP",
    "protocol_UDP",
]
PROTOCOL_PREFIX = "protocol_"
NIGHT_COLUMN = "night_activity"

# A column counts as "already standardized" when its mean/std are close to 0/1.
MEAN_TOLERANCE = 0.10
STD_TOLERANCE = 0.10


# --------------------------------------------------------------------------
# Data classes
# --------------------------------------------------------------------------
@dataclass
class ValidationReport:
    """Result of the data-quality checks run on the raw CSV."""

    n_rows: int
    n_columns: int
    missing_expected_columns: list[str]
    unexpected_columns: list[str]
    total_missing: int
    missing_by_column: pd.Series
    duplicate_rows: int
    duplicate_feature_rows: int
    duplicate_ids: int
    non_numeric_columns: list[str]
    infinite_values: int
    dtypes: pd.Series
    issues: list[str] = field(default_factory=list)

    @property
    def is_clean(self) -> bool:
        return not self.issues

    def summary_table(self) -> pd.DataFrame:
        """Compact table used by the dashboard."""
        rows = [
            ("Records", f"{self.n_rows:,}", True),
            ("Columns", f"{self.n_columns}", True),
            ("Expected columns present", "Yes" if not self.missing_expected_columns else
             f"No - missing {', '.join(self.missing_expected_columns)}", not self.missing_expected_columns),
            ("Missing values", f"{self.total_missing}", self.total_missing == 0),
            ("Duplicate rows", f"{self.duplicate_rows}", self.duplicate_rows == 0),
            ("Duplicate feature rows (ignoring ID)", f"{self.duplicate_feature_rows}",
             self.duplicate_feature_rows == 0),
            ("Duplicate activity IDs", f"{self.duplicate_ids}", self.duplicate_ids == 0),
            ("Non-numeric columns", ", ".join(self.non_numeric_columns) or "None",
             not self.non_numeric_columns),
            ("Infinite values", f"{self.infinite_values}", self.infinite_values == 0),
        ]
        return pd.DataFrame(rows, columns=["Check", "Result", "Passed"])


@dataclass
class PreparedData:
    """Everything the ML pipeline needs, produced from the raw CSV."""

    frame: pd.DataFrame            # cleaned rows: ID + feature columns
    feature_cols: list[str]
    X: np.ndarray                  # model matrix (float64, already standardized)
    standardization: pd.DataFrame  # per-feature mean / std check
    was_already_standardized: bool
    scaler_applied: bool
    protocol: pd.Series            # decoded protocol label per row
    night_flag: pd.Series | None   # decoded boolean night-activity flag
    notes: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------
# Loading and validation
# --------------------------------------------------------------------------
def load_dataset(path: str | Path) -> pd.DataFrame:
    """Read the CSV. Raises a clear error if the file is missing or empty."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at '{path}'.")
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError("The dataset file is empty.")
    return frame


def validate_dataset(frame: pd.DataFrame) -> ValidationReport:
    """Run data-quality checks and collect human-readable issues."""
    expected = [ID_COLUMN] + EXPECTED_FEATURES
    missing_expected = [c for c in expected if c not in frame.columns]
    unexpected = [c for c in frame.columns if c not in expected]

    numeric_cols = frame.select_dtypes(include="number").columns
    non_numeric = [c for c in frame.columns if c not in numeric_cols]

    infinite = int(np.isinf(frame[numeric_cols].to_numpy(dtype=float)).sum()) if len(numeric_cols) else 0
    feature_like = [c for c in frame.columns if c != ID_COLUMN]
    duplicate_ids = int(frame[ID_COLUMN].duplicated().sum()) if ID_COLUMN in frame.columns else 0

    report = ValidationReport(
        n_rows=len(frame),
        n_columns=frame.shape[1],
        missing_expected_columns=missing_expected,
        unexpected_columns=unexpected,
        total_missing=int(frame.isna().sum().sum()),
        missing_by_column=frame.isna().sum(),
        duplicate_rows=int(frame.duplicated().sum()),
        duplicate_feature_rows=int(frame.duplicated(subset=feature_like).sum()) if feature_like else 0,
        duplicate_ids=duplicate_ids,
        non_numeric_columns=non_numeric,
        infinite_values=infinite,
        dtypes=frame.dtypes.astype(str),
    )

    if missing_expected:
        report.issues.append(f"Missing expected columns: {', '.join(missing_expected)}.")
    if unexpected:
        report.issues.append(f"Unexpected extra columns found: {', '.join(unexpected)}.")
    if report.total_missing:
        report.issues.append(f"{report.total_missing} missing values detected (affected rows are dropped).")
    if report.duplicate_rows:
        report.issues.append(f"{report.duplicate_rows} fully duplicated rows detected.")
    elif report.duplicate_feature_rows:
        report.issues.append(f"{report.duplicate_feature_rows} rows share identical feature values.")
    if duplicate_ids:
        report.issues.append(f"{duplicate_ids} duplicated activity IDs (first occurrence is kept).")
    if non_numeric:
        report.issues.append(f"Non-numeric columns: {', '.join(non_numeric)}.")
    if infinite:
        report.issues.append(f"{infinite} infinite values detected (affected rows are dropped).")
    return report


# --------------------------------------------------------------------------
# Feature preparation
# --------------------------------------------------------------------------
def check_standardization(features: pd.DataFrame) -> pd.DataFrame:
    """Return per-feature mean/std and whether each column looks standardized."""
    table = pd.DataFrame({"mean": features.mean(), "std": features.std(ddof=0)})
    table["standardized"] = (table["mean"].abs() <= MEAN_TOLERANCE) & (
        (table["std"] - 1.0).abs() <= STD_TOLERANCE
    )
    return table


def _decode_binary(series: pd.Series) -> pd.Series | None:
    """Recover a boolean flag from a (possibly standardized) two-valued column.

    A standardized 0/1 column still has exactly two distinct values; the larger
    one corresponds to the original value 1.
    """
    if series.nunique() != 2:
        return None
    midpoint = (series.min() + series.max()) / 2.0
    return series > midpoint


def decode_protocol(frame: pd.DataFrame, protocol_cols: list[str]) -> pd.Series:
    """Recover a readable protocol label from the one-hot columns."""
    flags = {}
    for col in protocol_cols:
        decoded = _decode_binary(frame[col])
        if decoded is not None:
            flags[col.replace(PROTOCOL_PREFIX, "")] = decoded
    if not flags:
        return pd.Series("Unknown", index=frame.index, name="protocol")
    flag_frame = pd.DataFrame(flags)
    active_count = flag_frame.sum(axis=1)
    label = flag_frame.idxmax(axis=1).where(active_count == 1, "Unknown")
    return label.rename("protocol")


def prepare_dataset(raw: pd.DataFrame) -> PreparedData:
    """Clean the raw frame and build the model matrix.

    Steps: choose feature columns -> coerce to numeric -> drop unusable rows ->
    drop constant columns -> verify standardization -> (only if needed) scale.
    """
    notes: list[str] = []
    frame = raw.copy()

    # 1. Identifier handling -------------------------------------------------
    if ID_COLUMN not in frame.columns:
        frame.insert(0, ID_COLUMN, np.arange(1, len(frame) + 1))
        notes.append("No 'activity_id' column found; sequential IDs were generated.")

    # 2. Feature selection: every numeric column except the identifier -------
    candidate_cols = [c for c in frame.columns if c != ID_COLUMN]
    feature_cols = []
    for col in candidate_cols:
        converted = pd.to_numeric(frame[col], errors="coerce")
        if converted.notna().mean() >= 0.5:  # mostly numeric -> treat as feature
            frame[col] = converted
            feature_cols.append(col)
        else:
            notes.append(f"Column '{col}' is not numeric and was excluded from the features.")
    absent = [c for c in EXPECTED_FEATURES if c not in feature_cols]
    if absent:
        notes.append(f"Expected feature(s) not available and skipped: {', '.join(absent)}.")
    extra = [c for c in feature_cols if c not in EXPECTED_FEATURES]
    if extra:
        notes.append(f"Additional numeric column(s) used as features: {', '.join(extra)}.")

    # 3. Drop unusable rows (missing / infinite / duplicate ID) --------------
    frame[feature_cols] = frame[feature_cols].replace([np.inf, -np.inf], np.nan)
    before = len(frame)
    frame = frame.dropna(subset=feature_cols + [ID_COLUMN])
    if len(frame) < before:
        notes.append(f"Dropped {before - len(frame)} rows containing missing or infinite values.")
    before = len(frame)
    frame = frame.drop_duplicates(subset=ID_COLUMN, keep="first")
    if len(frame) < before:
        notes.append(f"Dropped {before - len(frame)} rows with duplicated activity IDs.")
    frame = frame.reset_index(drop=True)

    # 4. Drop constant columns (zero variance cannot be modelled) ------------
    constant = [c for c in feature_cols if frame[c].std(ddof=0) == 0]
    if constant:
        feature_cols = [c for c in feature_cols if c not in constant]
        notes.append(f"Dropped constant feature(s): {', '.join(constant)}.")
    if len(feature_cols) < 2 or len(frame) < 50:
        raise ValueError("Not enough valid data to train the models "
                         "(need at least 2 features and 50 rows).")

    # 5. Verify standardization; scale only if the data is NOT standardized --
    stats = check_standardization(frame[feature_cols])
    already_standardized = bool(stats["standardized"].all())
    if already_standardized:
        X = frame[feature_cols].to_numpy(dtype=float)
        notes.append("All features already have mean ~ 0 and std ~ 1, so no additional scaling was applied.")
    else:
        X = StandardScaler().fit_transform(frame[feature_cols])
        notes.append("Some features were not standardized, so a StandardScaler was applied.")

    # 6. Decoded helper columns for EDA only (never used as extra features) --
    protocol_cols = [c for c in feature_cols if c.startswith(PROTOCOL_PREFIX)]
    protocol = decode_protocol(frame, protocol_cols) if protocol_cols else \
        pd.Series("Unknown", index=frame.index, name="protocol")
    night_flag = _decode_binary(frame[NIGHT_COLUMN]) if NIGHT_COLUMN in feature_cols else None

    return PreparedData(
        frame=frame[[ID_COLUMN] + feature_cols],
        feature_cols=feature_cols,
        X=X,
        standardization=stats,
        was_already_standardized=already_standardized,
        scaler_applied=not already_standardized,
        protocol=protocol,
        night_flag=night_flag,
        notes=notes,
    )
