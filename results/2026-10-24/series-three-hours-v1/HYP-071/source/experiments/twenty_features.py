"""Train-only DataFrame transformations for HYP-022..032.

No data is read and no model is trained on import. Domain names and units below
were checked in data/{regression,multiclass}/data_dictionary.csv and task.md.
The caller removes identifiers/targets and controls downstream encoding/scaling.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import QuantileTransformer
from sklearn.utils.validation import check_is_fitted


MONTHS = "january february march april may june july august september october november december".split()
TCP_FIELDS = ("swin", "dwin", "stcpb", "dtcpb", "tcprtt", "synack", "ackdat")
WIND_FIELDS = ("direction_max_wind_speed", "direction_peak_wind_speed")
GROUP_FIELDS = ("State_Factor", "facility_type")


def _frame(X):
    if not isinstance(X, pd.DataFrame) or not X.columns.is_unique:
        raise ValueError("X must be a DataFrame with unique columns")
    return X


def _numeric(X, column):
    return pd.to_numeric(X[column], errors="coerce").astype(float).replace([np.inf, -np.inf], np.nan)


def _keys(X, columns):
    # Tagged, length-prefixed strings do not conflate missing/literal sentinels.
    result = pd.Series("", index=X.index, dtype=object)
    for column in columns:
        value = ("V:" + X[column].astype("string")).fillna("M:").astype(str)
        result = result + value.str.len().astype(str) + ":" + value
    return result


class _FrameTransformer(TransformerMixin, BaseEstimator):
    def _start(self, X):
        _frame(X)
        if not len(X):
            raise ValueError("fit needs at least one training row")
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = len(X.columns)
        self.numeric_ = X.select_dtypes(include="number").columns.tolist()
        self.output_columns_ = list(X.columns)
        self.audit_ = {"n_fit": len(X), "target_used": False, "row_deletion": False}

    def _add(self, names):
        for name in names:
            if name in self.output_columns_:
                raise ValueError("Generated feature name collides with an input: " + name)
            self.output_columns_.append(name)

    def _copy(self, X):
        check_is_fitted(self, "feature_names_in_")
        _frame(X)
        if set(X.columns) != set(self.feature_names_in_):
            raise ValueError("transform schema differs from fitted schema")
        return X.loc[:, self.feature_names_in_].copy()

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "output_columns_")
        return np.asarray(self.output_columns_, dtype=object)


class Winsorizer(_FrameTransformer):
    """H22: clip numeric values to fitted quantiles without dropping rows."""
    def __init__(self, lower=.005, upper=.995):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        self._start(X)
        if not 0 <= self.lower < self.upper <= 1:
            raise ValueError("quantile bounds must satisfy 0 <= lower < upper <= 1")
        self.bounds_ = {}
        for c in self.numeric_:
            clean = _numeric(X, c).dropna()
            self.bounds_[c] = tuple(clean.quantile([self.lower, self.upper])) if len(clean) else (None, None)
        self.audit_["quantiles"] = [float(self.lower), float(self.upper)]
        return self

    def transform(self, X):
        out = self._copy(X)
        for c, (lower, upper) in self.bounds_.items():
            out[c] = _numeric(out, c).clip(lower=lower, upper=upper)
        return out


class IterativeNumericImputer(_FrameTransformer):
    """H23: BayesianRidge iterative imputation, never receiving y.

    Bounds concern imputed values only: area/year and degree-day totals are
    nonnegative; ENERGY STAR is 0..100; checked wind angles are 0..360.
    Empty training columns use zero while retaining their original positions.
    """
    def __init__(self, max_iter=5, n_nearest_features=10, random_state=42):
        self.max_iter = max_iter
        self.n_nearest_features = n_nearest_features
        self.random_state = random_state

    def fit(self, X, y=None):
        self._start(X)
        self.active_ = [c for c in self.numeric_ if _numeric(X, c).notna().any()]
        self.empty_ = [c for c in self.numeric_ if c not in self.active_]
        self.imputer_ = None
        if self.active_:
            lower, upper = [], []
            nonnegative = {"floor_area", "year_built", "energy_star_rating", "heating_degree_days_c", "cooling_degree_days_c", *WIND_FIELDS}
            for c in self.active_:
                lower.append(0. if c in nonnegative else -np.inf)
                upper.append(100. if c == "energy_star_rating" else 360. if c in WIND_FIELDS else np.inf)
            self.imputer_ = IterativeImputer(max_iter=self.max_iter, n_nearest_features=self.n_nearest_features,
                                            sample_posterior=False, random_state=self.random_state,
                                            min_value=np.asarray(lower), max_value=np.asarray(upper))
            self.imputer_.fit(pd.DataFrame({c: _numeric(X, c).to_numpy() for c in self.active_}))
        self.audit_["numeric_columns"] = list(self.numeric_)
        self.audit_["empty_numeric_columns"] = list(self.empty_)
        return self

    def transform(self, X):
        out = self._copy(X)
        if self.active_:
            numeric = pd.DataFrame({c: _numeric(out, c).to_numpy() for c in self.active_})
            values = self.imputer_.transform(numeric)
            for j, c in enumerate(self.active_):
                out[c] = values[:, j]
        for c in self.empty_:
            out[c] = _numeric(out, c).fillna(0.)
        return out


class _GroupStatistics(_FrameTransformer):
    def _fit_stats(self, X, columns, min_count):
        self.levels_ = [list(GROUP_FIELDS), ["facility_type"]]
        self.levels_ = [level for level in self.levels_ if all(c in X for c in level)]
        self.statistics_ = {}
        for c in columns:
            values = _numeric(X, c)
            finite = values.dropna()
            global_median = float(finite.median()) if len(finite) else 0.
            global_iqr = float(finite.quantile(.75) - finite.quantile(.25)) if len(finite) else 0.
            maps = []
            for level in self.levels_:
                frame = pd.DataFrame({"key": _keys(X, level).to_numpy(), "value": values.to_numpy()})
                grouped = frame.groupby("key", sort=False)["value"]
                counts = grouped.count()
                median = grouped.median()
                iqr = grouped.quantile(.75) - grouped.quantile(.25)
                eligible = counts >= min_count
                maps.append((median[eligible].to_dict(), iqr[eligible].to_dict()))
            self.statistics_[c] = (global_median, global_iqr, maps)

    def _lookup(self, X, c):
        median, iqr, maps = self.statistics_[c]
        medians = np.full(len(X), median, dtype=float)
        iqrs = np.full(len(X), iqr, dtype=float)
        for level, (median_map, iqr_map) in reversed(list(zip(self.levels_, maps))):
            keys = _keys(X, level)
            m = keys.map(median_map).to_numpy(dtype=float)
            q = keys.map(iqr_map).to_numpy(dtype=float)
            good = np.isfinite(m)
            medians[good] = m[good]
            iqrs[good] = q[good]
        return medians, iqrs


class GroupMedianImputer(_GroupStatistics):
    """H24: >=100 available values; state/facility -> facility -> global."""
    def __init__(self, min_count=100):
        self.min_count = min_count

    def fit(self, X, y=None):
        self._start(X)
        self._fit_stats(X, self.numeric_, self.min_count)
        self._add([c + "__missing" for c in self.numeric_])
        self.audit_["min_count"] = int(self.min_count)
        return self

    def transform(self, X):
        out = self._copy(X)
        for c in self.numeric_:
            values = _numeric(out, c).to_numpy()
            missing = ~np.isfinite(values)
            medians, _ = self._lookup(out, c)
            values[missing] = medians[missing]
            out[c] = values
            out[c + "__missing"] = missing.astype(np.uint8)
        return out


def _quantile(values):
    values = np.asarray(values, dtype=float).reshape(-1, 1)
    return QuantileTransformer(n_quantiles=min(512, len(values)), output_distribution="normal",
                               subsample=None, random_state=42).fit(values)


class ConditionalGaussian(_FrameTransformer):
    """H25: per-protocol quantile maps with a train-global fallback."""
    def __init__(self, min_group=500, min_unique=20):
        self.min_group = min_group
        self.min_unique = min_unique

    def fit(self, X, y=None):
        self._start(X)
        self.transforms_ = {}
        self.protocol_ = "proto" if "proto" in X else None
        protocol_indices = {}
        if self.protocol_:
            keys = _keys(X, [self.protocol_])
            protocol_indices = {key: rows for key, rows in keys.groupby(keys, sort=False).indices.items()
                                if len(rows) >= self.min_group}
        for c in self.numeric_:
            clean = _numeric(X, c)
            finite = clean.dropna()
            if finite.nunique() < self.min_unique or finite.value_counts(normalize=True).max() > .8:
                continue
            groups = {}
            if self.protocol_:
                for key, rows in protocol_indices.items():
                    values = clean.iloc[rows].dropna()
                    if len(values) >= self.min_group and values.nunique() >= self.min_unique:
                        groups[key] = _quantile(values)
            self.transforms_[c] = (_quantile(finite), float(finite.median()), groups)
        self.audit_["transformed_columns"] = list(self.transforms_)
        self.audit_["protocol_group_counts"] = {c: len(v[2]) for c, v in self.transforms_.items()}
        return self

    def transform(self, X):
        out = self._copy(X)
        keys = _keys(out, [self.protocol_]) if self.protocol_ else None
        for c, (global_transform, median, groups) in self.transforms_.items():
            values = _numeric(out, c).fillna(median).to_numpy().reshape(-1, 1)
            transformed = global_transform.transform(values).ravel()
            for key, transform in groups.items():
                mask = (keys == key).to_numpy()
                if mask.any():
                    transformed[mask] = transform.transform(values[mask]).ravel()
            out[c] = transformed
        return out


class ZeroAtomGaussian(_FrameTransformer):
    """H26: zero indicator plus a Gaussian map fitted on positive values."""
    def __init__(self, lower_fraction=.2, upper_fraction=.8, min_positive_unique=20):
        self.lower_fraction = lower_fraction
        self.upper_fraction = upper_fraction
        self.min_positive_unique = min_positive_unique

    def fit(self, X, y=None):
        self._start(X)
        self.transforms_ = {}
        for c in self.numeric_:
            clean = _numeric(X, c).dropna()
            positive = clean[clean > 0]
            if (len(clean) and clean.min() >= 0 and self.lower_fraction <= (clean == 0).mean() <= self.upper_fraction
                    and positive.nunique() >= self.min_positive_unique):
                self.transforms_[c] = _quantile(positive)
        self._add([c + "__is_zero" for c in self.transforms_])
        self.audit_["transformed_columns"] = list(self.transforms_)
        return self

    def transform(self, X):
        out = self._copy(X)
        for c, transform in self.transforms_.items():
            values = _numeric(out, c).to_numpy()
            mapped = np.full(len(values), np.nan)
            positive = values > 0
            if positive.any():
                mapped[positive] = transform.transform(values[positive].reshape(-1, 1)).ravel()
            mapped[values == 0] = 0.
            out[c] = mapped
            out[c + "__is_zero"] = (values == 0).astype(np.uint8)
        return out


class TCPApplicability(_FrameTransformer):
    """H27: dictionary-confirmed TCP fields only; unknown protocols non-TCP.

    An inapplicable value is NaN before downstream imputation, accompanied by
    an explicit applicability indicator. Observed TCP zeros are retained.
    """
    def fit(self, X, y=None):
        self._start(X)
        self.fields_ = [c for c in TCP_FIELDS if c in X] if "proto" in X else []
        self._add([c + "__applicable" for c in self.fields_])
        self.audit_["validated_fields"] = list(self.fields_)
        self.audit_["semantic_source"] = "data/multiclass/data_dictionary.csv + task.md"
        return self

    def transform(self, X):
        out = self._copy(X)
        if self.fields_:
            applicable = out["proto"].astype("string").str.strip().str.lower().eq("tcp").fillna(False).to_numpy(bool)
            for c in self.fields_:
                values = _numeric(out, c).to_numpy()
                values[~applicable] = np.nan
                out[c] = values
                out[c + "__applicable"] = applicable.astype(np.uint8)
        return out


class WindCyclic(_FrameTransformer):
    """H29: degrees mapped to circular sine/cosine while retaining originals."""
    def fit(self, X, y=None):
        self._start(X)
        self.fields_ = [c for c in WIND_FIELDS if c in X]
        self._add([c + suffix for c in self.fields_ for suffix in ("__sin", "__cos")])
        return self

    def transform(self, X):
        out = self._copy(X)
        for c in self.fields_:
            angle = np.deg2rad(_numeric(out, c).to_numpy())
            out[c + "__sin"] = np.sin(angle)
            out[c + "__cos"] = np.cos(angle)
        return out


class MonthlyHarmonics(_FrameTransformer):
    """H30: first two Fourier harmonics, January at angle zero, scale 2/12."""
    def fit(self, X, y=None):
        self._start(X)
        self.monthly_ = []
        for suffix in ("_avg_temp_c", "_avg_temp"):
            names = [m + suffix for m in MONTHS]
            if all(c in X for c in names):
                self.monthly_ = names
                break
        self.medians_ = {c: float(_numeric(X, c).median()) if _numeric(X, c).notna().any() else 0. for c in self.monthly_}
        self.names_ = [f"monthly_temp__{kind}{h}" for h in (1, 2) for kind in ("sin", "cos")] if self.monthly_ else []
        self._add(self.names_)
        self.audit_["monthly_columns"] = list(self.monthly_)
        return self

    def transform(self, X):
        out = self._copy(X)
        if self.monthly_:
            matrix = np.column_stack([_numeric(out, c).fillna(self.medians_[c]).to_numpy() for c in self.monthly_])
            angle = 2 * np.pi * np.arange(12) / 12
            for h in (1, 2):
                out[f"monthly_temp__sin{h}"] = matrix @ np.sin(h * angle) / 6
                out[f"monthly_temp__cos{h}"] = matrix @ np.cos(h * angle) / 6
        return out


class GroupRelative(_GroupStatistics):
    """H31: feature deviations from comparable training buildings."""
    def __init__(self, min_count=100, eps=1e-6):
        self.min_count = min_count
        self.eps = eps

    def fit(self, X, y=None):
        self._start(X)
        self.fields_ = [c for c in ("floor_area", "energy_star_rating") if c in X]
        self._fit_stats(X, self.fields_, self.min_count)
        self._add([c + "__group_relative" for c in self.fields_])
        return self

    def transform(self, X):
        out = self._copy(X)
        for c in self.fields_:
            median, iqr = self._lookup(out, c)
            out[c + "__group_relative"] = (_numeric(out, c).to_numpy() - median) / np.maximum(iqr, self.eps)
        return out


class WeatherAreaInteractions(_FrameTransformer):
    """H32: degree days times log(area), and training-class interactions."""
    def fit(self, X, y=None):
        self._start(X)
        self.fields_ = [c for c in ("heating_degree_days_c", "cooling_degree_days_c") if c in X] if "floor_area" in X else []
        self.classes_ = sorted(X["building_class"].dropna().astype(str).unique()) if "building_class" in X else []
        # Stable integer suffixes avoid names derived from arbitrary category text.
        self.names_ = [c + "__log_area" for c in self.fields_]
        self.names_ += [c + f"__log_area__class_{j}" for c in self.fields_ for j in range(len(self.classes_))]
        self._add(self.names_)
        self.audit_["building_classes"] = list(self.classes_)
        return self

    def transform(self, X):
        out = self._copy(X)
        if self.fields_:
            area = _numeric(out, "floor_area").to_numpy()
            log_area = np.log1p(np.where(area >= 0, area, np.nan))
            for c in self.fields_:
                interaction = _numeric(out, c).to_numpy() * log_area
                out[c + "__log_area"] = interaction
                for j, label in enumerate(self.classes_):
                    mask = out["building_class"].astype("string").eq(label).fillna(False).to_numpy(bool)
                    out[c + f"__log_area__class_{j}"] = np.where(mask, interaction, 0.)
        return out.loc[:, self.output_columns_]


def make_features(number: int):
    """Return a fresh unfitted transform; selection/model hypotheses live elsewhere."""
    classes = {22: Winsorizer, 23: IterativeNumericImputer, 24: GroupMedianImputer,
               25: ConditionalGaussian, 26: ZeroAtomGaussian, 27: TCPApplicability,
               29: WindCyclic, 30: MonthlyHarmonics, 31: GroupRelative, 32: WeatherAreaInteractions}
    if number not in classes:
        raise ValueError("No standalone feature transformer for HYP-" + str(number))
    return classes[number]()
