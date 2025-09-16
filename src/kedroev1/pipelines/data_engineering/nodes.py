# src/proyecto_ml/pipelines/data_engineering/nodes.py
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler

logger = logging.getLogger(__name__)


# =============== 1) INTEGRACIÓN ===============
def integrate_datasets(
    df_main: pd.DataFrame,
    df_ref: Optional[pd.DataFrame],
    df_supp: Optional[pd.DataFrame],
    params_join: Dict,
) -> pd.DataFrame:
    """
    Integra hasta 3 fuentes: main + (opcional) ref + (opcional) supp.
    Las uniones se controlan vía 'params_join'.
    """
    result = df_main.copy()
    if params_join.get("ref", {}).get("enabled", True) and df_ref is not None:
        j = params_join.get("ref", {})
        result = result.merge(
            df_ref,
            how=j.get("how", "left"),
            left_on=j.get("left_on"),
            right_on=j.get("right_on"),
            suffixes=("", j.get("suffix", "_ref")),
        )
        logger.info(f"Join ref: {j} -> shape {result.shape}")

    if params_join.get("supp", {}).get("enabled", True) and df_supp is not None:
        j = params_join.get("supp", {})
        result = result.merge(
            df_supp,
            how=j.get("how", "left"),
            left_on=j.get("left_on"),
            right_on=j.get("right_on"),
            suffixes=("", j.get("suffix", "_supp")),
        )
        logger.info(f"Join supp: {j} -> shape {result.shape}")

    return result


# =============== 2) SELECCIÓN & CAST ===============
def select_and_cast(
    df: pd.DataFrame,
    params_features: Dict,
) -> pd.DataFrame:
    """
    Selecciona columnas y castea tipos según parámetros.
    """
    df = df.copy()

    id_cols: List[str] = list(params_features.get("id_columns", []))
    num_cols: List[str] = list(params_features.get("numeric_features", []))
    cat_cols: List[str] = list(params_features.get("categorical_features", []))
    dt_cols:  List[str] = list(params_features.get("datetime_features", []))
    target: Optional[str] = params_features.get("target")

    selected = params_features.get("selected_columns")
    if selected:
        keep_cols = [c for c in selected if c in df.columns]
    else:
        keep_cols = [c for c in (id_cols + num_cols + cat_cols + dt_cols + ([target] if target else [])) if c in df.columns]

    df = df[keep_cols].copy()

    # Casts
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in cat_cols:
        if c in df.columns:
            # 'category' reduce memoria y ayuda en encoding
            df[c] = df[c].astype("category")
            # limpieza básica de strings
            df[c] = df[c].astype(str).str.strip().str.lower().replace({"nan": np.nan})
            df[c] = df[c].astype("category")
    for c in dt_cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")

    logger.info(f"select_and_cast keep={len(keep_cols)} -> shape {df.shape}")
    return df


# =============== 3) LIMPIEZA ===============
def clean_data(
    df: pd.DataFrame,
    params_clean: Dict,
    params_features: Dict,
) -> pd.DataFrame:
    """
    Limpia duplicados, missing values y outliers (IQR/Z-score) de forma parametrizable.
    """
    df = df.copy()

    # Duplicados
    if params_clean.get("drop_duplicates", True):
        before = len(df)
        df = df.drop_duplicates()
        logger.info(f"drop_duplicates: {before} -> {len(df)}")

    # Trimming genérico de texto
    if params_clean.get("string_trim", True):
        obj_cols = df.select_dtypes(include=["object", "string"]).columns
        for c in obj_cols:
            df[c] = df[c].astype(str).str.strip()
        logger.info(f"string_trim applied to {len(obj_cols)} columns")

    # Missing values
    num_cols = [c for c in params_features.get("numeric_features", []) if c in df.columns]
    cat_cols = [c for c in params_features.get("categorical_features", []) if c in df.columns]
    dt_cols  = [c for c in params_features.get("datetime_features", []) if c in df.columns]

    # Indicadores de missing (opcional)
    if params_clean.get("missing", {}).get("add_missing_indicators", True):
        for c in num_cols + cat_cols + dt_cols:
            if c in df.columns:
                df[f"{c}__missing"] = df[c].isna().astype(int)

    # Numéricos
    m_num = params_clean.get("missing", {}).get("numeric_strategy", "median")
    const_num = params_clean.get("missing", {}).get("numeric_constant", 0)
    if num_cols:
        if m_num == "median":
            df[num_cols] = df[num_cols].fillna(df[num_cols].median())
        elif m_num == "mean":
            df[num_cols] = df[num_cols].fillna(df[num_cols].mean())
        elif m_num == "constant":
            df[num_cols] = df[num_cols].fillna(const_num)

    # Categóricos
    m_cat = params_clean.get("missing", {}).get("categorical_strategy", "most_frequent")
    const_cat = params_clean.get("missing", {}).get("categorical_constant", "desconocido")
    if cat_cols:
        if m_cat == "most_frequent":
            modes = {c: df[c].mode(dropna=True)[0] if not df[c].mode(dropna=True).empty else const_cat for c in cat_cols}
            df[cat_cols] = df[cat_cols].fillna(value=modes)
        elif m_cat == "constant":
            df[cat_cols] = df[cat_cols].fillna(const_cat)

    # Datetime
    m_dt = params_clean.get("missing", {}).get("datetime_strategy", "constant")
    const_dt = params_clean.get("missing", {}).get("datetime_constant", "1970-01-01")
    if dt_cols:
        if m_dt == "constant":
            df[dt_cols] = df[dt_cols].fillna(pd.to_datetime(const_dt))

    # Outliers
    out_cfg = params_clean.get("outliers", {"method": "iqr", "factor": 1.5, "clip": True})
    method = out_cfg.get("method", "iqr")
    factor = float(out_cfg.get("factor", 1.5))
    clip   = bool(out_cfg.get("clip", True))

    if num_cols:
        if method == "iqr":
            for c in num_cols:
                q1, q3 = df[c].quantile(0.25), df[c].quantile(0.75)
                iqr = q3 - q1
                lo, hi = q1 - factor * iqr, q3 + factor * iqr
                if clip:
                    df[c] = df[c].clip(lo, hi)
        elif method == "zscore":
            for c in num_cols:
                m, s = df[c].mean(), df[c].std(ddof=0)
                if s == 0 or np.isnan(s):
                    continue
                z = (df[c] - m) / s
                mask = z.abs() > factor
                if clip:
                    df.loc[mask, c] = np.sign(df.loc[mask, c] - m) * factor * s + m

    logger.info("clean_data finished")
    return df


# =============== 4) FEATURE ENGINEERING ===============
def engineer_features(
    df: pd.DataFrame,
    params_fe: Dict,
    params_features: Dict,
) -> pd.DataFrame:
    """
    Crea features de fecha, ratios, interacciones y bins.
    """
    df = df.copy()
    dt_cols = params_fe.get("datetime_features_to_expand") or params_features.get("datetime_features", [])

    if params_fe.get("datetime_expand", True):
        for c in dt_cols:
            if c in df.columns:
                df[f"{c}__year"] = df[c].dt.year
                df[f"{c}__month"] = df[c].dt.month
                df[f"{c}__dayofweek"] = df[c].dt.dayof_week if hasattr(df[c].dt, "dayof_week") else df[c].dt.dayofweek
                df[f"{c}__is_month_start"] = df[c].dt.is_month_start.astype(int)
                df[f"{c}__is_month_end"] = df[c].dt.is_month_end.astype(int)

    # Ratios
    for r in params_fe.get("ratios", []):
        name, num, den = r["name"], r["numerator"], r["denominator"]
        if num in df.columns and den in df.columns:
            df[name] = df[num] / df[den].replace({0: np.nan})
            df[name] = df[name].fillna(0)

    # Interacciones (producto)
    for it in params_fe.get("interactions", []):
        name, a, b = it["name"], it["a"], it["b"]
        if a in df.columns and b in df.columns:
            df[name] = df[a] * df[b]

    # Bins (cuantiles u ordinales)
    for b in params_fe.get("bins", []):
        col, n_bins, encode = b["col"], int(b.get("n_bins", 5)), b.get("encode", "ordinal")
        if col in df.columns:
            q, labels = pd.qcut(df[col], q=n_bins, duplicates="drop", labels=False), None
            if encode == "ordinal":
                df[f"{col}__qbin"] = q
            elif encode == "onehot":
                oh = pd.get_dummies(q, prefix=f"{col}__qbin", dummy_na=False)
                df = pd.concat([df, oh], axis=1)

    # Drop opcional
    to_drop = params_fe.get("to_drop_after_fe", [])
    df = df.drop(columns=[c for c in to_drop if c in df.columns], errors="ignore")

    logger.info("engineer_features finished")
    return df


# =============== 5) SPLIT & PREPROCESS ===============
def split_data(
    df: pd.DataFrame,
    params_split: Dict,
    params_features: Dict,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Separa en train/test sin transformar.
    """
    target = params_features.get("target")
    stratify_col = params_split.get("stratify_by")

    stratify = df[stratify_col] if stratify_col and stratify_col in df.columns else None
    train_df, test_df = train_test_split(
        df,
        test_size=float(params_split.get("test_size", 0.2)),
        random_state=int(params_split.get("random_state", 42)),
        stratify=stratify,
        shuffle=True,
    )
    logger.info(f"split_data -> train {train_df.shape}, test {test_df.shape}")
    return train_df, test_df


def _build_preprocessor(
    numeric_features: List[str],
    categorical_features: List[str],
    scale_numeric: str = "standard",
    onehot_handle_unknown: str = "ignore",
    sparse: bool = False,
) -> ColumnTransformer:
    if scale_numeric == "standard":
        scaler = StandardScaler()
    elif scale_numeric == "minmax":
        scaler = MinMaxScaler()
    else:
        scaler = "passthrough"

    ohe = OneHotEncoder(handle_unknown=onehot_handle_unknown, sparse_output=sparse)

    transformers = []
    if numeric_features:
        transformers.append(("num", scaler, numeric_features))
    if categorical_features:
        transformers.append(("cat", ohe, categorical_features))

    pre = ColumnTransformer(transformers=transformers, remainder="drop", verbose_feature_names_out=False)
    return pre


def build_preprocessor(
    train_df: pd.DataFrame,
    params_features: Dict,
    params_preprocess: Dict,
) -> ColumnTransformer:
    """
    Ajusta el preprocesador SOLO con train para evitar fuga de info.
    """
    target = params_features.get("target")
    id_cols = params_features.get("id_columns", [])

    num_cols = [c for c in params_features.get("numeric_features", []) if c in train_df.columns]
    cat_cols = [c for c in params_features.get("categorical_features", []) if c in train_df.columns]

    cols_to_use = [c for c in (num_cols + cat_cols) if c not in (id_cols + [target])]
    num_cols = [c for c in cols_to_use if c in num_cols]
    cat_cols = [c for c in cols_to_use if c in cat_cols]

    pre = _build_preprocessor(
        numeric_features=num_cols,
        categorical_features=cat_cols,
        scale_numeric=params_preprocess.get("scale_numeric", "standard"),
        onehot_handle_unknown=params_preprocess.get("onehot_handle_unknown", "ignore"),
        sparse=bool(params_preprocess.get("sparse", False)),
    )

    X_train = train_df.drop(columns=[c for c in id_cols + [target] if c in train_df.columns], errors="ignore")
    pre.fit(X_train)
    logger.info("Preprocessor fitted on train_df")
    return pre


def transform_with_preprocessor(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    preprocessor: ColumnTransformer,
    params_features: Dict,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Aplica el preprocesador a train y test, devuelve X/y en DataFrames.
    """
    target = params_features.get("target")
    id_cols = params_features.get("id_columns", [])

    # X matrices
    X_train = train_df.drop(columns=[c for c in id_cols + [target] if c in train_df.columns], errors="ignore")
    X_test  = test_df.drop(columns=[c for c in id_cols + [target] if c in test_df.columns], errors="ignore")

    Xt_train = preprocessor.transform(X_train)
    Xt_test  = preprocessor.transform(X_test)

    # Nombres de columnas
    try:
        feature_names = preprocessor.get_feature_names_out()
    except Exception:
        # fallback si la versión no soporta get_feature_names_out en algún paso:
        feature_names = [f"f_{i}" for i in range(Xt_train.shape[1])]

    X_train_df = pd.DataFrame(Xt_train, columns=feature_names, index=train_df.index)
    X_test_df  = pd.DataFrame(Xt_test, columns=feature_names, index=test_df.index)

    # y
    y_train_df = pd.DataFrame(train_df[target]) if target in train_df.columns else pd.DataFrame()
    y_test_df  = pd.DataFrame(test_df[target]) if target in test_df.columns else pd.DataFrame()

    logger.info(f"transform_with_preprocessor -> X_train {X_train_df.shape}, X_test {X_test_df.shape}")
    return X_train_df, X_test_df, y_train_df, y_test_df
