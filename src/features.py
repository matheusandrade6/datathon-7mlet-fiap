"""
Feature engineering para o dataset bank-marketing (Etapa 2 do datathon).

Decisões (derivadas da EDA em notebooks/eda.ipynb):
- `duration` já foi removida na EDA por vazamento temporal; este módulo assume
  que a coluna pode estar presente (dado bruto) e a remove de forma defensiva.
- `pdays == 999` significa "nunca contatado antes" -> flag binária `foi_contatado_antes`,
  em vez de manter `pdays` como distância contínua.
- Colunas categóricas mantêm `unknown` como categoria própria (carrega sinal real).
- Variáveis macroeconômicas (`emp.var.rate`, `cons.price.idx`, `cons.conf.idx`,
  `euribor3m`, `nr.employed`) são tratadas como contexto de campanha, não atributo
  individual, mas entram como features numéricas normalizadas.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

LEAKAGE_COLS = ["duration"]

NUMERIC_COLS = [
    "age",
    "campaign",
    "previous",
    "emp.var.rate",
    "cons.price.idx",
    "cons.conf.idx",
    "euribor3m",
    "nr.employed",
]

CATEGORICAL_COLS = [
    "job",
    "marital",
    "education",
    "default",
    "housing",
    "loan",
    "contact",
    "month",
    "day_of_week",
    "poutcome",
]

TARGET_COL = "y"


def load_raw(path: str) -> pd.DataFrame:
    """Carrega o CSV bruto do bank-marketing (separador ';')."""
    return pd.read_csv(path, sep=";")


def add_contact_flag(df: pd.DataFrame) -> pd.DataFrame:
    """Deriva `foi_contatado_antes` a partir de `pdays` (999 = nunca contatado)."""
    df = df.copy()
    df["foi_contatado_antes"] = (df["pdays"] != 999).astype(int)
    df = df.drop(columns=["pdays"])
    return df


def drop_leakage_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove colunas com vazamento temporal (ex.: duration), se presentes."""
    cols_present = [c for c in LEAKAGE_COLS if c in df.columns]
    return df.drop(columns=cols_present)


def encode_target(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if TARGET_COL in df.columns:
        df[TARGET_COL] = (df[TARGET_COL] == "yes").astype(int)
    return df


def build_features(
    df: pd.DataFrame,
    scaler: StandardScaler | None = None,
    fit_scaler: bool = True,
) -> tuple[pd.DataFrame, StandardScaler]:
    """
    Pipeline completo de feature engineering:
      1. remove vazamento temporal (duration)
      2. deriva flag foi_contatado_antes a partir de pdays
      3. one-hot encoding das categóricas (unknown mantido como categoria)
      4. normalização (z-score) das numéricas via StandardScaler

    Retorna (df_features, scaler) para permitir reaproveitar o mesmo scaler
    no split de simulação online e, futuramente, no serviço de recomendação.
    """
    df = drop_leakage_columns(df)
    df = add_contact_flag(df)
    df = encode_target(df)

    df_encoded = pd.get_dummies(df, columns=CATEGORICAL_COLS, prefix=CATEGORICAL_COLS)

    if scaler is None:
        scaler = StandardScaler()

    if fit_scaler:
        df_encoded[NUMERIC_COLS] = scaler.fit_transform(df_encoded[NUMERIC_COLS])
    else:
        df_encoded[NUMERIC_COLS] = scaler.transform(df_encoded[NUMERIC_COLS])

    return df_encoded, scaler


def feature_columns(df_encoded: pd.DataFrame) -> list[str]:
    """Lista de colunas de feature (exclui target, flags de braço/arm e ids auxiliares)."""
    exclude = {TARGET_COL, "foi_contatado_antes_raw"}
    return [c for c in df_encoded.columns if c not in exclude]
