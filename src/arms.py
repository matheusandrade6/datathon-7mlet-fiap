"""
Definição e simulação dos braços (arms) do bandit — Task 2.1.2 (Epic 2).

Contexto (ver Seção 1.5 do PLANO_DATATHON.md):
O dataset `bank-marketing` só tem UM produto real (depósito a prazo) e não
múltiplas ofertas. Para satisfazer o requisito de "braços" do bandit, simulamos
4 braços = variações de MENSAGEM/CANAL para esse mesmo produto. Cada braço tem
uma probabilidade-base de conversão derivada da taxa real do dataset
(≈ 11,27%), ajustada por segmentos que a EDA (notebooks/eda.ipynb) já mostrou
terem sinal real de conversão (poutcome, job, month, contato prévio), e com
ruído controlado somado ao final.

Isso é uma limitação assumida e documentada (não um dado real apresentado como
tal): os 4 braços não existiram de fato na campanha original, mas a
probabilidade de conversão simulada por braço é ancorada em padrões reais
observados na base, não inventada arbitrariamente.

Braços definidos:
  arm 0 — "Oferta padrão"                 (mensagem genérica, sem personalização;
                                            referência = comportamento histórico médio)
  arm 1 — "Oferta com apelo a benefício"   (mensagem de valor/taxa; melhor para
                                            perfis mais analíticos/qualificados)
  arm 2 — "Oferta reforçada para quem já converteu" (mensagem personalizada para
                                            quem tem histórico de sucesso em campanha
                                            anterior — maior ganho, mas segmento pequeno)
  arm 3 — "Oferta introdutória (primeiro contato)"  (mensagem mais leve para quem
                                            nunca foi contatado antes / estudantes /
                                            aposentados / meses de maior conversão)

Regra de probabilidade-base + ruído:
    p_arm(cliente) = clip(
        p_global * multiplicador_base[arm] * multiplicador_segmento(cliente, arm)
        + ruido_gaussiano(0, NOISE_STD),
        P_MIN, P_MAX,
    )

`p_global` é a taxa de conversão real do dataset (recalculada em runtime, não
hardcoded). Os multiplicadores estão documentados em ARM_DEFINITIONS abaixo.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SEED = 42
NOISE_STD = 0.02
P_MIN, P_MAX = 0.01, 0.95

ARM_DEFINITIONS = {
    0: {
        "nome": "Oferta padrão",
        "descricao": "Mensagem genérica via canal telefônico, sem personalização. "
        "Serve como braço de referência (comportamento histórico médio).",
        "multiplicador_base": 1.00,
    },
    1: {
        "nome": "Oferta com apelo a benefício/taxa",
        "descricao": "Mensagem com destaque para taxa/benefício financeiro do produto. "
        "Boost simulado para perfis com job em admin./management/technician "
        "ou educação superior, segmentos com maior engajamento observado na EDA.",
        "multiplicador_base": 1.10,
    },
    2: {
        "nome": "Oferta reforçada para quem já converteu",
        "descricao": "Mensagem personalizada para clientes com poutcome == 'success' "
        "(já converteram em campanha anterior) ou previous > 0. A EDA mostrou "
        "que poutcome='success' tem taxa de conversão muito acima da média — "
        "maior ganho simulado, mas segmento pequeno.",
        "multiplicador_base": 1.00,
    },
    3: {
        "nome": "Oferta introdutória (primeiro contato)",
        "descricao": "Mensagem mais leve para quem nunca foi contatado antes "
        "(pdays == 999, ~96% da base), estudantes/aposentados e meses de maior "
        "conversão histórica (mar, dec, sep, oct), conforme sinal encontrado na EDA.",
        "multiplicador_base": 0.85,
    },
}

_HIGH_EDU = {"university.degree", "professional.course"}
_HIGH_JOB_ARM1 = {"admin.", "management", "technician"}
_HIGH_JOB_ARM3 = {"student", "retired"}
_HIGH_MONTHS_ARM3 = {"mar", "dec", "sep", "oct"}


def _segment_multiplier(row: pd.Series, arm: int) -> float:
    """Multiplicador de segmento (contexto do cliente) por braço, ancorado em sinais reais da EDA."""
    mult = 1.0

    if arm == 1:
        if row.get("job") in _HIGH_JOB_ARM1 or row.get("education") in _HIGH_EDU:
            mult *= 1.15

    elif arm == 2:
        if row.get("poutcome") == "success":
            mult *= 1.35
        elif row.get("previous", 0) and row.get("previous", 0) > 0:
            mult *= 1.10

    elif arm == 3:
        if row.get("pdays") == 999:
            mult *= 1.20
        if row.get("job") in _HIGH_JOB_ARM3:
            mult *= 1.15
        if row.get("month") in _HIGH_MONTHS_ARM3:
            mult *= 1.10

    return mult


def simulate_arm_probabilities(
    df_raw: pd.DataFrame,
    seed: int = SEED,
    noise_std: float = NOISE_STD,
) -> pd.DataFrame:
    """
    Gera, para cada cliente (linha de df_raw) e cada braço definido em
    ARM_DEFINITIONS, a probabilidade-base de conversão simulada.

    df_raw precisa conter as colunas originais: job, education, poutcome,
    previous, pdays, month, y.

    Retorna um DataFrame com colunas p_arm_0..p_arm_{n-1}, alinhado ao índice
    de df_raw (reprodutível: mesmo seed -> mesmos valores).
    """
    rng = np.random.default_rng(seed)
    p_global = (df_raw["y"] == "yes").mean()

    n = len(df_raw)
    n_arms = len(ARM_DEFINITIONS)
    probs = np.zeros((n, n_arms), dtype=float)

    for arm, spec in ARM_DEFINITIONS.items():
        base_mult = spec["multiplicador_base"]
        seg_mult = df_raw.apply(lambda row: _segment_multiplier(row, arm), axis=1).to_numpy()
        noise = rng.normal(loc=0.0, scale=noise_std, size=n)
        raw_p = p_global * base_mult * seg_mult + noise
        probs[:, arm] = np.clip(raw_p, P_MIN, P_MAX)

    cols = [f"p_arm_{arm}" for arm in sorted(ARM_DEFINITIONS)]
    return pd.DataFrame(probs, columns=cols, index=df_raw.index)


def arms_metadata() -> list[dict]:
    """Metadata serializável (para salvar em JSON/README) dos braços definidos."""
    return [
        {"arm": arm, **{k: v for k, v in spec.items()}}
        for arm, spec in sorted(ARM_DEFINITIONS.items())
    ]
