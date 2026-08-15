"""
Treino/serialização das 3 políticas de recomendação — Epic 5, Task 5.1.1.

Este script NÃO re-treina do zero nem reformula nada: ele reproduz, byte a
byte, o mesmo pipeline já validado nas Etapas 3 e 4 (`notebooks/baseline_bandit.ipynb`
e `notebooks/avaliacao_golden_set.ipynb`) — mesmo split (`SPLIT_SEED=42`,
`ONLINE_SIM_SIZE=0.30`, estratificado por `y`), mesmos priors do Thompson
Sampling (`PRIOR_STRENGTH=500`) e mesma simulação online (`seed=123`) — para
chegar exatamente ao mesmo estado final treinado das 3 políticas
(`baseline`, `epsilon_greedy`, `thompson_sampling`) que a Etapa 4 avaliou.

O motivo de existir como script (em vez de só reaproveitar o notebook) é
permitir que o serviço FastAPI (Epic 5, Task 5.1.2) carregue as políticas já
treinadas sem precisar rodar um notebook Jupyter em produção: `python -m
src.train_policies` gera os artefatos serializados em `models/` uma única
vez; o serviço só faz `joblib.load` no startup.

Saídas (em `models/`):
  - baseline.joblib            -> objeto DeterministicBaseline treinado
  - epsilon_greedy.joblib      -> objeto EpsilonGreedyBandit pós-simulação online
  - thompson_sampling.joblib   -> objeto ThompsonSamplingContextual pós-simulação online
  - training_metadata.json     -> metadados do treino (seeds, tamanho da simulação,
                                   conferência de reprodutibilidade vs. epic3_resultados.json)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.features import load_raw
from src.bandit import (
    DeterministicBaseline,
    EpsilonGreedyBandit,
    ThompsonSamplingContextual,
    run_online_simulation,
    summarize_policies,
    add_segment_column,
    N_ARMS,
    EPSILON,
    PRIOR_STRENGTH,
)

ROOT = Path(__file__).resolve().parent.parent
RAW_PATH = ROOT / "data" / "raw" / "bank-additional-full.csv"
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"

SPLIT_SEED = 42
ONLINE_SIM_SIZE = 0.30
ONLINE_SIM_SEED = 123
EPSILON_GREEDY_SEED = 42


def train_and_serialize() -> dict:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    df_raw = load_raw(str(RAW_PATH))
    y_raw = (df_raw["y"] == "yes").astype(int)
    p_global = float(y_raw.mean())

    idx_train, idx_online = train_test_split(
        df_raw.index, test_size=ONLINE_SIM_SIZE, stratify=y_raw, random_state=SPLIT_SEED
    )
    raw_train = df_raw.loc[idx_train].reset_index(drop=True)
    raw_online = df_raw.loc[idx_online].reset_index(drop=True)

    arm_probabilities_train = pd.read_csv(PROCESSED_DIR / "arm_probabilities_train.csv")
    arm_probabilities_online = pd.read_csv(PROCESSED_DIR / "arm_probabilities_online.csv")

    baseline = DeterministicBaseline.fit(arm_probabilities_train)
    thompson = ThompsonSamplingContextual.fit_priors(
        raw_train, arm_probabilities_train, prior_strength=PRIOR_STRENGTH, seed=EPSILON_GREEDY_SEED
    )
    epsilon_greedy = EpsilonGreedyBandit(n_arms=N_ARMS, epsilon=EPSILON, seed=EPSILON_GREEDY_SEED)

    sim = run_online_simulation(
        raw_online, arm_probabilities_online, baseline, epsilon_greedy, thompson, seed=ONLINE_SIM_SEED
    )
    resumo_geral = summarize_policies(sim)

    # Conferência de reprodutibilidade: o resultado deve bater com o que a
    # Etapa 3/4 já salvaram em epic3_resultados.json (mesma lógica de sanity
    # check feita em notebooks/avaliacao_golden_set.ipynb, célula 3).
    with open(PROCESSED_DIR / "epic3_resultados.json", encoding="utf-8") as f:
        epic3 = json.load(f)
    check = pd.DataFrame(epic3["simulacao_online"]["resumo_final_realizado"])
    # epic3_resultados.json foi salvo com valores arredondados a 5 casas
    # decimais (ver notebooks/baseline_bandit.ipynb); arredondamos os dois
    # lados antes de comparar para não gerar falso-negativo por causa só do
    # arredondamento já aplicado no arquivo de referência.
    reproduziu = bool(
        np.allclose(
            resumo_geral.set_index("policy")[["conversoes", "taxa_conversao"]].round(5).to_numpy(),
            check.set_index("policy")[["conversoes", "taxa_conversao"]].round(5).to_numpy(),
        )
    )
    if not reproduziu:
        raise RuntimeError(
            "O treino reproduzido neste script NÃO bateu com data/processed/epic3_resultados.json — "
            "verifique se os seeds/params (SPLIT_SEED, ONLINE_SIM_SEED, PRIOR_STRENGTH, EPSILON) "
            "ainda estão alinhados com notebooks/baseline_bandit.ipynb antes de servir o modelo."
        )

    import joblib

    joblib.dump(baseline, MODELS_DIR / "baseline.joblib")
    joblib.dump(epsilon_greedy, MODELS_DIR / "epsilon_greedy.joblib")
    joblib.dump(thompson, MODELS_DIR / "thompson_sampling.joblib")

    metadata = {
        "split_seed": SPLIT_SEED,
        "online_sim_size": ONLINE_SIM_SIZE,
        "online_sim_seed": ONLINE_SIM_SEED,
        "epsilon_greedy_seed": EPSILON_GREEDY_SEED,
        "prior_strength": PRIOR_STRENGTH,
        "epsilon": EPSILON,
        "p_global_treino": p_global,
        "n_rounds_treinados": int(len(raw_online)),
        "reproduziu_epic3_resultados": reproduziu,
        "resumo_geral": resumo_geral.round(5).to_dict(orient="records"),
    }
    with open(MODELS_DIR / "training_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    return metadata


if __name__ == "__main__":
    meta = train_and_serialize()
    print(f"Reproduziu epic3_resultados.json? {meta['reproduziu_epic3_resultados']}")
    print(f"Políticas serializadas em {MODELS_DIR}/")
    for row in meta["resumo_geral"]:
        print(row)
