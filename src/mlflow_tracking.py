"""
Tracking de experimentos via MLflow — Epic 7, Tasks 7.1.1-7.1.3.

Este script NÃO re-treina nada: ele reaproveita os resultados já calculados
e validados nas Etapas 3/5 (`data/processed/epic3_resultados.json` e,
principalmente, `models/training_metadata.json`, gerado por
`src/train_policies.py`) como fonte única de params e métricas, e apenas os
registra no MLflow — 1 run por política (`baseline`, `epsilon_greedy`,
`thompson_sampling`), todos dentro do mesmo experimento, para permitir a
comparação lado a lado na MLflow UI (Task 7.1.4).

Tracking URI: arquivo local `mlflow.db` (SQLite) na raiz do repositório
(Task 7.1.1) — decisão fechada na Seção 1.5 do `PLANO_DATATHON.md` ("MLflow
localmente", grupo solo, sem necessidade de backend remoto/managed). O
backend "file store" clássico (diretório `mlruns/` sem banco) está em modo
de manutenção a partir do MLflow 3.x (aviso oficial recomendando migrar para
backend em banco) — por isso o tracking URI aqui é `sqlite:///mlflow.db`:
continua sendo um único arquivo local, sem servidor remoto, só num formato
que o MLflow atual suporta sem avisos de depreciação.

Uso:
    python -m src.mlflow_tracking
    mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
"""

from __future__ import annotations

import json
from pathlib import Path

import mlflow

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
DB_PATH = ROOT / "mlflow.db"

TRACKING_URI = f"sqlite:///{DB_PATH.as_posix()}"
EXPERIMENT_NAME = "datathon-bandit-recommendation"

# Params específicos de cada política (o resto — split_seed, online_sim_seed,
# n_rounds_treinados — é comum às 3 e vira param de nível de experimento/run).
POLICY_PARAMS = {
    "baseline": {
        "estrategia": "determinístico (maior conversão média histórica no treino)",
    },
    "epsilon_greedy": {
        "estrategia": "epsilon-greedy não contextual",
    },
    "thompson_sampling": {
        "estrategia": "Thompson Sampling contextual (Beta-Bernoulli por segmento, warm start)",
    },
}


def log_runs() -> list[str]:
    with open(MODELS_DIR / "training_metadata.json", encoding="utf-8") as f:
        metadata = json.load(f)

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    common_params = {
        "split_seed": metadata["split_seed"],
        "online_sim_size": metadata["online_sim_size"],
        "online_sim_seed": metadata["online_sim_seed"],
        "n_rounds_treinados": metadata["n_rounds_treinados"],
        "p_global_treino": round(metadata["p_global_treino"], 5),
    }

    run_ids = []
    for row in metadata["resumo_geral"]:
        policy = row["policy"]
        with mlflow.start_run(run_name=policy) as run:
            mlflow.set_tags(
                {
                    "projeto": "datathon-7mlet-fiap",
                    "etapa": "7 - Ciclo de vida MLOps",
                    "policy": policy,
                    "reproduziu_epic3_resultados": str(metadata["reproduziu_epic3_resultados"]),
                }
            )

            mlflow.log_params(common_params)
            mlflow.log_params(POLICY_PARAMS[policy])
            if policy == "epsilon_greedy":
                mlflow.log_param("epsilon", metadata["epsilon"])
                mlflow.log_param("epsilon_greedy_seed", metadata["epsilon_greedy_seed"])
            if policy == "thompson_sampling":
                mlflow.log_param("prior_strength", metadata["prior_strength"])

            mlflow.log_metric("rounds", row["rounds"])
            mlflow.log_metric("conversoes", row["conversoes"])
            mlflow.log_metric("taxa_conversao", row["taxa_conversao"])
            mlflow.log_metric("regret_acumulado", row["regret_acumulado"])
            mlflow.log_metric("uplift_vs_baseline_pct", row["uplift_vs_baseline_%"])

            run_ids.append(run.info.run_id)
            print(f"Run '{policy}' logado: run_id={run.info.run_id}")

    return run_ids


if __name__ == "__main__":
    ids = log_runs()
    print(f"\n{len(ids)} runs logados no experimento '{EXPERIMENT_NAME}'.")
    print(f"Tracking URI: {TRACKING_URI}")
    print("Para visualizar: mlflow ui --backend-store-uri file:./mlruns --port 5000")
