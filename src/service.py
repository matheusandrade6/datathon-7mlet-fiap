"""
Serviço de recomendação de oferta — Epic 5, Task 5.1.2.

Expõe `POST /recommend`: recebe features de um cliente e devolve a oferta
(braço do bandit) recomendada pelas 3 políticas treinadas na Etapa 3/4
(`baseline`, `epsilon_greedy`, `thompson_sampling`), no MESMO modo "servir"
(determinístico, sem o termo de exploração aleatória usado só durante o
treino) usado em `notebooks/avaliacao_golden_set.ipynb` — ver a Seção 3
daquele notebook para a justificativa de cada regra:

  - baseline         -> sempre `baseline.best_arm` (não depende do cliente)
  - epsilon_greedy   -> argmax(estimated_rates) pós-treino (não depende do
                        cliente — a política não é contextual)
  - thompson_sampling -> argmax(posterior_means) DO SEGMENTO do cliente
                        (única política que varia por cliente)

O campo `recomendacao_primaria` no response usa `thompson_sampling` por ser o
algoritmo adaptativo principal do projeto (Seção 1.5 do PLANO_DATATHON.md) —
mas o response sempre traz as 3 recomendações lado a lado, exatamente como no
golden set da Etapa 4, para manter a decisão auditável/transparente.

Como rodar localmente:
    uvicorn src.service:app --reload --port 8000

Os modelos treinados (`models/*.joblib`) são carregados uma vez no startup.
Se não existirem, rode antes: `python -m src.train_policies`.
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.arms import ARM_DEFINITIONS, P_MIN, P_MAX, _segment_multiplier
from src.bandit import segment_customer, N_ARMS

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
PROCESSED_DIR = ROOT / "data" / "processed"

_STATE: dict = {}


def _load_arms_metadata() -> dict[int, dict]:
    with open(PROCESSED_DIR / "arms_definition.json", encoding="utf-8") as f:
        data = json.load(f)
    return {a["arm"]: a for a in data["arms"]}


def load_policies() -> None:
    """Carrega as políticas serializadas por `src/train_policies.py`."""
    required = ["baseline.joblib", "epsilon_greedy.joblib", "thompson_sampling.joblib"]
    missing = [f for f in required if not (MODELS_DIR / f).exists()]
    if missing:
        raise RuntimeError(
            f"Modelos ausentes em {MODELS_DIR}: {missing}. "
            "Rode `python -m src.train_policies` antes de subir o serviço."
        )
    _STATE["baseline"] = joblib.load(MODELS_DIR / "baseline.joblib")
    _STATE["epsilon_greedy"] = joblib.load(MODELS_DIR / "epsilon_greedy.joblib")
    _STATE["thompson_sampling"] = joblib.load(MODELS_DIR / "thompson_sampling.joblib")
    _STATE["arms_meta"] = _load_arms_metadata()
    _STATE["posterior_means"] = _STATE["thompson_sampling"].posterior_means().set_index("segmento")
    _STATE["rec_baseline_arm"] = int(_STATE["baseline"].best_arm)
    _STATE["rec_epsilon_arm"] = int(np.argmax(_STATE["epsilon_greedy"].estimated_rates))

    training_metadata_path = MODELS_DIR / "training_metadata.json"
    if not training_metadata_path.exists():
        raise RuntimeError(
            f"{training_metadata_path} ausente. Rode `python -m src.train_policies` antes de subir o serviço."
        )
    with open(training_metadata_path, encoding="utf-8") as f:
        _STATE["p_global_treino"] = json.load(f)["p_global_treino"]


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_policies()
    yield


app = FastAPI(
    title="Datathon MLET — Serviço de Recomendação de Oferta",
    description=(
        "Bandit adaptativo (Thompson Sampling contextual) para recomendação de oferta/mensagem "
        "bancária, comparado a um baseline determinístico e a um Epsilon-Greedy. Etapa 5 do "
        "Datathon POSTECH MLET (FIAP)."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


class ClienteRequest(BaseModel):
    """
    Features do cliente. Os campos que efetivamente determinam o segmento de
    contexto (e portanto a recomendação do Thompson Sampling) são: `job`,
    `education`, `poutcome`, `previous`, `pdays`, `month` — ver
    `src/bandit.segment_customer`. Os demais campos são aceitos para compor
    um payload realista de cliente (mesmo formato do golden set da Etapa 4)
    mas não influenciam a decisão das 3 políticas.
    """

    age: Optional[int] = Field(None, examples=[41])
    job: str = Field(..., examples=["management"], description="Ex.: admin., management, technician, student, retired, blue-collar, ...")
    marital: Optional[str] = Field(None, examples=["married"])
    education: str = Field(..., examples=["university.degree"])
    housing: Optional[str] = Field(None, examples=["yes"])
    loan: Optional[str] = Field(None, examples=["no"])
    contact: Optional[str] = Field(None, examples=["cellular"])
    month: str = Field(..., examples=["may"], description="Mês do último contato (abreviação em inglês, ex.: 'may', 'mar', 'dec').")
    pdays: int = Field(..., examples=[999], description="Dias desde o último contato de campanha anterior; 999 = nunca contatado.")
    previous: int = Field(..., examples=[0], description="Número de contatos em campanhas anteriores.")
    poutcome: str = Field(..., examples=["nonexistent"], description="Resultado da campanha anterior: success, failure ou nonexistent.")


class RecomendacaoPolitica(BaseModel):
    arm: int
    nome: str
    descricao: str


class RecommendResponse(BaseModel):
    segmento: str
    p_arm_simulado: dict[str, float]
    recomendacoes: dict[str, RecomendacaoPolitica]
    recomendacao_primaria: Literal["baseline", "epsilon_greedy", "thompson_sampling"]
    politicas_concordam: bool


def _build_recommendation(cliente: ClienteRequest) -> RecommendResponse:
    row = pd.Series(cliente.model_dump())
    segmento = segment_customer(row)

    arms_meta = _STATE["arms_meta"]
    rec_baseline_arm = _STATE["rec_baseline_arm"]
    rec_epsilon_arm = _STATE["rec_epsilon_arm"]

    posterior_means = _STATE["posterior_means"]
    if segmento not in posterior_means.index:
        raise HTTPException(status_code=500, detail=f"Segmento '{segmento}' sem posterior treinado.")
    rec_ts_arm = int(
        posterior_means.loc[segmento, [f"arm_{a}" for a in range(N_ARMS)]].to_numpy().argmax()
    )

    def _rec(arm: int) -> RecomendacaoPolitica:
        meta = arms_meta[arm]
        return RecomendacaoPolitica(arm=arm, nome=meta["nome"], descricao=meta["descricao"])

    # Probabilidade de conversão *esperada* por braço (mesma regra de
    # src/arms.py: p_global * multiplicador_base * multiplicador_segmento),
    # SEM o termo de ruído gaussiano usado só na simulação da Etapa 2 — aqui é
    # só um valor ilustrativo/auditável no response (não é usado pelas
    # políticas para decidir, e por isso é determinístico por cliente).
    p_global = _STATE["p_global_treino"]
    p_arm_simulado = {}
    for arm, spec in sorted(ARM_DEFINITIONS.items()):
        p = p_global * spec["multiplicador_base"] * _segment_multiplier(row, arm)
        p_arm_simulado[f"p_arm_{arm}"] = round(float(np.clip(p, P_MIN, P_MAX)), 4)

    recomendacoes = {
        "baseline": _rec(rec_baseline_arm),
        "epsilon_greedy": _rec(rec_epsilon_arm),
        "thompson_sampling": _rec(rec_ts_arm),
    }
    concordam = len({rec_baseline_arm, rec_epsilon_arm, rec_ts_arm}) == 1

    return RecommendResponse(
        segmento=segmento,
        p_arm_simulado=p_arm_simulado,
        recomendacoes=recomendacoes,
        recomendacao_primaria="thompson_sampling",
        politicas_concordam=concordam,
    )


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "modelos_carregados": list(_STATE.keys()) != []}


@app.get("/arms")
def arms() -> list[dict]:
    """Metadados dos 4 braços simulados (ver src/arms.py e data/processed/arms_definition.json)."""
    return list(_STATE["arms_meta"].values())


@app.post("/recommend", response_model=RecommendResponse)
def recommend(cliente: ClienteRequest) -> RecommendResponse:
    return _build_recommendation(cliente)
