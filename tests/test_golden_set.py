"""
Teste do endpoint `POST /recommend` com os 5 casos do golden set — Epic 5,
Task 5.1.3.

Reaproveita os mesmos 5 clientes e o mesmo estado treinado avaliados em
`notebooks/avaliacao_golden_set.ipynb` (salvos em
`data/processed/epic4_resultados.json`) e confere que o serviço FastAPI
devolve, para cada cliente, a MESMA recomendação (arm) das 3 políticas que o
notebook calculou — ou seja, que servir via API não mudou nenhum resultado
já validado na Etapa 4.

Rodar:
    python -m pytest tests/test_golden_set.py -v
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from src.service import app

ROOT = Path(__file__).resolve().parent.parent


def _load_golden_set() -> list[dict]:
    with open(ROOT / "data" / "processed" / "epic4_resultados.json", encoding="utf-8") as f:
        data = json.load(f)
    return data["golden_set"]["clientes"]


def test_health():
    with TestClient(app) as client:
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"


def test_arms_metadata():
    with TestClient(app) as client:
        resp = client.get("/arms")
        assert resp.status_code == 200
        arms = resp.json()
        assert len(arms) == 4
        assert {a["arm"] for a in arms} == {0, 1, 2, 3}


def test_golden_set_matches_notebook_recommendations():
    golden = _load_golden_set()
    assert len(golden) == 5

    with TestClient(app) as client:
        for cliente in golden:
            features = cliente["features"]
            payload = {
                "age": features["age"],
                "job": features["job"],
                "marital": features["marital"],
                "education": features["education"],
                "housing": features["housing"],
                "loan": features["loan"],
                "contact": features["contact"],
                "month": features["month"],
                "pdays": features["pdays"],
                "previous": features["previous"],
                "poutcome": features["poutcome"],
            }
            resp = client.post("/recommend", json=payload)
            assert resp.status_code == 200, resp.text
            body = resp.json()

            esperado = cliente["recomendacao"]
            caso = cliente["caso"]

            assert body["segmento"] == cliente["segmento"], f"{caso}: segmento divergente"
            for politica in ["baseline", "epsilon_greedy", "thompson_sampling"]:
                arm_obtido = body["recomendacoes"][politica]["arm"]
                arm_esperado = esperado[politica]["arm"]
                assert arm_obtido == arm_esperado, (
                    f"{caso} / {politica}: esperado braço {arm_esperado}, "
                    f"serviço devolveu braço {arm_obtido}"
                )

            print(f"OK — {caso}: segmento={body['segmento']}, "
                  f"baseline={body['recomendacoes']['baseline']['arm']}, "
                  f"epsilon_greedy={body['recomendacoes']['epsilon_greedy']['arm']}, "
                  f"thompson_sampling={body['recomendacoes']['thompson_sampling']['arm']}")


if __name__ == "__main__":
    test_health()
    test_arms_metadata()
    test_golden_set_matches_notebook_recommendations()
    print("\nTodos os 5 casos do golden set bateram com data/processed/epic4_resultados.json.")
