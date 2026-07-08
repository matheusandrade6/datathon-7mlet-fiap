# datathon-7mlet-fiap

Plataforma de experimentação adaptativa (multi-armed bandit) para decidir, por cliente e canal digital, qual oferta, mensagem ou próximo passo apresentar — Datathon POSTECH MLET.

## Visão do problema

Instituições financeiras digitais precisam decidir, em diferentes canais, qual oferta apresentar a cada cliente elegível. Regras fixas e testes A/B longos desperdiçam tráfego e reagem lentamente a mudanças de contexto. Este projeto implementa uma abordagem adaptativa (bandit) que aprende continuamente qual oferta converte melhor, equilibrando exploração e explotação, e compara essa política contra um baseline determinístico.

O objetivo do projeto **não** é reproduzir um sistema bancário real, e sim demonstrar maturidade em ML Engineering: formulação do problema, baseline, versionamento de experimentos, serviço de inferência, avaliação e documentação de limitações/governança.

## Índice do projeto

| Etapa | Conteúdo | Status |
|-------|----------|--------|
| 0 — Organização do projeto | Repositório, dependências, README | ✅ |
| 1 — Base Kaggle e EDA | `notebooks/01_eda.ipynb`, link da base | ⬜ TODO |
| 2 — Preparação da base | Features do cliente + definição dos braços | ⬜ TODO |
| 3 — Baseline e estratégia algorítmica | Baseline vs. Thompson Sampling / Epsilon-Greedy | ⬜ TODO |
| 4 — Avaliação e casos de teste | Métricas + golden set de 5 clientes | ⬜ TODO |
| 5 — Serviço demonstrável | API FastAPI `/recommend` | ⬜ TODO |
| 6 — Arquitetura-alvo em nuvem | Parágrafo de arquitetura AWS | ⬜ TODO |
| 7 — Ciclo de vida MLOps | Tracking de experimentos via MLflow | ⬜ TODO |
| 8 — Apresentação final | Vídeo pitch (≤5 min) | ⬜ TODO |

## Estrutura do repositório

```
.
├── data/        # dados brutos e processados (Kaggle)
├── docs/        # documentação de apoio
├── models/      # artefatos de modelo/política serializados
├── notebooks/   # EDA, baseline, bandit, avaliação
└── src/         # código do serviço de recomendação (API)
```

## Como executar localmente

```bash
# 1. criar e ativar o ambiente virtual
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 2. instalar dependências
pip install -r requirements.txt

# 3. abrir os notebooks (EDA, baseline/bandit, avaliação)
jupyter notebook notebooks/

# 4. rodar o serviço de recomendação (após a Etapa 5)
uvicorn src.main:app --reload
```

> Nota: `requirements.txt` ainda será adicionado (Etapa 0, em andamento).

## Base de dados (Etapa 1)

*A preencher: link da base Kaggle escolhida, número de linhas/colunas, variável alvo e colunas descartadas por vazamento (ex. `duration`).*

## Preparação da base e definição dos braços (Etapa 2)

*A preencher: features do cliente utilizadas e como os braços (ofertas) foram definidos/simulados a partir da base.*

## Baseline e estratégia algorítmica (Etapa 3)

*A preencher: métrica do baseline determinístico, algoritmo adaptativo escolhido (priors, estratégia de exploração) e comparação de conversão contra o baseline.*

## Avaliação e casos de teste (Etapa 4)

*A preencher: métricas de avaliação e os 5 casos do golden set com a oferta recomendada para cada cliente.*

## Serviço de recomendação (Etapa 5)

*A preencher: como chamar o endpoint/script e exemplo de request/response.*

## Arquitetura-alvo em nuvem (Etapa 6)

*A preencher: parágrafo explicando os serviços de nuvem (AWS) usados para colocar o projeto em produção.*

## Ciclo de vida MLOps (Etapa 7)

*A preencher: como o MLflow foi usado para versionar parâmetros e métricas dos experimentos.*

## Governança e uso de dados

Este projeto usa exclusivamente dados públicos do Kaggle, sem dados reais de clientes, identificadores, patrimônio, renda, gênero ou raça. Decisões de oferta mantêm humano no loop antes de qualquer expansão de braços, e o uso dos dados segue princípios de minimização e retenção mínima necessária para fins educacionais deste desafio.

## Apresentação final (Etapa 8)

*A preencher: link do vídeo pitch (≤5 min).*
