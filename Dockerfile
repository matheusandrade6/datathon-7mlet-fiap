# Datathon POSTECH MLET — Serviço de recomendação (Epic 5, Task 5.1.4 — opcional)
#
# Reforça a Etapa 6 (arquitetura em nuvem): esta é a mesma imagem que rodaria,
# em produção, atrás de API Gateway em ECS Fargate (ver README, seção
# "Arquitetura-alvo em nuvem — Etapa 6").
#
# Build:  docker build -t datathon-bandit-service .
# Run:    docker run --rm -p 8000:8000 datathon-bandit-service
# Nota: a imagem já treina/serializa as políticas no build (ver linha do
# train_policies abaixo), então o container sobe pronto para servir.

FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY data/ data/

RUN python -m src.train_policies

EXPOSE 8000

CMD ["uvicorn", "src.service:app", "--host", "0.0.0.0", "--port", "8000"]
