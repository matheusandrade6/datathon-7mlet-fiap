"""
Baseline determinístico + políticas adaptativas (bandit) — Epic 3 (Tasks 3.1.1-3.1.4).

Contexto (ver README "Preparação da base e definição dos braços — Etapa 2" e
`src/arms.py`): cada cliente tem uma probabilidade de conversão simulada por
braço (`p_arm_0..p_arm_3`, em `data/processed/arm_probabilities_{train,online}.csv`).
Este módulo implementa três políticas de decisão de braço e a infraestrutura
para simular sua performance na base de simulação online (`online_sim.csv`):

1. `DeterministicBaseline` — sempre oferece o mesmo braço (o de maior
   conversão média histórica no treino). Não se adapta ao cliente nem ao
   tempo — é o piso de comparação.
2. `EpsilonGreedyBandit` — política adaptativa *não contextual*: mantém uma
   média de conversão observada por braço (global) e explora com
   probabilidade `epsilon`, do contrário explota o braço de maior média.
3. `ThompsonSamplingContextual` — política adaptativa *contextual
   simplificada*: mantém um posterior Beta-Bernoulli por braço **dentro de
   cada segmento de cliente** (ver `segment_customer`), amostrando de cada
   posterior e escolhendo o braço com maior amostra (Thompson Sampling
   clássico), o que equilibra exploração/explotação automaticamente via a
   incerteza do posterior.

Priors do Thompson Sampling (documentado para Task 3.1.5): em vez de partir de
um prior uniforme Beta(1,1) (equivalente a "não sabemos nada sobre nenhum
braço"), usamos um *warm start* informado pelos dados de treino: para cada
(segmento, braço), calculamos a probabilidade média simulada (`p_arm_i`) dos
clientes de treino naquele segmento e a convertemos em pseudo-observações
`k = min(n_clientes_do_segmento_no_treino, PRIOR_STRENGTH)` — ou seja, a força
do prior é proporcional ao quanto de histórico realmente existe naquele
segmento, com um teto (`PRIOR_STRENGTH = 500`) para não zerar a capacidade de
adaptação online mesmo em segmentos com muitos dados de treino. Isso reflete o
fato de que, na prática, já existe histórico de campanha (a base de treino,
28.831 clientes) antes de a política adaptativa começar a operar — não faria
sentido ignorá-lo e começar do zero.

Ablação registrada no notebook (Task 3.1.5): com prior uninformado Beta(1,1)
(`PRIOR_STRENGTH=0`), o Thompson Sampling fica **pior** que o Epsilon-Greedy
não contextual durante toda a simulação online (~12 mil rounds não bastam
para aprender do zero 4 braços x 4 segmentos com diferenças pequenas entre
braços). Com o warm start (`PRIOR_STRENGTH=500`), o Thompson Sampling passa a
superar o Epsilon-Greedy de forma consistente — mostrando que o prior
informado não é um "atalho" cosmético, é o que torna a política contextual
viável dentro do horizonte de simulação disponível.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

N_ARMS = 4
PRIOR_STRENGTH = 500  # teto de pseudo-observações (por segmento) usado para converter p_arm médio (treino) em (alpha0, beta0)
EPSILON = 0.10  # taxa de exploração do Epsilon-Greedy

_HIGH_EDU = {"university.degree", "professional.course"}
_HIGH_JOB_ARM1 = {"admin.", "management", "technician"}
_HIGH_JOB_ARM3 = {"student", "retired"}
_HIGH_MONTHS_ARM3 = {"mar", "dec", "sep", "oct"}

SEGMENTS = ["poutcome_success", "novo_contato_qualificado", "perfil_qualificado", "padrao"]


def segment_customer(row: pd.Series) -> str:
    """
    Bucket de contexto (Thompson Sampling contextual simplificado).

    Reaproveita os sinais usados em `src/arms.py` para simular os braços
    (poutcome, job, education, pdays, month), reduzidos a 4 segmentos
    mutuamente exclusivos, com prioridade decrescente (do sinal mais forte
    e mais específico para o mais fraco/genérico, replicando a checagem de
    sanidade da Etapa 2 — `notebooks/preparacao_base.ipynb`):

      1. "poutcome_success"         -> poutcome == 'success'
                                        (braço 2 foi desenhado para este segmento;
                                        é o único onde o braço 2 supera o braço 1
                                        na média histórica de treino — ver Task 3.1.5)
      2. "novo_contato_qualificado" -> nunca contatado antes (pdays==999) E
                                        (estudante/aposentado OU mês de maior
                                        conversão histórica)
                                        (braço 3 foi desenhado para este segmento)
      3. "perfil_qualificado"       -> job em admin./management/technician OU
                                        educação superior
                                        (braço 1 foi desenhado para este segmento)
      4. "padrao"                   -> nenhum dos anteriores
                                        (nenhum braço tem boost -> braço 0 é a referência)

    Clientes com `previous > 0` mas sem `poutcome == 'success'` (contato
    anterior sem conversão) não formam um segmento próprio: empiricamente
    (Task 3.1.5) o multiplicador é só "moderado" nesses casos e não muda o
    braço vencedor, então caem em "perfil_qualificado"/"padrao" conforme o
    restante do perfil. A ordem de prioridade evita que um cliente caia em
    múltiplos segmentos ao mesmo tempo.
    """
    poutcome = row.get("poutcome")
    job = row.get("job")
    education = row.get("education")
    pdays = row.get("pdays")
    month = row.get("month")

    if poutcome == "success":
        return "poutcome_success"
    if pdays == 999 and (job in _HIGH_JOB_ARM3 or month in _HIGH_MONTHS_ARM3):
        return "novo_contato_qualificado"
    if job in _HIGH_JOB_ARM1 or education in _HIGH_EDU:
        return "perfil_qualificado"
    return "padrao"


def add_segment_column(df_raw: pd.DataFrame) -> pd.Series:
    """Aplica `segment_customer` a cada linha de um DataFrame bruto (colunas originais)."""
    return df_raw.apply(segment_customer, axis=1)


# ---------------------------------------------------------------------------
# Política 1 — Baseline determinístico (Task 3.1.1)
# ---------------------------------------------------------------------------


class DeterministicBaseline:
    """Sempre recomenda o braço de maior conversão média observada no treino."""

    def __init__(self, best_arm: int, historical_rates: np.ndarray):
        self.best_arm = best_arm
        self.historical_rates = historical_rates

    @classmethod
    def fit(cls, arm_probabilities_train: pd.DataFrame) -> "DeterministicBaseline":
        rates = arm_probabilities_train[[f"p_arm_{a}" for a in range(N_ARMS)]].mean().to_numpy()
        best_arm = int(np.argmax(rates))
        return cls(best_arm=best_arm, historical_rates=rates)

    def choose_arm(self, segment: str | None = None) -> int:
        return self.best_arm


# ---------------------------------------------------------------------------
# Política 2 — Epsilon-Greedy não contextual (Task 3.1.3)
# ---------------------------------------------------------------------------


class EpsilonGreedyBandit:
    """Epsilon-Greedy global (sem contexto): média corrente por braço + exploração aleatória."""

    def __init__(self, n_arms: int = N_ARMS, epsilon: float = EPSILON, seed: int = 42):
        self.n_arms = n_arms
        self.epsilon = epsilon
        self.counts = np.zeros(n_arms, dtype=int)
        self.sums = np.zeros(n_arms, dtype=float)
        self.rng = np.random.default_rng(seed)

    def choose_arm(self, segment: str | None = None) -> int:
        if self.rng.random() < self.epsilon or self.counts.sum() < self.n_arms:
            return int(self.rng.integers(self.n_arms))
        means = np.divide(self.sums, self.counts, out=np.full(self.n_arms, np.nan), where=self.counts > 0)
        means = np.nan_to_num(means, nan=-np.inf)
        return int(np.argmax(means))

    def update(self, arm: int, reward: float, segment: str | None = None) -> None:
        self.counts[arm] += 1
        self.sums[arm] += reward

    @property
    def estimated_rates(self) -> np.ndarray:
        return np.divide(self.sums, self.counts, out=np.zeros(self.n_arms), where=self.counts > 0)


# ---------------------------------------------------------------------------
# Política 3 — Thompson Sampling contextual simplificado (Task 3.1.2)
# ---------------------------------------------------------------------------


class ThompsonSamplingContextual:
    """
    Beta-Bernoulli por (segmento, braço), com prior informado pelo treino
    (ver docstring do módulo). `sample_arm` amostra theta~Beta(alpha,beta)
    para cada braço do segmento e escolhe o maior (exploração via incerteza
    do posterior); `update` faz o update bayesiano conjugado padrão.
    """

    def __init__(self, priors: dict[str, tuple[np.ndarray, np.ndarray]], n_arms: int = N_ARMS, seed: int = 42):
        self.n_arms = n_arms
        self.alpha = {seg: a.copy() for seg, (a, b) in priors.items()}
        self.beta = {seg: b.copy() for seg, (a, b) in priors.items()}
        self.rng = np.random.default_rng(seed)

    @classmethod
    def fit_priors(
        cls,
        raw_train: pd.DataFrame,
        arm_probabilities_train: pd.DataFrame,
        segments: list[str] = SEGMENTS,
        n_arms: int = N_ARMS,
        prior_strength: float = PRIOR_STRENGTH,
        seed: int = 42,
    ) -> "ThompsonSamplingContextual":
        """
        `prior_strength` é um TETO de pseudo-observações por segmento. A força
        real usada é `k = min(n_clientes_do_segmento_no_treino, prior_strength)`,
        para que a confiança do prior nunca exceda o histórico realmente
        observado naquele segmento (`prior_strength=0` reproduz o prior
        uninformado Beta(1,1) da ablação documentada no notebook).
        """
        seg_col = add_segment_column(raw_train)
        priors: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for seg in segments:
            mask = seg_col == seg
            n_seg = int(mask.sum())
            k = min(n_seg, prior_strength)
            if n_seg == 0:
                mean_p = np.full(n_arms, 0.5)
            else:
                mean_p = arm_probabilities_train.loc[mask, [f"p_arm_{a}" for a in range(n_arms)]].mean().to_numpy()
            alpha0 = mean_p * k + 1.0
            beta0 = (1.0 - mean_p) * k + 1.0
            priors[seg] = (alpha0, beta0)
        return cls(priors=priors, n_arms=n_arms, seed=seed)

    def choose_arm(self, segment: str) -> int:
        samples = self.rng.beta(self.alpha[segment], self.beta[segment])
        return int(np.argmax(samples))

    def update(self, arm: int, reward: float, segment: str) -> None:
        self.alpha[segment][arm] += reward
        self.beta[segment][arm] += 1.0 - reward

    def posterior_means(self) -> pd.DataFrame:
        rows = []
        for seg, alpha in self.alpha.items():
            beta = self.beta[seg]
            rows.append({"segmento": seg, **{f"arm_{a}": alpha[a] / (alpha[a] + beta[a]) for a in range(self.n_arms)}})
        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Simulação online (Task 3.1.4)
# ---------------------------------------------------------------------------


def run_online_simulation(
    raw_online: pd.DataFrame,
    arm_probabilities_online: pd.DataFrame,
    baseline: DeterministicBaseline,
    epsilon_greedy: EpsilonGreedyBandit,
    thompson: ThompsonSamplingContextual,
    seed: int = 123,
) -> pd.DataFrame:
    """
    Percorre `raw_online`/`arm_probabilities_online` linha a linha (simulando a
    chegada sequencial de clientes) e, para cada política, escolhe um braço e
    observa uma recompensa Bernoulli amostrada da probabilidade *verdadeira*
    simulada daquele braço para aquele cliente (`arm_probabilities_online`).

    Retorna um DataFrame long (1 linha por round x política) com:
      round, policy, segment, arm_chosen, reward, oracle_prob, chosen_prob
    `oracle_prob` é a maior probabilidade possível entre os 4 braços para
    aquele cliente (usada para calcular o regret esperado/pseudo-regret).
    """
    rng = np.random.default_rng(seed)
    prob_cols = [f"p_arm_{a}" for a in range(N_ARMS)]
    seg_col = add_segment_column(raw_online)

    records = []
    n = len(raw_online)
    for i in range(n):
        probs = arm_probabilities_online.iloc[i][prob_cols].to_numpy()
        oracle_prob = float(probs.max())
        segment = seg_col.iloc[i]

        # --- baseline determinístico ---
        arm_b = baseline.choose_arm(segment)
        reward_b = float(rng.random() < probs[arm_b])
        records.append((i, "baseline", segment, arm_b, reward_b, oracle_prob, float(probs[arm_b])))

        # --- epsilon-greedy (não contextual) ---
        arm_e = epsilon_greedy.choose_arm(segment)
        reward_e = float(rng.random() < probs[arm_e])
        epsilon_greedy.update(arm_e, reward_e, segment)
        records.append((i, "epsilon_greedy", segment, arm_e, reward_e, oracle_prob, float(probs[arm_e])))

        # --- thompson sampling contextual ---
        arm_t = thompson.choose_arm(segment)
        reward_t = float(rng.random() < probs[arm_t])
        thompson.update(arm_t, reward_t, segment)
        records.append((i, "thompson_sampling", segment, arm_t, reward_t, oracle_prob, float(probs[arm_t])))

    return pd.DataFrame(
        records,
        columns=["round", "policy", "segment", "arm_chosen", "reward", "oracle_prob", "chosen_prob"],
    )


def summarize_policies(sim_results: pd.DataFrame) -> pd.DataFrame:
    """Métricas finais por política: conversão total, regret acumulado, uplift vs. baseline."""
    sim_results = sim_results.sort_values(["policy", "round"]).copy()
    sim_results["cum_reward"] = sim_results.groupby("policy")["reward"].cumsum()
    sim_results["cum_regret"] = sim_results.groupby("policy").apply(
        lambda g: (g["oracle_prob"] - g["chosen_prob"]).cumsum()
    ).reset_index(level=0, drop=True)

    summary = (
        sim_results.groupby("policy")
        .agg(
            rounds=("round", "count"),
            conversoes=("reward", "sum"),
            taxa_conversao=("reward", "mean"),
            regret_acumulado=("cum_regret", "last"),
        )
        .reset_index()
    )
    baseline_rate = summary.loc[summary["policy"] == "baseline", "taxa_conversao"].iloc[0]
    summary["uplift_vs_baseline_%"] = (summary["taxa_conversao"] / baseline_rate - 1.0) * 100
    order = {"baseline": 0, "epsilon_greedy": 1, "thompson_sampling": 2}
    summary["_ord"] = summary["policy"].map(order)
    summary = summary.sort_values("_ord").drop(columns="_ord").reset_index(drop=True)
    return summary
