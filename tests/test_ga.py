"""GA kromozom, operator ve fitness modulleri dogrulama scripti."""
import os
import random
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd

from src.ga.chromosome import (
    GENE_COUNT,
    StrategyParams,
    create_individual,
    decode_chromosome,
    is_valid_individual,
    repair_individual,
)
from src.ga.fitness import evaluate_individual, fitness_multi, fitness_single
from src.ga.operators import (
    blend_crossover,
    gaussian_mutation,
    select_elite,
    tournament_selection,
    uniform_mutation,
)


print("=" * 60)
print("GA MODUL DOGRULAMA TESTLERI")
print("=" * 60)

rng = random.Random(42)


print("\nTEST 1: Kromozom olusturma ve decode")
print("-" * 40)

individuals = [create_individual(rng) for _ in range(50)]
for individual in individuals:
    assert len(individual) == GENE_COUNT
    assert is_valid_individual(individual)
    assert abs(sum(individual[12:16]) - 1.0) < 1e-10

params = decode_chromosome(individuals[0])
assert isinstance(params, StrategyParams)
assert isinstance(params.rsi_period, int)
assert isinstance(params.macd_fast, int)
print("  [OK] 50 birey gecerli araliklarda olusturuldu.")


print("\nTEST 2: Gecersiz birey tamiri")
print("-" * 40)

bad = [
    100, 90.0, 10.0, 40, 1, 99, 99, 9.0,
    80, 1, -1.0, 1.0, 0.0, 0.0, 0.0, 0.0,
]
repaired = repair_individual(bad)
assert is_valid_individual(repaired)
assert repaired[3] < repaired[4]
assert repaired[8] < repaired[9]
assert repaired[1] < repaired[2]
assert abs(sum(repaired[12:16]) - 1.0) < 1e-10
print("  [OK] Sinir, siralama ve agirlik kisitlari onarildi.")


print("\nTEST 3: GA operatorleri")
print("-" * 40)

parent1 = create_individual(rng)
parent2 = create_individual(rng)
child1, child2 = blend_crossover(parent1[:], parent2[:], rng=rng)
assert is_valid_individual(child1)
assert is_valid_individual(child2)

mutated_g, = gaussian_mutation(child1[:], indpb=1.0, rng=rng)
mutated_u, = uniform_mutation(child2[:], indpb=1.0, rng=rng)
assert is_valid_individual(mutated_g)
assert is_valid_individual(mutated_u)

population = [create_individual(rng) for _ in range(10)]
fitness_getter = lambda ind: float(ind[0])
selected = tournament_selection(population, 5, tournsize=3, fitness_getter=fitness_getter, rng=rng)
elite = select_elite(population, 3, fitness_getter=fitness_getter)
assert len(selected) == 5
assert len(elite) == 3
assert elite[0][0] >= elite[-1][0]
print("  [OK] Caprazlama, mutasyon, turnuva ve elitizm calisti.")


print("\nTEST 4: Fitness entegrasyonu")
print("-" * 40)

np.random.seed(42)
n = 500
trend = np.linspace(0, 4000, n)
noise = np.random.randn(n).cumsum() * 80
close = 25_000.0 + trend + noise

base_individual = [
    14, 30.0, 70.0,
    12, 26, 9,
    20, 2.0,
    10, 50,
    0.05, 0.10,
    0.25, 0.25, 0.25, 0.25,
]

metrics = evaluate_individual(base_individual, close, min_trades=0)
assert np.isfinite(metrics["fitness_score"])
assert "total_return" in metrics
assert "sharpe_ratio" in metrics
assert "max_drawdown_risk" in metrics

single = fitness_single(base_individual, close, min_trades=0)
multi = fitness_multi(base_individual, close, min_trades=0)
assert len(single) == 1 and np.isfinite(single[0])
assert len(multi) == 3 and np.all(np.isfinite(multi))
print(
    f"  [OK] Fitness hesaplandi: Sharpe={metrics['sharpe_ratio']}, "
    f"Return={metrics['total_return']}%, Trades={metrics['total_trades']}"
)


print("\nTEST 5: Hatali birey cezasi")
print("-" * 40)

failed = evaluate_individual([1, 2, 3], close, repair=False)
assert failed["is_penalized"]
assert failed["fitness_score"] < -100_000
print("  [OK] Hatali birey dusuk fitness ile cezalandirildi.")


print("\nTEST 6: Stop-loss / take-profit genleri fitness'i etkiliyor")
print("-" * 40)

risk_n = 120
risk_close = np.linspace(100.0, 140.0, risk_n)
risk_df = pd.DataFrame({
    "High": risk_close * 1.03,
    "Low": risk_close * 0.995,
    "Close": risk_close,
})

risk_base = [
    14, 30.0, 70.0,
    12, 26, 9,
    20, 2.0,
    5, 30,
    0.01, 0.02,
    0.0, 0.0, 0.0, 1.0,
]
risk_loose = risk_base[:]
risk_loose[10] = 0.10
risk_loose[11] = 0.20

tight_metrics = evaluate_individual(risk_base, risk_df, min_trades=0)
loose_metrics = evaluate_individual(risk_loose, risk_df, min_trades=0)

assert tight_metrics["take_profit_exits"] > loose_metrics["take_profit_exits"], \
    "HATA: Take-profit geni cikis sayisini etkilemedi!"
assert tight_metrics["fitness_score"] != loose_metrics["fitness_score"], \
    "HATA: Stop-loss/take-profit genleri fitness sonucunu etkilemedi!"
print(
    "  [OK] Risk genleri fitness'i degistirdi: "
    f"tight TP={tight_metrics['take_profit_exits']}, "
    f"loose TP={loose_metrics['take_profit_exits']}"
)


print("\n" + "=" * 60)
print("[BASARILI] Tum GA dogrulama kontrolleri gecti!")
print("=" * 60)
