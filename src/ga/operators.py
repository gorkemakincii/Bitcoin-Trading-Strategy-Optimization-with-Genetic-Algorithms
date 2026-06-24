"""Genetik algoritma secim, caprazlama ve mutasyon operatorleri."""
from __future__ import annotations

import random
from typing import Callable, Sequence

from src.ga.chromosome import GENE_BOUNDS, clone_individual, repair_individual

FitnessGetter = Callable[[Sequence[float | int]], float]


def _rng(rng: random.Random | None = None) -> random.Random:
    return rng or random


def _default_fitness(individual: Sequence[float | int]) -> float:
    fitness = getattr(individual, "fitness", None)
    if fitness is not None and hasattr(fitness, "values") and fitness.values:
        return float(fitness.values[0])
    if fitness is not None and isinstance(fitness, (int, float)):
        return float(fitness)
    if hasattr(individual, "fitness_score"):
        return float(getattr(individual, "fitness_score"))
    raise ValueError("fitness_getter verilmedi ve bireyde fitness degeri bulunamadi.")


def tournament_selection(
    population: Sequence[Sequence[float | int]],
    k: int,
    *,
    tournsize: int = 3,
    fitness_getter: FitnessGetter | None = None,
    rng: random.Random | None = None,
) -> list[Sequence[float | int]]:
    """Turnuva secimi ile k adet birey secer."""
    if not population:
        raise ValueError("Populasyon bos olamaz.")
    if k < 0:
        raise ValueError("Secilecek birey sayisi negatif olamaz.")

    rng = _rng(rng)
    fitness_getter = fitness_getter or _default_fitness
    tournsize = max(1, min(tournsize, len(population)))

    selected: list[Sequence[float | int]] = []
    for _ in range(k):
        competitors = rng.sample(list(population), tournsize)
        selected.append(max(competitors, key=fitness_getter))
    return selected


def select_elite(
    population: Sequence[Sequence[float | int]],
    elite_size: int,
    *,
    fitness_getter: FitnessGetter | None = None,
) -> list[Sequence[float | int]]:
    """Populasyondaki en iyi elite_size bireyi sirali olarak dondurur."""
    if elite_size < 0:
        raise ValueError("Elit birey sayisi negatif olamaz.")
    fitness_getter = fitness_getter or _default_fitness
    ordered = sorted(population, key=fitness_getter, reverse=True)
    return ordered[:elite_size]


def blend_crossover(
    ind1: list[float | int],
    ind2: list[float | int],
    *,
    alpha: float = 0.5,
    rng: random.Random | None = None,
) -> tuple[list[float | int], list[float | int]]:
    """BLX-alpha caprazlama uygular ve iki bireyi gen sinirlarina tamir eder."""
    if len(ind1) != len(GENE_BOUNDS) or len(ind2) != len(GENE_BOUNDS):
        raise ValueError("Caprazlama icin iki birey de 16 gen icermeli.")
    rng = _rng(rng)

    for index, (low_bound, high_bound) in enumerate(GENE_BOUNDS):
        low = min(float(ind1[index]), float(ind2[index]))
        high = max(float(ind1[index]), float(ind2[index]))
        width = high - low
        if width == 0:
            child1 = child2 = low
        else:
            child_low = low - alpha * width
            child_high = high + alpha * width
            child1 = rng.uniform(child_low, child_high)
            child2 = rng.uniform(child_low, child_high)

        ind1[index] = max(float(low_bound), min(float(high_bound), child1))
        ind2[index] = max(float(low_bound), min(float(high_bound), child2))

    return repair_individual(ind1), repair_individual(ind2)


def gaussian_mutation(
    individual: list[float | int],
    *,
    mu: float = 0.0,
    sigma_ratio: float = 0.1,
    indpb: float = 0.2,
    rng: random.Random | None = None,
) -> tuple[list[float | int]]:
    """Her gen icin olasilikli Gaussian mutasyon uygular."""
    if len(individual) != len(GENE_BOUNDS):
        raise ValueError("Mutasyon icin birey 16 gen icermeli.")
    rng = _rng(rng)

    for index, (low, high) in enumerate(GENE_BOUNDS):
        if rng.random() < indpb:
            sigma = (float(high) - float(low)) * sigma_ratio
            individual[index] = float(individual[index]) + rng.gauss(mu, sigma)

    return (repair_individual(individual),)


def uniform_mutation(
    individual: list[float | int],
    *,
    indpb: float = 0.1,
    rng: random.Random | None = None,
) -> tuple[list[float | int]]:
    """Secilen genleri kendi sinirlari icinde rastgele yeniden ornekler."""
    if len(individual) != len(GENE_BOUNDS):
        raise ValueError("Mutasyon icin birey 16 gen icermeli.")
    rng = _rng(rng)

    for index, (low, high) in enumerate(GENE_BOUNDS):
        if rng.random() < indpb:
            if isinstance(low, int) and isinstance(high, int):
                individual[index] = rng.randint(low, high)
            else:
                individual[index] = rng.uniform(float(low), float(high))

    return (repair_individual(individual),)


def clone_population(
    population: Sequence[Sequence[float | int]],
) -> list[list[float | int]]:
    """Yan etkisiz operator kullanimi icin populasyon kopyasi uretir."""
    return [clone_individual(individual) for individual in population]
