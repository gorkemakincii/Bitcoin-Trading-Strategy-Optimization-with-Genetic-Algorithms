"""Genetik Algoritma ana dongusu (engine).

Mevcut chromosome, fitness ve operators modullerini birlestirerek
evrimsel optimizasyon dongusunu calistirir.
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import pandas as pd

from src.ga.chromosome import (
    GENE_NAMES,
    clone_individual,
    create_individual,
    decode_chromosome,
)
from src.ga.fitness import evaluate_individual
from src.ga.operators import (
    blend_crossover,
    clone_population,
    gaussian_mutation,
    select_elite,
    tournament_selection,
)


@dataclass
class GAResult:
    """GA calistirmasinin sonuclari."""

    best_individual: list[float | int]
    best_fitness: float
    best_params: dict[str, Any]
    best_metrics: dict[str, Any]
    hall_of_fame: list[dict[str, Any]]
    logbook: list[dict[str, Any]]
    total_time: float
    config: dict[str, Any]


@dataclass
class GenerationStats:
    """Bir neslin istatistikleri."""

    generation: int
    avg_fitness: float
    max_fitness: float
    min_fitness: float
    std_fitness: float
    best_return: float
    best_sharpe: float
    best_trades: int
    elapsed: float


def run_ga(
    data: pd.DataFrame | np.ndarray,
    *,
    population_size: int = 100,
    num_generations: int = 50,
    crossover_rate: float = 0.8,
    mutation_rate: float = 0.2,
    tournament_size: int = 3,
    elite_size: int = 5,
    hof_size: int = 10,
    initial_capital: float = 10_000.0,
    position_size: float = 0.95,
    commission_rate: float = 0.001,
    slippage_rate: float = 0.0005,
    min_trades: int = 1,
    seed: int | None = None,
    verbose: bool = True,
) -> GAResult:
    """
    Genetik Algoritma ana dongusunu calistirir.

    Parametreler
    ----------
    data : DataFrame veya ndarray
        BTC kapanis fiyatlari (DataFrame ise 'Close' sutunu olmali).
    population_size : int
        Populasyon boyutu.
    num_generations : int
        Nesil sayisi.
    crossover_rate : float
        Caprazlama olasiligi.
    mutation_rate : float
        Mutasyon olasiligi.
    tournament_size : int
        Turnuva secim boyutu.
    elite_size : int
        Her nesilde korunan en iyi birey sayisi.
    hof_size : int
        Hall of Fame boyutu (tum calistirma boyunca en iyi N birey).
    seed : int | None
        Rastgele tohum (tekrarlanabilirlik icin).
    verbose : bool
        Nesil bazli ilerleme ciktisi.

    Donus
    -----
    GAResult
        En iyi birey, metrikler, hall of fame ve nesil logbook.
    """
    rng = random.Random(seed)
    start_time = time.time()

    fitness_kwargs = dict(
        initial_capital=initial_capital,
        position_size=position_size,
        commission_rate=commission_rate,
        slippage_rate=slippage_rate,
        min_trades=min_trades,
    )

    config = dict(
        population_size=population_size,
        num_generations=num_generations,
        crossover_rate=crossover_rate,
        mutation_rate=mutation_rate,
        tournament_size=tournament_size,
        elite_size=elite_size,
        hof_size=hof_size,
        seed=seed,
        **fitness_kwargs,
    )

    # --- Populasyonu baslat ---
    population = [create_individual(rng) for _ in range(population_size)]

    # --- Hall of Fame ---
    hall_of_fame: list[dict[str, Any]] = []
    logbook: list[dict[str, Any]] = []

    if verbose:
        print(f"GA Baslatildi | Pop: {population_size} | Nesil: {num_generations} | Seed: {seed}")
        print("-" * 80)

    # --- Evrim dongusu ---
    for gen in range(num_generations):
        gen_start = time.time()

        # 1. Tum populasyonu degerlendir
        fitness_results = []
        for ind in population:
            metrics = evaluate_individual(ind, data, **fitness_kwargs)
            fitness_results.append(metrics)

        fitness_scores = [m["fitness_score"] for m in fitness_results]

        # 2. Fitness index haritasi (operator secimi icin)
        score_map: dict[int, float] = {}
        for idx, score in enumerate(fitness_scores):
            score_map[id(population[idx])] = score

        # 3. Hall of Fame guncelle
        for ind, metrics in zip(population, fitness_results):
            _update_hof(hall_of_fame, ind, metrics, hof_size)

        # 4. Nesil istatistikleri
        valid_scores = [s for s in fitness_scores if s > -999_999]
        if not valid_scores:
            valid_scores = fitness_scores

        best_idx = int(np.argmax(fitness_scores))
        best_metrics_gen = fitness_results[best_idx]

        stats = GenerationStats(
            generation=gen,
            avg_fitness=float(np.mean(valid_scores)),
            max_fitness=float(np.max(fitness_scores)),
            min_fitness=float(np.min(valid_scores)),
            std_fitness=float(np.std(valid_scores)),
            best_return=float(best_metrics_gen.get("total_return", 0.0)),
            best_sharpe=float(best_metrics_gen.get("sharpe_ratio", 0.0)),
            best_trades=int(best_metrics_gen.get("total_trades", 0)),
            elapsed=time.time() - gen_start,
        )
        logbook.append(stats.__dict__)

        if verbose:
            print(
                f"Nesil {gen:3d}/{num_generations} | "
                f"Avg: {stats.avg_fitness:8.3f} | "
                f"Max: {stats.max_fitness:8.3f} | "
                f"Return: {stats.best_return:+7.1f}% | "
                f"Sharpe: {stats.best_sharpe:6.3f} | "
                f"Trades: {stats.best_trades:3d} | "
                f"{stats.elapsed:.1f}s"
            )

        # Son nesilde artik operatör uygulamaya gerek yok
        if gen == num_generations - 1:
            break

        # 5. Yeni nesil olustur

        # Birey-skor eslemesi icin index-tabanli lookup
        idx_fitness = {i: fitness_scores[i] for i in range(len(population))}
        fitness_getter = lambda ind, _m=idx_fitness, _p=population: _m.get(
            next((i for i, p in enumerate(_p) if p is ind), 0), -1e9
        )

        # Elitizm: en iyi bireyleri koru
        elite = select_elite(
            population, elite_size, fitness_getter=fitness_getter
        )
        new_population = [clone_individual(e) for e in elite]

        # Turnuva secimi ile ebeveyn havuzu
        parents = tournament_selection(
            population,
            population_size - elite_size,
            tournsize=tournament_size,
            fitness_getter=fitness_getter,
            rng=rng,
        )

        # Caprazlama ve mutasyon
        offspring = []
        i = 0
        while i < len(parents):
            p1 = clone_individual(parents[i])
            if i + 1 < len(parents):
                p2 = clone_individual(parents[i + 1])
            else:
                p2 = clone_individual(parents[0])

            # Caprazlama
            if rng.random() < crossover_rate:
                p1, p2 = blend_crossover(p1, p2, rng=rng)

            # Mutasyon
            if rng.random() < mutation_rate:
                (p1,) = gaussian_mutation(p1, rng=rng)
            if rng.random() < mutation_rate:
                (p2,) = gaussian_mutation(p2, rng=rng)

            offspring.append(p1)
            if len(new_population) + len(offspring) < population_size:
                offspring.append(p2)

            i += 2

        new_population.extend(offspring[: population_size - len(new_population)])
        population = new_population

    # --- Sonuclari derle ---
    total_time = time.time() - start_time

    if not hall_of_fame:
        # Fallback: son populasyonun en iyisi
        best_ind = population[0]
        best_met = evaluate_individual(best_ind, data, **fitness_kwargs)
        hall_of_fame.append({"individual": best_ind, "metrics": best_met})

    best_entry = hall_of_fame[0]
    best_individual = best_entry["individual"]
    best_metrics_final = best_entry["metrics"]
    best_params = decode_chromosome(best_individual)

    if verbose:
        print("-" * 80)
        print(f"GA Tamamlandi | Sure: {total_time:.1f}s")
        print(f"En Iyi Fitness: {best_metrics_final['fitness_score']:.4f}")
        print(f"En Iyi Return:  {best_metrics_final.get('total_return', 0):.2f}%")
        print(f"En Iyi Sharpe:  {best_metrics_final.get('sharpe_ratio', 0):.4f}")
        print(f"Max Drawdown:   {best_metrics_final.get('max_drawdown', 0):.2f}%")
        print(f"Trade Sayisi:   {best_metrics_final.get('total_trades', 0)}")
        print(f"\nOptimal Parametreler:")
        for name, value in best_params.__dict__.items():
            print(f"  {name:20s}: {value}")

    return GAResult(
        best_individual=best_individual,
        best_fitness=float(best_metrics_final["fitness_score"]),
        best_params=best_params.__dict__,
        best_metrics=best_metrics_final,
        hall_of_fame=hall_of_fame,
        logbook=logbook,
        total_time=total_time,
        config=config,
    )


def _update_hof(
    hof: list[dict[str, Any]],
    individual: list[float | int],
    metrics: dict[str, Any],
    max_size: int,
) -> None:
    """Hall of Fame'i gunceller (en iyi max_size bireyi tutar)."""
    score = float(metrics.get("fitness_score", -1e9))

    entry = {
        "individual": clone_individual(individual),
        "metrics": dict(metrics),
        "fitness_score": score,
    }

    if len(hof) < max_size:
        hof.append(entry)
        hof.sort(key=lambda x: x["fitness_score"], reverse=True)
    elif score > hof[-1]["fitness_score"]:
        hof[-1] = entry
        hof.sort(key=lambda x: x["fitness_score"], reverse=True)
