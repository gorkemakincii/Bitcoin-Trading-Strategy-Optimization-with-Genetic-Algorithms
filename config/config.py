"""Proje konfigurasyonlari."""
from dataclasses import dataclass


@dataclass
class DataConfig:
    start_date: str = "2018-01-01"
    end_date: str = "2025-12-31"
    train_end_date: str = "2023-12-31"
    test_start_date: str = "2024-01-01"
    symbol: str = "BTC-USD"
    interval: str = "1d"


@dataclass
class StrategyConfig:
    initial_capital: float = 10000.0
    commission_rate: float = 0.001
    slippage_rate: float = 0.0005
    position_size: float = 0.95


@dataclass
class GAConfig:
    population_size: int = 100
    num_generations: int = 50
    crossover_rate: float = 0.8
    mutation_rate: float = 0.2
    tournament_size: int = 3
    elite_size: int = 5
    hof_size: int = 10
