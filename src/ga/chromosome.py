"""Kromozom yapisi, gen sinirlari ve strateji parametresi donusumleri."""
from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Iterable, Sequence


# Gen sinirlari: (min, max). Tam sayi sinirlari int genleri temsil eder.
GENE_BOUNDS: list[tuple[int | float, int | float]] = [
    (7, 30),        # rsi_period
    (20.0, 40.0),   # rsi_oversold
    (60.0, 80.0),   # rsi_overbought
    (8, 16),        # macd_fast
    (20, 30),       # macd_slow
    (5, 12),        # macd_signal
    (15, 30),       # bb_period
    (1.5, 3.0),     # bb_std
    (5, 20),        # sma_short
    (30, 100),      # sma_long
    (0.01, 0.10),   # stop_loss
    (0.02, 0.20),   # take_profit
    (0.0, 1.0),     # weight_rsi
    (0.0, 1.0),     # weight_macd
    (0.0, 1.0),     # weight_bb
    (0.0, 1.0),     # weight_sma
]

GENE_NAMES: list[str] = [
    "rsi_period", "rsi_oversold", "rsi_overbought",
    "macd_fast", "macd_slow", "macd_signal",
    "bb_period", "bb_std",
    "sma_short", "sma_long",
    "stop_loss", "take_profit",
    "weight_rsi", "weight_macd", "weight_bb", "weight_sma",
]

GENE_COUNT = len(GENE_NAMES)
WEIGHT_SLICE = slice(12, 16)


@dataclass(frozen=True)
class StrategyParams:
    """GA kromozomundan cozulen trading stratejisi parametreleri."""

    rsi_period: int
    rsi_oversold: float
    rsi_overbought: float
    macd_fast: int
    macd_slow: int
    macd_signal: int
    bb_period: int
    bb_std: float
    sma_short: int
    sma_long: int
    stop_loss: float
    take_profit: float
    weight_rsi: float
    weight_macd: float
    weight_bb: float
    weight_sma: float


def _is_integer_gene(index: int) -> bool:
    low, high = GENE_BOUNDS[index]
    return isinstance(low, int) and isinstance(high, int)


def _coerce_gene(index: int, value: float | int) -> float | int:
    low, high = GENE_BOUNDS[index]
    coerced = max(low, min(high, value))
    if _is_integer_gene(index):
        return int(round(coerced))
    return float(coerced)


def _ensure_gene_count(individual: Sequence[float | int]) -> None:
    if len(individual) != GENE_COUNT:
        raise ValueError(
            f"Kromozom {GENE_COUNT} gen icermeli, bulunan: {len(individual)}"
        )


def create_individual(rng: random.Random | None = None) -> list[float | int]:
    """Gen sinirlari icinde rastgele bir birey olusturur."""
    rng = rng or random
    genes: list[float | int] = []
    for low, high in GENE_BOUNDS:
        if isinstance(low, int) and isinstance(high, int):
            genes.append(rng.randint(low, high))
        else:
            genes.append(rng.uniform(float(low), float(high)))
    return repair_individual(genes)


def decode_chromosome(
    individual: Sequence[float | int],
    *,
    repair: bool = True,
) -> StrategyParams:
    """Kromozomu StrategyParams dataclass'ina donusturur."""
    _ensure_gene_count(individual)
    values = list(individual)
    if repair:
        repair_individual(values)

    decoded = {
        name: _coerce_gene(index, values[index])
        for index, name in enumerate(GENE_NAMES)
    }
    return StrategyParams(**decoded)


def repair_individual(
    individual: list[float | int],
    *,
    normalize_weights: bool = True,
) -> list[float | int]:
    """Gen sinirlarini, temel siralama kisitlarini ve agirliklari duzeltir."""
    _ensure_gene_count(individual)

    for index, value in enumerate(individual):
        individual[index] = _coerce_gene(index, value)

    # Bu gen ciftlerinin araliklari ayrik tutuldugu icin sinir onarimi normalde
    # yeterlidir; yine de araliklar degisirse kisitlar korunur.
    if individual[3] >= individual[4]:
        individual[3] = min(int(individual[3]), int(individual[4]) - 1)
        individual[3] = _coerce_gene(3, individual[3])

    if individual[8] >= individual[9]:
        individual[8] = min(int(individual[8]), int(individual[9]) - 1)
        individual[8] = _coerce_gene(8, individual[8])

    if individual[1] >= individual[2]:
        midpoint = (float(individual[1]) + float(individual[2])) / 2.0
        individual[1] = _coerce_gene(1, midpoint - 1.0)
        individual[2] = _coerce_gene(2, midpoint + 1.0)

    if normalize_weights:
        weights = [max(0.0, float(weight)) for weight in individual[WEIGHT_SLICE]]
        weight_sum = sum(weights)
        if weight_sum <= 0.0:
            weights = [0.25, 0.25, 0.25, 0.25]
        else:
            weights = [weight / weight_sum for weight in weights]
        individual[WEIGHT_SLICE] = weights

    return individual


def validate_individual(individual: Sequence[float | int]) -> list[str]:
    """Bireyin ihlal ettigi kisitlari okunabilir hata listesi olarak dondurur."""
    errors: list[str] = []
    if len(individual) != GENE_COUNT:
        return [f"Kromozom uzunlugu {GENE_COUNT} olmali."]

    for index, value in enumerate(individual):
        low, high = GENE_BOUNDS[index]
        if value < low or value > high:
            errors.append(f"{GENE_NAMES[index]} sinir disinda: {value}")
        if _is_integer_gene(index) and int(round(value)) != value:
            errors.append(f"{GENE_NAMES[index]} tam sayi olmali: {value}")

    if individual[3] >= individual[4]:
        errors.append("macd_fast, macd_slow degerinden kucuk olmali.")
    if individual[8] >= individual[9]:
        errors.append("sma_short, sma_long degerinden kucuk olmali.")
    if individual[1] >= individual[2]:
        errors.append("rsi_oversold, rsi_overbought degerinden kucuk olmali.")

    weight_sum = sum(float(weight) for weight in individual[WEIGHT_SLICE])
    if weight_sum <= 0.0:
        errors.append("Indikator agirliklarinin toplami pozitif olmali.")

    return errors


def is_valid_individual(individual: Sequence[float | int]) -> bool:
    """Bireyin tum gen ve kisit kontrollerinden gecip gecmedigini dondurur."""
    return not validate_individual(individual)


def clone_individual(individual: Iterable[float | int]) -> list[float | int]:
    """Operatorlerin yan etkisiz kullanimi icin basit liste kopyasi olusturur."""
    return list(individual)
