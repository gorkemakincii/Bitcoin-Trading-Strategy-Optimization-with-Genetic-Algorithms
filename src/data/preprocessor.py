"""Veri temizleme ve on isleme modulu."""
import pandas as pd
import numpy as np


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Ham veriyi temizler ve dogrular."""
    df = df.copy()
    df = df.dropna(how="all")
    df = df[~df.index.duplicated(keep="first")]
    df = df.ffill()
    return df


def split_train_test(df: pd.DataFrame, train_end: str, test_start: str):
    """Zaman serisi icin kronolojik train/test ayirimi."""
    train = df[df.index <= train_end]
    test = df[df.index >= test_start]
    return train, test


def get_data_report(df: pd.DataFrame) -> dict:
    """Veri butunlugu raporu olusturur."""
    return {
        "total_rows": len(df),
        "date_range": f"{df.index[0]} - {df.index[-1]}",
        "missing_values": df.isnull().sum().to_dict(),
        "basic_stats": df.describe().to_dict(),
    }
