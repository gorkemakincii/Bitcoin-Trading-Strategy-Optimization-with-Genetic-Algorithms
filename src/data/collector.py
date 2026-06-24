"""BTC tarihsel veri toplama modulu."""
import pandas as pd
import yfinance as yf


def collect_btc_data(start_date: str, end_date: str, interval: str = "1d") -> pd.DataFrame:
    """BTC/USD tarihsel fiyat verilerini toplar."""
    btc = yf.download("BTC-USD", start=start_date, end=end_date, interval=interval)
    # yfinance yeni versiyonlarda MultiIndex column dondurebilir
    if isinstance(btc.columns, pd.MultiIndex):
        btc.columns = btc.columns.get_level_values(0)
    return btc


if __name__ == "__main__":
    from config.config import DataConfig

    cfg = DataConfig()
    data = collect_btc_data(cfg.start_date, cfg.end_date, cfg.interval)
    data.to_csv("data/raw/btc_daily.csv")
    print(f"Veri toplandi: {len(data)} satir, {data.index[0]} - {data.index[-1]}")
