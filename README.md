# Genetik Algoritmalar ile Bitcoin Trading Stratejisi Optimizasyonu

Ege Universitesi - Bilgisayar Muhendisligi Bolumu
Yapay Zeka Yontemleri (3+0) - 2025-2026 Bahar Donemi - Proje 2

## Takim

| Uye | Rol |
|-----|-----|
| Mert | Genetik Algoritma Cekirdek Gelistirme |
| Yigit | Veri Toplama, On Isleme & Feature Engineering |
| Mert Kerem | Trading Stratejisi & Backtesting |
| Gorkem Ege | Deneysel Calismalar & Rapor/Dokumantasyon |

## Proje Ozeti

Bu projede Genetik Algoritmalar (GA) kullanilarak Bitcoin (BTC) trading stratejilerinin parametreleri optimize edilmektedir. Sistem, tarihsel BTC fiyat verilerinden teknik indikatorler cikararak, GA ile en uygun trading parametrelerini (RSI/MACD/BB esikleri, stop-loss/take-profit seviyeleri, indikatör agirliklari) bulmaktadir.

## Kurulum

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Proje Yapisi

```
├── config/
│   └── config.py              # Proje konfigurasyonlari (DataClass)
├── src/
│   ├── data/
│   │   ├── collector.py       # BTC veri toplama (yfinance)
│   │   └── preprocessor.py    # Veri temizleme, train/test ayirimi
│   ├── indicators/
│   │   └── technical.py       # Teknik indikatorler (RSI, MACD, BB, vb.)
│   ├── ga/
│   │   ├── chromosome.py      # Kromozom yapisi, gen tanimlari, tamir fonk.
│   │   ├── fitness.py         # Sharpe tabanli fitness ve backtest entegrasyonu
│   │   └── operators.py       # Secim, caprazlama ve mutasyon operatorleri
│   ├── strategy/              # Trading sinyalleri ve backtesting
│   └── visualization/         # Grafik ve gorsellesitirme
├── data/
│   ├── raw/                   # Ham BTC verisi
│   └── processed/             # Islenmis veri
├── docs/
│   ├── AIM_2026_ENG.doc       # Proje gereksinimleri
│   ├── Project 2 Report Template - EN.docx
│   ├── Literature_Review.docx # Literatur taramasi (11 kaynak)
│   ├── teknoloji_secim_raporu.md
│   └── mert_rapor_notlari.md
├── results/
│   ├── figures/               # Cikti grafikleri
│   ├── tables/                # Sonuc tablolari
│   └── logs/                  # GA calisma loglari
├── tests/                     # Birim testleri
├── fazlar/                    # Proje faz planlama dokumanlari
├── ROADMAP.md                 # Proje yol haritasi
└── requirements.txt           # Python bagimliliklari
```

## Kullanim

```bash
# 1. Veri toplama
python -m src.data.collector

# 2. GA fitness hattini ornek veriyle deneme
python tests/test_ga.py
```

GA modulleri dogrudan import edilerek de kullanilabilir:

```python
from src.ga import create_individual, fitness_single

individual = create_individual()
score = fitness_single(individual, close_prices)
```

## Teknolojiler

- **Python 3.10+**
- **Ozel GA motoru** - Genetik Algoritma
- **pandas / numpy** - Teknik indikatorler ve vektorel hesaplama
- **yfinance** - BTC veri toplama
- **matplotlib** - Gorsellesitirme
