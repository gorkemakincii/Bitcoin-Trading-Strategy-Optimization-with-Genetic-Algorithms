# Faz 2: Veri Toplama ve On Isleme

**Sure:** Gun 1 - Ogleden Sonra (31 Mayis Cumartesi)
**Tarih:** 31 Mayis 2026
**Durum:** Tamamlandi
**Sorumlu:** Yigit (Ana), Mert Kerem (Destek)
**Bagimlilik:** Faz 1 (ayni gunun sabahi)

---

## Hedef

BTC tarihsel fiyat verilerini toplamak, temizlemek, teknik indikatörleri hesaplamak ve GA motoru icin kullanima hazir bir veri pipeline'i olusturmak.

---

## Gorevler

### 2.1 Veri Toplama
**Sorumlu:** Yigit
**Sure:** 1 saat (13:00-14:00)

- [ ] `src/data/collector.py` modulunun gelistirilmesi
- [ ] BTC/USD (veya BTC/USDT) tarihsel fiyat verilerinin cekilmesi:
  - **Kaynak:** yfinance (BTC-USD) veya Binance API (BTCUSDT)
  - **Zaman Araligi:** 2018-01-01 ile 2025-12-31 arasi (yeterli veri icin)
  - **Granularite:** Gunluk (daily) mumlar (OHLCV)
  - **Alanlar:** Date, Open, High, Low, Close, Volume
- [ ] Alternatif veri kaynagi ile dogrulama (cross-validation of data)
- [ ] Ham verinin `data/raw/` klasorune CSV olarak kaydedilmesi
- [ ] Veri toplama scriptinin tekrarlanabilir (reproducible) olmasi

```python
# Ornek veri toplama kodu taslagi
import yfinance as yf
import pandas as pd

def collect_btc_data(start_date: str, end_date: str, interval: str = '1d') -> pd.DataFrame:
    """BTC/USD tarihsel fiyat verilerini toplar."""
    btc = yf.download('BTC-USD', start=start_date, end=end_date, interval=interval)
    return btc

# Veriyi kaydet
btc_data = collect_btc_data('2018-01-01', '2025-12-31')
btc_data.to_csv('data/raw/btc_daily.csv')
```

### 2.2 Veri Temizleme ve Dogrulama
**Sorumlu:** Yigit
**Sure:** 1 saat (14:00-15:00)

- [ ] `src/data/preprocessor.py` modulunun gelistirilmesi
- [ ] Eksik veri (missing values) kontrolu ve islenmesi:
  - NaN degerlerinin tespiti
  - Forward-fill veya interpolasyon ile doldurma
- [ ] Asiri deger (outlier) analizi
- [ ] Veri tiplerinin dogrulanmasi (tarih formati, sayisal degerler)
- [ ] Duplike satirlarin kontrolu
- [ ] Veri butunlugu raporu olusturma:
  - Toplam satir sayisi
  - Tarih araligi
  - Eksik gun sayisi
  - Temel istatistikler (ortalama, std, min, max)

### 2.3 Teknik Indikatörlerin Hesaplanmasi
**Sorumlu:** Yigit & Mert Kerem
**Sure:** 2 saat (15:00-17:00)

- [ ] `src/indicators/technical.py` modulunun gelistirilmesi
- [ ] Asagidaki teknik indikatörlerin hesaplanmasi:

#### Trend Indikatörleri
| Indikatör | Aciklama | Varsayilan Periyot |
|-----------|----------|-------------------|
| SMA | Basit Hareketli Ortalama | 20, 50 |
| EMA | Ustel Hareketli Ortalama | 12, 26 |
| MACD | Hareketli Ort. Yakinlasma/Iraksakligi | 12, 26, 9 |

#### Momentum Indikatörleri
| Indikatör | Aciklama | Varsayilan Periyot |
|-----------|----------|-------------------|
| RSI | Goreceli Guc Endeksi | 14 |
| Stochastic | Stokastik Osilatör | 14, 3 |

#### Volatilite Indikatörleri
| Indikatör | Aciklama | Varsayilan Periyot |
|-----------|----------|-------------------|
| BB | Bollinger Bantlari | 20, 2 |
| ATR | Ortalama Gercek Aralik | 14 |

#### Hacim Indikatörleri
| Indikatör | Aciklama | Varsayilan Periyot |
|-----------|----------|-------------------|
| OBV | Dengeli Hacim | - |

```python
# Özel implementasyon (pure pandas/numpy) - harici kütüphane kullanilmadi
import numpy as np
import pandas as pd

def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Tum teknik indikatörleri hesaplar ve DataFrame'e ekler.
    Harici teknik analiz kütüphanesi kullanilmadan, pandas/numpy ile özel implementasyon."""
    # Trend
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']

    # Momentum
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI_14'] = 100 - (100 / (1 + rs))

    # Volatilite
    df['BB_Mid'] = df['Close'].rolling(window=20).mean()
    bb_std = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Mid'] + 2 * bb_std
    df['BB_Lower'] = df['BB_Mid'] - 2 * bb_std
    high_low = df['High'] - df['Low']
    high_close = (df['High'] - df['Close'].shift()).abs()
    low_close = (df['Low'] - df['Close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR_14'] = tr.rolling(window=14).mean()

    # Hacim
    df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()

    return df
```

### 2.4 Feature Engineering
**Sorumlu:** Mert Kerem
**Sure:** 1 saat (17:00-18:00)

- [ ] Turetilmis ozellikler olusturma:
  - Fiyat degisim oranlari (returns): gunluk, 3 gunluk, 7 gunluk
  - Logaritmik getiriler (log returns)
  - Volatilite (rolling std)
  - Fiyat momentum (ROC - Rate of Change)
  - Indikatörler arasi iliskiler (orn: RSI + MACD kombinasyonlari)
- [ ] Normalizasyon/Olceklendirme:
  - Min-Max normalizasyon veya Z-score standardizasyon
  - Her feature icin uygun olcekleme yonteminin secilmesi
- [ ] Korelasyon analizi:
  - Feature'lar arasi korelasyon matrisi
  - Yuksek korelasyonlu (multicollinear) feature'larin tespiti
  - Gereksiz feature'larin elenmesi

### 2.5 Train/Test Veri Ayirimi
**Sorumlu:** Yigit
**Sure:** 30 dk (18:00-18:30)

- [ ] Zaman serisi icin uygun train/test ayirimi:
  - **Train Seti:** 2018-01-01 ~ 2023-12-31 (%~80)
  - **Test Seti:** 2024-01-01 ~ 2025-12-31 (%~20)
  - **NOT:** Zaman serisi verilerinde rastgele bölme yapilmaz, kronolojik sira korunur
- [ ] Walk-forward validation yaklasiminin planlanmasi (Faz 5 icin)
- [ ] Islenmis verinin `data/processed/` klasorune kaydedilmesi

---

## Veri Akis Seması

```
Yahoo Finance / Binance API
        |
        v
  [Ham Veri Toplama]
  btc_daily_raw.csv
        |
        v
  [Temizleme & Dogrulama]
  - Eksik veri doldurma
  - Outlier kontrolu
        |
        v
  [Teknik Indikatörler]
  - RSI, MACD, BB, EMA, ATR...
        |
        v
  [Feature Engineering]
  - Returns, volatilite
  - Normalizasyon
  - Korelasyon eleme
        |
        v
  [Train/Test Bolme]
  btc_train.csv | btc_test.csv
```

---

## Ciktilar

| Cikti | Dosya/Konum |
|-------|-------------|
| Ham BTC verisi | `data/raw/btc_daily.csv` |
| Islenmis veri (indikatörlu) | `data/processed/btc_features.csv` |
| Train seti | `data/processed/btc_train.csv` |
| Test seti | `data/processed/btc_test.csv` |
| Veri Toplama Modulu | `src/data/collector.py` |
| On Isleme Modulu | `src/data/preprocessor.py` |
| Teknik Indikatör Modulu | `src/indicators/technical.py` |
| Veri Butunlugu Raporu | `notebooks/data_exploration.ipynb` |

---

## Basari Kriterleri

- [ ] En az 5 yillik gunluk BTC verisi toplanmis
- [ ] Eksik veri orani %1'in altinda
- [ ] En az 10 teknik indikatör hesaplanmis
- [ ] Feature'lar normalize edilmis
- [ ] Train/Test ayirimi kronolojik olarak yapilmis
- [ ] Tum veri pipeline'i tekrarlanabilir (tek komutla calisabilir)

---

## Notlar

- Zaman serisi verisinde **data leakage** (veri sizintisi) onemli bir risktir; test verisinden train verisine bilgi sizmamali
- Indikatör periyotlari GA tarafindan da optimize edilebilir (Faz 3'te kromozoma eklenebilir)
- Veri toplama asamasinda API rate limit'lerine dikkat edilmeli
- Islenmis verinin istatistiksel ozetini cikarip dokumante etmek rapor icin faydali olacaktir
