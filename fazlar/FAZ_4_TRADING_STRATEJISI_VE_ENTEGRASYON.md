# Faz 4: Trading Stratejisi ve Entegrasyon

**Sure:** Gun 2 - Ogleden Sonra (1 Haziran Pazar)
**Tarih:** 1 Haziran 2026
**Durum:** Tamamlandi
**Sorumlu:** Mert Kerem (Ana), Mert (Destek)
**Bagimlilik:** Faz 3 (ayni gunun sabahi)

---

## Hedef

GA tarafindan optimize edilen parametreleri kullanarak trading sinyalleri ureten bir strateji motoru ve bu stratejinin tarihsel veri uzerinde performansini olcen bir backtesting framework'u gelistirmek.

---

## Gorevler

### 4.1 Trading Sinyal Uretim Mekanizmasi
**Sorumlu:** Mert Kerem
**Sure:** 1.5 saat (13:00-14:30)

- [ ] `src/strategy/signals.py` modulunun gelistirilmesi
- [ ] Sinyal uretim mantigi:

#### Indikatör Bazli Sinyaller

Her indikatör icin alis/satis sinyali uretimi:

```python
def generate_rsi_signal(data, rsi_period, oversold, overbought):
    """
    RSI bazli sinyal uretimi.
    +1: Alis (RSI < oversold)
    -1: Satis (RSI > overbought)
     0: Notr
    """
    rsi = calculate_rsi(data['Close'], rsi_period)
    signal = pd.Series(0, index=data.index)
    signal[rsi < oversold] = 1     # Asiri satim -> Alis
    signal[rsi > overbought] = -1  # Asiri alim -> Satis
    return signal

def generate_macd_signal(data, fast, slow, signal_period):
    """
    MACD bazli sinyal uretimi.
    +1: MACD, sinyal cizgisini yukari kestiginde
    -1: MACD, sinyal cizgisini asagi kestiginde
    """
    macd_line, signal_line = calculate_macd(data['Close'], fast, slow, signal_period)
    signal = pd.Series(0, index=data.index)
    # Yukari kesisme (bullish crossover)
    signal[(macd_line > signal_line) & (macd_line.shift(1) <= signal_line.shift(1))] = 1
    # Asagi kesisme (bearish crossover)
    signal[(macd_line < signal_line) & (macd_line.shift(1) >= signal_line.shift(1))] = -1
    return signal

def generate_bb_signal(data, period, std_dev):
    """
    Bollinger Band bazli sinyal uretimi.
    +1: Fiyat alt bandın altina dustugunde
    -1: Fiyat ust bandın ustune ciktiginda
    """
    upper, middle, lower = calculate_bbands(data['Close'], period, std_dev)
    signal = pd.Series(0, index=data.index)
    signal[data['Close'] < lower] = 1    # Alis
    signal[data['Close'] > upper] = -1   # Satis
    return signal

def generate_sma_signal(data, short_period, long_period):
    """
    SMA crossover sinyal uretimi.
    +1: Kisa SMA uzun SMA'yi yukari kestiginde (Golden Cross)
    -1: Kisa SMA uzun SMA'yi asagi kestiginde (Death Cross)
    """
    sma_short = data['Close'].rolling(short_period).mean()
    sma_long = data['Close'].rolling(long_period).mean()
    signal = pd.Series(0, index=data.index)
    signal[(sma_short > sma_long) & (sma_short.shift(1) <= sma_long.shift(1))] = 1
    signal[(sma_short < sma_long) & (sma_short.shift(1) >= sma_long.shift(1))] = -1
    return signal
```

#### Bilesik (Composite) Sinyal

```python
def generate_composite_signal(data, params):
    """
    Tum indikatör sinyallerini agirlikli olarak birlestirir.
    GA'nin optimize ettigi agirliklar kullanilir.
    """
    rsi_sig = generate_rsi_signal(data, params['rsi_period'],
                                   params['rsi_oversold'], params['rsi_overbought'])
    macd_sig = generate_macd_signal(data, params['macd_fast'],
                                     params['macd_slow'], params['macd_signal'])
    bb_sig = generate_bb_signal(data, params['bb_period'], params['bb_std'])
    sma_sig = generate_sma_signal(data, params['sma_short'], params['sma_long'])

    # Agirlikli bilesik sinyal
    composite = (params['weight_rsi'] * rsi_sig +
                 params['weight_macd'] * macd_sig +
                 params['weight_bb'] * bb_sig +
                 params['weight_sma'] * sma_sig)

    # Esik degeri ile karar
    final_signal = pd.Series(0, index=data.index)
    threshold = 0.3  # veya bu da GA ile optimize edilebilir
    final_signal[composite > threshold] = 1    # ALIS
    final_signal[composite < -threshold] = -1  # SATIS

    return final_signal
```

### 4.2 Backtesting Framework'u
**Sorumlu:** Mert Kerem & Mert
**Sure:** 1.5 saat (14:30-16:00)

- [x] `src/strategy/backtest.py` modulunun gelistirilmesi
- [x] Backtesting motoru:

> **Not:** Asagidaki class-based event-driven backtest taslagi yerine, performans icin **vektörel backtesting** (`run_backtest_fast`) yaklasimiyla implement edildi. Bu yaklasim pandas vektörel islemleri kullanarak dongü (loop) yerine toplu hesaplama yapar ve GA fitness degerlendirmesini onemli olcude hizlandirir.

```python
class Backtester:
    """Trading stratejisini tarihsel veri uzerinde test eder."""

    def __init__(self, initial_capital=10000, commission=0.001):
        self.initial_capital = initial_capital
        self.commission = commission  # %0.1 islem komisyonu

    def run(self, data, signals, stop_loss, take_profit):
        """
        Backtesting calistirir.

        Parametreler:
        - data: OHLCV veri
        - signals: Alis/Satis sinyalleri (+1, -1, 0)
        - stop_loss: Zarar durdur yuzdesi
        - take_profit: Kar al yuzdesi

        Donus:
        - portfolio: Portfoy degisimi (equity curve)
        - trades: Yapilan islemler listesi
        - metrics: Performans metrikleri
        """
        portfolio = pd.DataFrame(index=data.index)
        portfolio['cash'] = self.initial_capital
        portfolio['holdings'] = 0.0
        portfolio['total'] = self.initial_capital

        trades = []
        position = None  # Acik pozisyon bilgisi

        for i in range(1, len(data)):
            date = data.index[i]
            price = data['Close'].iloc[i]

            # Stop-loss / Take-profit kontrolu
            if position is not None:
                pnl_pct = (price - position['entry_price']) / position['entry_price']
                if pnl_pct <= -stop_loss or pnl_pct >= take_profit:
                    # Pozisyonu kapat
                    profit = position['quantity'] * (price - position['entry_price'])
                    commission_cost = position['quantity'] * price * self.commission
                    trades.append({
                        'exit_date': date,
                        'exit_price': price,
                        'profit': profit - commission_cost,
                        'type': 'stop_loss' if pnl_pct <= -stop_loss else 'take_profit'
                    })
                    position = None

            # Sinyal isleme
            if signals.iloc[i] == 1 and position is None:
                # ALIS
                quantity = (portfolio['cash'].iloc[i-1] * 0.95) / price
                commission_cost = quantity * price * self.commission
                position = {
                    'entry_date': date,
                    'entry_price': price,
                    'quantity': quantity
                }
            elif signals.iloc[i] == -1 and position is not None:
                # SATIS
                profit = position['quantity'] * (price - position['entry_price'])
                commission_cost = position['quantity'] * price * self.commission
                trades.append({
                    'exit_date': date,
                    'exit_price': price,
                    'profit': profit - commission_cost,
                    'type': 'signal'
                })
                position = None

            # Portfoy guncelleme
            # ... (equity curve hesaplama)

        metrics = self._calculate_metrics(portfolio, trades)
        return portfolio, trades, metrics
```

- [ ] Backtesting kurallari:
  - Komisyon: %0.1 (Binance spot)
  - Slippage: %0.05 (opsiyonel)
  - Baslangic sermayesi: $10,000
  - Pozisyon boyutu: Sermayenin %95'i (tek pozisyon)
  - Ayni anda tek pozisyon (long-only stratejisi)

### 4.3 Performans Metrikleri Hesaplama
**Sorumlu:** Mert Kerem
**Sure:** 1 saat (16:00-17:00)

- [ ] Asagidaki metriklerin hesaplanmasi:

```python
def calculate_metrics(portfolio, trades, risk_free_rate=0.02):
    """Tum performans metriklerini hesaplar."""
    returns = portfolio['total'].pct_change().dropna()

    metrics = {
        # Getiri Metrikleri
        'total_return': (portfolio['total'].iloc[-1] / portfolio['total'].iloc[0]) - 1,
        'annual_return': ((1 + total_return) ** (252 / len(returns))) - 1,
        'buy_hold_return': (data['Close'].iloc[-1] / data['Close'].iloc[0]) - 1,

        # Risk Metrikleri
        'sharpe_ratio': (returns.mean() - risk_free_rate/252) / returns.std() * np.sqrt(252),
        'sortino_ratio': (returns.mean() - risk_free_rate/252) / returns[returns < 0].std() * np.sqrt(252),
        'max_drawdown': calculate_max_drawdown(portfolio['total']),
        'volatility': returns.std() * np.sqrt(252),

        # Trade Metrikleri
        'total_trades': len(trades),
        'win_rate': sum(1 for t in trades if t['profit'] > 0) / max(len(trades), 1),
        'profit_factor': abs(sum(t['profit'] for t in trades if t['profit'] > 0)) /
                         max(abs(sum(t['profit'] for t in trades if t['profit'] < 0)), 1),
        'avg_profit': np.mean([t['profit'] for t in trades]) if trades else 0,
        'max_profit': max([t['profit'] for t in trades]) if trades else 0,
        'max_loss': min([t['profit'] for t in trades]) if trades else 0,
    }
    return metrics
```

### 4.4 GA ile Strateji Entegrasyonu
**Sorumlu:** Mert & Mert Kerem
**Sure:** 1 saat (17:00-18:00)

- [ ] GA fitness fonksiyonunun backtester ile baglantisi:

```python
def evaluate_individual(individual, train_data):
    """
    Bir bireyi (trading stratejisi parametrelerini) degerlendirir.
    GA'nin fitness fonksiyonu olarak kullanilir.
    """
    # Kromozomu parametrelere coz
    params = decode_chromosome(individual)

    # Gecerlilik kontrolu ve tamir
    params = repair_params(params)

    # Indikatörleri hesapla (bu parametrelerle)
    data_with_indicators = calculate_indicators_with_params(train_data, params)

    # Sinyalleri uret
    signals = generate_composite_signal(data_with_indicators, params)

    # Backtesting yap
    backtester = Backtester(initial_capital=10000)
    portfolio, trades, metrics = backtester.run(
        data_with_indicators, signals,
        params['stop_loss'], params['take_profit']
    )

    # Fitness degerini dondur
    # Minimum trade sayisi kontrolu
    if metrics['total_trades'] < 10:
        return (-999.0,)  # Ceza

    return (metrics['sharpe_ratio'],)
```

- [x] Özel GA motoruna fitness fonksiyonunun entegre edilmesi:
```python
# Özel GA motorunda fitness fonksiyonu dogrudan engine.py icinde cagirilir
# DEAP toolbox kullanilmadi, bunun yerine:
fitnesses = np.array([evaluate_individual(ind, train_data) for ind in population])
```

### 4.5 Benchmark Stratejileri
**Sorumlu:** Mert Kerem
**Sure:** 1 saat (18:00-19:00)

- [ ] Karsilastirma icin baseline stratejilerin implementasyonu:
  - **Buy & Hold:** Baslangicta al, sonunda sat
  - **Rastgele Strateji:** Rastgele alis/satis sinyalleri
  - **Basit SMA Crossover:** Sabit parametreli (50/200 gunluk)
  - **Basit RSI Stratejisi:** Sabit esikli (30/70)
- [ ] GA optimize stratejisinin bu benchmark'larla karsilastirilmasi

---

## Entegrasyon Diyagrami

```
GA Motoru (Faz 3)
    |
    | Birey (kromozom) = [rsi_period, rsi_oversold, ..., weight_sma]
    v
Kromozom Cozucu (decode_chromosome)
    |
    | params = {rsi_period: 14, rsi_oversold: 30, ...}
    v
Indikatör Hesaplama (Faz 2)
    |
    | data + indikatörler (bu parametrelerle hesaplanmis)
    v
Sinyal Ureteci (generate_composite_signal)
    |
    | signals = [0, 0, 1, 0, -1, 0, 1, ...]
    v
Backtester (run)
    |
    | portfolio, trades, metrics
    v
Fitness Hesaplama
    |
    | fitness = (sharpe_ratio,)
    v
GA Motoru'na geri doner --> Secim, Caprazlama, Mutasyon --> Yeni Nesil
```

---

## Ciktilar

| Cikti | Dosya |
|-------|-------|
| Sinyal Uretim Modulu | `src/strategy/signals.py` |
| Backtesting Modulu | `src/strategy/backtest.py` |
| Benchmark Stratejileri | `src/strategy/benchmarks.py` |
| Entegrasyon Kodu | `src/ga/fitness.py` (guncellenmis) |

---

## Basari Kriterleri

- [ ] Her indikatör icin bagimsiz sinyal uretimi calisiyor
- [ ] Bilesik sinyal GA parametreleriyle dogru calisiyor
- [ ] Backtester komisyon ve stop-loss/take-profit'i dogru hesapliyor
- [ ] Performans metrikleri mantikli degerler uretiyor
- [ ] GA fitness fonksiyonu backtester ile entegre calisiyor
- [ ] Benchmark stratejileri implement edilmis
- [ ] Tum pipeline uctan uca calisiyor (GA -> Sinyal -> Backtest -> Fitness)

---

## Notlar

- Backtesting'de **look-ahead bias**'tan kacinilmali (gelecek veriyi kullanmama)
- Komisyon ve slippage gercekci olmali, aksi takdirde sonuclar yaniltici olur
- Long-only strateji ile baslanacak, short satista ek karmasiklik var
- Fitness degerlendirme hizli olmali cunku her nesilde populasyon_boyutu kadar calisacak
- Trade sayisi cok dusukse strateji anlamli degil, penalizasyon sart
