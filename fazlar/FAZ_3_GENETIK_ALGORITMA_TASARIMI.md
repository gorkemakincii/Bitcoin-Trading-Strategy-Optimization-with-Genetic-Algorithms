# Faz 3: Genetik Algoritma Tasarimi ve Gelistirme

**Sure:** Gun 2 - Sabah (1 Haziran Pazar)
**Tarih:** 1 Haziran 2026
**Durum:** Tamamlandi
**Sorumlu:** Mert (Ana), Yigit (Destek)
**Bagimlilik:** Faz 2 tamamlanmis olmali (onceki gun)

---

## Hedef

Trading stratejisi parametrelerini optimize edecek Genetik Algoritma motorunu tasarlamak ve tamamen özel (custom) bir GA motoru olarak sadece numpy/pandas kullanarak implement etmek. Bu faz projenin cekirdek bilesenidir.

---

## Gorevler

### 3.1 Kromozom Yapisi Tasarimi
**Sorumlu:** Mert
**Sure:** 1 saat (09:00-10:00)

- [x] `src/ga/chromosome.py` modulunun gelistirilmesi
- [x] Kromozom (birey) yapisinin tanimlanmasi:

#### Kromozom Genleri

Her birey asagidaki parametreleri temsil eden bir vektörden olusur:

| Gen | Aciklama | Tip | Aralik |
|-----|----------|-----|--------|
| `rsi_period` | RSI hesaplama periyodu | int | [7, 30] |
| `rsi_oversold` | RSI asiri satim esigi | float | [20, 40] |
| `rsi_overbought` | RSI asiri alim esigi | float | [60, 80] |
| `macd_fast` | MACD hizli periyot | int | [8, 16] |
| `macd_slow` | MACD yavas periyot | int | [20, 30] |
| `macd_signal` | MACD sinyal periyodu | int | [5, 12] |
| `bb_period` | Bollinger Band periyodu | int | [15, 30] |
| `bb_std` | Bollinger Band std carpi | float | [1.5, 3.0] |
| `sma_short` | Kisa SMA periyodu | int | [5, 20] |
| `sma_long` | Uzun SMA periyodu | int | [30, 100] |
| `stop_loss` | Zarar durdur yuzdesi | float | [0.01, 0.10] |
| `take_profit` | Kar al yuzdesi | float | [0.02, 0.20] |
| `weight_rsi` | RSI sinyal agirligi | float | [0, 1] |
| `weight_macd` | MACD sinyal agirligi | float | [0, 1] |
| `weight_bb` | BB sinyal agirligi | float | [0, 1] |
| `weight_sma` | SMA sinyal agirligi | float | [0, 1] |

**Toplam Kromozom Uzunlugu:** 16 gen

```python
# Özel kromozom yapisi (custom GA motoru - DEAP kullanilmadi)
from src.ga.chromosome import create_individual, decode_chromosome
from src.ga.operators import blend_crossover, gaussian_mutation, tournament_selection
from src.ga.fitness import evaluate_individual
from src.ga.engine import run_ga
import numpy as np

# Gen sinirlarini tanimla
GENE_BOUNDS = [
    (7, 30),       # rsi_period (int)
    (20.0, 40.0),  # rsi_oversold (float)
    (60.0, 80.0),  # rsi_overbought (float)
    (8, 16),       # macd_fast (int)
    (20, 30),      # macd_slow (int)
    (5, 12),       # macd_signal (int)
    (15, 30),      # bb_period (int)
    (1.5, 3.0),    # bb_std (float)
    (5, 20),       # sma_short (int)
    (30, 100),     # sma_long (int)
    (0.01, 0.10),  # stop_loss (float)
    (0.02, 0.20),  # take_profit (float)
    (0.0, 1.0),    # weight_rsi (float)
    (0.0, 1.0),    # weight_macd (float)
    (0.0, 1.0),    # weight_bb (float)
    (0.0, 1.0),    # weight_sma (float)
]

def create_individual():
    """Rastgele bir birey olusturur (numpy array olarak)."""
    individual = np.empty(len(GENE_BOUNDS))
    for i, (low, high) in enumerate(GENE_BOUNDS):
        if isinstance(low, int) and isinstance(high, int):
            individual[i] = np.random.randint(low, high + 1)
        else:
            individual[i] = np.random.uniform(low, high)
    return individual
```

- [x] Gen kısıtlamalari (constraints):
  - `macd_fast < macd_slow` olmali
  - `sma_short < sma_long` olmali
  - `rsi_oversold < rsi_overbought` olmali
  - Agirliklar normalize edilebilir (toplam = 1)

### 3.2 Fitness Fonksiyonu Tasarimi
**Sorumlu:** Mert & Mert Kerem
**Sure:** 1.5 saat (10:00-11:30)

- [x] `src/ga/fitness.py` modulunun gelistirilmesi
- [x] Fitness fonksiyonunun tasarimi:

#### Tek Amacli (Single-Objective) Fitness
```python
def fitness_single(individual, data):
    """
    Tek bir skor uretir: Sharpe Ratio bazli.
    Sharpe Ratio = (Ortalama Getiri - Risksiz Getiri) / Getiri Std Sapmasi
    """
    strategy_params = decode_chromosome(individual)
    signals = generate_signals(data, strategy_params)
    portfolio = backtest(data, signals, strategy_params)

    sharpe_ratio = calculate_sharpe_ratio(portfolio)
    return (sharpe_ratio,)
```

#### Cok Amacli (Multi-Objective) Fitness
```python
def fitness_multi(individual, data):
    """
    Birden fazla metrik ile degerlendirme.
    Maximize: Sharpe Ratio, Total Return
    Minimize: Max Drawdown
    """
    strategy_params = decode_chromosome(individual)
    signals = generate_signals(data, strategy_params)
    portfolio = backtest(data, signals, strategy_params)

    sharpe = calculate_sharpe_ratio(portfolio)
    total_return = calculate_total_return(portfolio)
    max_drawdown = calculate_max_drawdown(portfolio)

    return (sharpe, total_return, max_drawdown)
```

- [x] Performans metrikleri:

| Metrik | Formul | Aciklama |
|--------|--------|----------|
| Total Return | (Son Deger - Baslangic) / Baslangic | Toplam getiri yuzdesi |
| Sharpe Ratio | (R_p - R_f) / sigma_p | Risk-ayarli getiri |
| Max Drawdown | max(Peak - Trough) / Peak | En buyuk dusus |
| Win Rate | Karli Trade / Toplam Trade | Kazanma orani |
| Profit Factor | Brut Kar / Brut Zarar | Kar faktoru |
| Sortino Ratio | (R_p - R_f) / sigma_down | Asagi yonlu risk-ayarli getiri |

- [x] Fitness fonksiyonunda **penalizasyon** mekanizmasi:
  - Cok az trade yapan stratejiler icin ceza (min. trade sayisi)
  - Kisitlamalari ihlal eden bireylere dusuk fitness degeri

### 3.3 GA Operatörleri
**Sorumlu:** Mert
**Sure:** 1 saat (10:00-11:00, Fitness ile paralel)

- [x] `src/ga/operators.py` modulunun gelistirilmesi

#### Secim (Selection) Operatörleri
- [x] **Turnuva Secimi (Tournament Selection)**
  - Turnuva boyutu: 3-5
  - En basit ve etkili yontem
- [x] **Elitizm**
  - Her nesilde en iyi N bireyin korunmasi
  - Elit boyutu: populasyonun %5-10'u

```python
# Özel turnuva secimi (src/ga/operators.py icinde)
def tournament_selection(population, fitnesses, k=3):
    """Turnuva secimi ile ebeveyn secer."""
    selected_indices = np.random.choice(len(population), size=k, replace=False)
    best_idx = selected_indices[np.argmax(fitnesses[selected_indices])]
    return population[best_idx].copy()
```

#### Caprazlama (Crossover) Operatörleri
- [x] **Blend Crossover (BLX-alpha)**
  - Surekli degerler icin en uygun
  - alpha = 0.5
- [ ] **Simulated Binary Crossover (SBX)**
  - Alternatif olarak denenebilir
- [x] Caprazlama orani: 0.7 - 0.9

```python
def blend_crossover(ind1, ind2, alpha=0.5):
    """BLX-alpha caprazlama."""
    for i in range(len(ind1)):
        low = min(ind1[i], ind2[i])
        high = max(ind1[i], ind2[i])
        range_val = high - low
        ind1[i] = random.uniform(low - alpha * range_val, high + alpha * range_val)
        ind2[i] = random.uniform(low - alpha * range_val, high + alpha * range_val)
        # Gen sinirlarini kontrol et
        ind1[i] = max(GENE_BOUNDS[i][0], min(GENE_BOUNDS[i][1], ind1[i]))
        ind2[i] = max(GENE_BOUNDS[i][0], min(GENE_BOUNDS[i][1], ind2[i]))
    return ind1, ind2
```

#### Mutasyon (Mutation) Operatörleri
- [x] **Gaussian Mutasyon**
  - Her gen icin farkli sigma degeri
  - Mutasyon orani: 0.1 - 0.3
- [x] **Uniform Mutasyon**
  - Gen sinirlar icinde rastgele deger atama
- [ ] Adaptif mutasyon oranı (nesil ilerledikce azaltma) - opsiyonel

```python
def gaussian_mutation(individual, mu=0, sigma_ratio=0.1, indpb=0.2):
    """Gaussian mutasyon."""
    for i in range(len(individual)):
        if random.random() < indpb:
            low, high = GENE_BOUNDS[i]
            sigma = (high - low) * sigma_ratio
            individual[i] += random.gauss(mu, sigma)
            individual[i] = max(low, min(high, individual[i]))
            # Integer genleri yuvarlama
            if isinstance(low, int):
                individual[i] = int(round(individual[i]))
    return (individual,)
```

### 3.4 GA Motoru (Engine)
**Sorumlu:** Mert & Yigit
**Sure:** 1.5 saat (11:00-12:30)

- [x] `src/ga/engine.py` modulunun gelistirilmesi
- [x] Ana GA dongusu:

```python
# Özel GA motoru (src/ga/engine.py) - DEAP yerine tamamen custom implementasyon
import numpy as np
from src.ga.chromosome import create_individual, decode_chromosome
from src.ga.operators import tournament_selection, blend_crossover, gaussian_mutation
from src.ga.fitness import evaluate_individual

def run_ga(data, config):
    """
    Özel Genetik Algoritma ana dongusunu calistirir.
    DEAP kullanilmadan, sadece numpy ile yazilmistir.

    Parametreler:
    - data: Islenmis BTC veri seti
    - config: GA konfigurasyonu (populasyon boyutu, nesil sayisi, vb.)

    Donus:
    - best_individual: En iyi birey
    - history: Nesil bazli istatistikler (dict)
    - hall_of_fame: En iyi N birey (numpy array)
    """
    # Populasyonu baslat
    population = np.array([create_individual() for _ in range(config.POPULATION_SIZE)])

    # Nesil bazli istatistikleri sakla
    history = {'avg': [], 'min': [], 'max': [], 'std': []}
    hall_of_fame = []

    for gen in range(config.NUM_GENERATIONS):
        # Fitness degerlendirme
        fitnesses = np.array([evaluate_individual(ind, data) for ind in population])

        # Istatistikleri kaydet
        history['avg'].append(np.mean(fitnesses))
        history['min'].append(np.min(fitnesses))
        history['max'].append(np.max(fitnesses))
        history['std'].append(np.std(fitnesses))

        # Elitizm: en iyi bireyleri koru
        elite_idx = np.argsort(fitnesses)[-config.ELITE_SIZE:]
        elites = population[elite_idx].copy()

        # Yeni nesil olustur
        new_population = list(elites)
        while len(new_population) < config.POPULATION_SIZE:
            parent1 = tournament_selection(population, fitnesses, k=config.TOURNAMENT_SIZE)
            parent2 = tournament_selection(population, fitnesses, k=config.TOURNAMENT_SIZE)
            child1, child2 = blend_crossover(parent1, parent2, alpha=0.5)
            child1 = gaussian_mutation(child1)
            child2 = gaussian_mutation(child2)
            new_population.extend([child1, child2])

        population = np.array(new_population[:config.POPULATION_SIZE])

    # En iyi bireyi dondur
    final_fitnesses = np.array([evaluate_individual(ind, data) for ind in population])
    best_idx = np.argmax(final_fitnesses)
    return population[best_idx], history, population[np.argsort(final_fitnesses)[-config.HOF_SIZE:]]
```

- [ ] GA Konfigurasyonu:

| Parametre | Varsayilan Deger | Aciklama |
|-----------|-----------------|----------|
| `POPULATION_SIZE` | 100 | Populasyon boyutu |
| `NUM_GENERATIONS` | 50-100 | Nesil sayisi |
| `CROSSOVER_RATE` | 0.8 | Caprazlama olasiligi |
| `MUTATION_RATE` | 0.2 | Mutasyon olasiligi |
| `TOURNAMENT_SIZE` | 3 | Turnuva secim boyutu |
| `ELITE_SIZE` | 5 | Elit birey sayisi |
| `HOF_SIZE` | 10 | Hall of Fame boyutu |

### 3.5 Kisit Yonetimi ve Tamir Fonksiyonu
**Sorumlu:** Mert
**Sure:** 30 dk (12:00-12:30, GA Motoru icinde)

- [x] Gecersiz bireylerin tamir edilmesi:
```python
def repair_individual(individual):
    """Kisitlamalari ihlal eden genleri duzeltir."""
    # macd_fast < macd_slow olmali
    if individual[3] >= individual[4]:
        individual[3], individual[4] = min(individual[3], individual[4]), max(individual[3], individual[4])

    # sma_short < sma_long olmali
    if individual[8] >= individual[9]:
        individual[8], individual[9] = min(individual[8], individual[9]), max(individual[8], individual[9])

    # rsi_oversold < rsi_overbought olmali
    if individual[1] >= individual[2]:
        individual[1], individual[2] = min(individual[1], individual[2]), max(individual[1], individual[2])

    # Agirlik normalizasyonu (opsiyonel)
    weight_sum = sum(individual[12:16])
    if weight_sum > 0:
        for i in range(12, 16):
            individual[i] /= weight_sum

    return individual
```

### 3.6 Birim Testleri
**Sorumlu:** Yigit
**Sure:** 30 dk (12:00-12:30, Kisit yonetimi ile paralel)

- [ ] `tests/test_ga.py` dosyasinin gelistirilmesi
- [ ] Test edilecek bilesenleri:
  - Birey olusturma (gen sinirlari icinde mi?)
  - Caprazlama sonrasi kisitlamalar saglanıyor mu?
  - Mutasyon sonrasi gen sinirlari korunuyor mu?
  - Fitness fonksiyonu dogru hesapliyor mu?
  - Tamir fonksiyonu calisiyor mu?
  - GA dongusu basi bos calisabiliyor mu (kucuk ornekle)?

---

## Ciktilar

| Cikti | Dosya |
|-------|-------|
| Kromozom Modulu | `src/ga/chromosome.py` |
| Fitness Modulu | `src/ga/fitness.py` |
| Operatörler Modulu | `src/ga/operators.py` |
| GA Motoru | `src/ga/engine.py` |
| Birim Testleri | `tests/test_ga.py` |

---

## Basari Kriterleri

- [x] Kromozom yapisi tum parametreleri temsil ediyor
- [x] Fitness fonksiyonu anlamli skorlar uretiyor
- [x] GA operatörleri gen sinirlarini koruyor
- [x] Kisit ihlalleri otomatik tamir ediliyor
- [x] GA motoru kucuk bir ornekle sorunsuz calisiyor (10 birey, 5 nesil)
- [x] Nesil bazli istatistikler (avg, min, max fitness) loglanıyor
- [x] Tum birim testleri gecik

---

## Notlar

- Özel GA motoru (numpy tabanli) hem tek amacli hem cok amacli optimizasyonu destekleyecek sekilde tasarlandi (DEAP kullanilmadi)
- Ilk asama icin tek amacli (Sharpe Ratio) ile baslanmasi tavsiye edilir, sonra multi-objective denenebilir
- Fitness degerlendirme en cok zaman alan adim olacak (her birey icin backtesting yapilmasi gerekiyor)
- Populasyon boyutu ve nesil sayisi hesaplama suresi ile dogrudan orantili - dengeyi iyi kurmak lazim
- Elitizm kullanmak yakinsama hizini arttirir ve en iyi cozumun kaybolmasini engeller
