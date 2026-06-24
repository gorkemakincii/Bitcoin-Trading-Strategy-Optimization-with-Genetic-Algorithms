# Faz 5: Deneysel Calismalar ve Optimizasyon

**Sure:** Gun 3 - Tam Gun (2 Haziran Pazartesi)
**Tarih:** 2 Haziran 2026
**Durum:** Tamamlandi
**Sorumlu:** Gorkem Ege (Ana), Tum Takim (Destek)
**Bagimlilik:** Faz 3 ve Faz 4 tamamlanmis olmali (onceki gun)

---

## Hedef

GA motorunu farkli konfigurasyonlarla calıstirmak, sonuclari analiz etmek, performans iyilestirmeleri yapmak, gorsellesitirmeler olusturmak ve tum sonuclari rapor icin dokumante etmek.

---

## Gorevler

### 5.1 Temel Deney (Baseline Experiment)
**Sorumlu:** Gorkem Ege & Mert
**Sure:** 2 saat (09:00-11:00)

- [ ] GA'yi varsayilan parametrelerle calıstirma:

| Parametre | Deger |
|-----------|-------|
| Populasyon Boyutu | 100 |
| Nesil Sayisi | 50 |
| Caprazlama Orani | 0.8 |
| Mutasyon Orani | 0.2 |
| Turnuva Boyutu | 3 |
| Elit Boyutu | 5 |

- [ ] Train seti (2018-2023) uzerinde GA'yi calıstirma
- [ ] En iyi bireyin test seti (2024-2025) uzerinde degerlendirilmesi
- [ ] Sonuclarin kaydedilmesi:
  - En iyi birey parametreleri
  - Fitness evrimi (nesil bazli)
  - Train ve test performans metrikleri
  - Islem gecmisi (trade log)

### 5.2 Parametre Duyarlilik Analizi
**Sorumlu:** Gorkem Ege
**Sure:** 2 saat (11:00-13:00)

- [ ] Farkli GA parametreleriyle deneyler:

#### Deney 1: Populasyon Boyutu Etkisi
| Deney | Pop. Boyutu | Nesil | Diger |
|-------|-------------|-------|-------|
| 1a | 50 | 50 | Varsayilan |
| 1b | 100 | 50 | Varsayilan |
| 1c | 200 | 50 | Varsayilan |
| 1d | 300 | 50 | Varsayilan |

#### Deney 2: Nesil Sayisi Etkisi
| Deney | Pop. Boyutu | Nesil | Diger |
|-------|-------------|-------|-------|
| 2a | 100 | 25 | Varsayilan |
| 2b | 100 | 50 | Varsayilan |
| 2c | 100 | 100 | Varsayilan |
| 2d | 100 | 200 | Varsayilan |

#### Deney 3: Caprazlama ve Mutasyon Orani
| Deney | CX Oran | MUT Oran |
|-------|---------|----------|
| 3a | 0.6 | 0.1 |
| 3b | 0.8 | 0.2 |
| 3c | 0.9 | 0.3 |
| 3d | 0.7 | 0.4 |

#### Deney 4: Secim Yontemi
| Deney | Secim Yontemi |
|-------|---------------|
| 4a | Turnuva (k=3) |
| 4b | Turnuva (k=5) |
| 4c | Rulet Tekerlegi |

- [ ] Her deney icin 3 bagimsiz calistirma (farkli random seed)
- [ ] Sonuclarin ortalama ve standart sapma olarak raporlanmasi

### 5.3 Performans Iyilestirme Calismalari
**Sorumlu:** Mert & Mert Kerem
**Sure:** 1.5 saat (11:00-12:30, Parametre analizi ile paralel)

- [ ] Iyilestirme deneyleri:

#### Iyilestirme 1: Cok Amacli Optimizasyon (NSGA-II)
- Sharpe Ratio + Total Return + Min Drawdown
- Pareto optimal cozumlerin bulunmasi
- Tek amacli ile karsilastirma

#### Iyilestirme 2: Adaptif Operatörler
- Nesil ilerledikce mutasyon oranini azaltma
- Erken nesillerde cesitlilik, gec nesillerde hassas ayar

#### Iyilestirme 3: Farkli Caprazlama Yontemleri
- BLX-alpha vs SBX vs Uniform caprazlama karsilastirmasi

#### Iyilestirme 4: Farkli Fitness Fonksiyonlari
- Sadece Sharpe Ratio
- Sadece Total Return
- Sharpe + Drawdown penalizasyonu
- Karli trade sayisi odakli

- [ ] Her iyilestirmenin etkisinin olculmesi ve karsilastirilmasi

### 5.4 Walk-Forward Validation
**Sorumlu:** Yigit & Mert Kerem
**Sure:** 1.5 saat (13:00-14:30)

- [ ] Walk-forward validation ile overfitting kontrolu:

```
Donem 1: Train [2018-2020] --> Test [2021]
Donem 2: Train [2018-2021] --> Test [2022]
Donem 3: Train [2018-2022] --> Test [2023]
Donem 4: Train [2018-2023] --> Test [2024]
Donem 5: Train [2018-2024] --> Test [2025]
```

- [ ] Her donemde GA'yi bagimsiz calıstirma
- [ ] Out-of-sample performansin raporlanmasi
- [ ] Overfitting gostergelerinin tespiti (train vs test performans farki)

### 5.5 Benchmark Karsilastirmasi
**Sorumlu:** Mert Kerem
**Sure:** 1 saat (14:30-15:30)

- [ ] GA optimize stratejisi vs benchmark stratejileri karsilastirmasi:

| Strateji | Total Return | Sharpe | Max DD | Win Rate | Trade # |
|----------|-------------|--------|--------|----------|---------|
| GA Optimize | ? | ? | ? | ? | ? |
| Buy & Hold | ? | ? | ? | ? | 1 |
| SMA 50/200 | ? | ? | ? | ? | ? |
| RSI 30/70 | ? | ? | ? | ? | ? |
| Rastgele | ? | ? | ? | ? | ? |

### 5.6 Gorsellesitirme
**Sorumlu:** Gorkem Ege
**Sure:** 2 saat (15:30-17:30)

- [ ] `src/visualization/plots.py` modulunun gelistirilmesi
- [ ] Olusturulacak grafikler:

#### 1. Fitness Evrimi Grafigi
- X ekseni: Nesil numarasi
- Y ekseni: Fitness degeri (avg, min, max)
- Yakinsama davranisini gosterir

#### 2. Equity Curve (Portfoy Degeri)
- GA strateji vs Buy & Hold
- Zaman icinde portfoy degerinin degisimi

#### 3. Alis/Satis Sinyalleri Grafigi
- BTC fiyat grafigi uzerinde alis (yesil) ve satis (kirmizi) noktaları
- Indikatörlerin alt grafikleri

#### 4. Drawdown Grafigi
- Zaman icinde portfoyun tepe noktasindan dusus yuzdesi

#### 5. Parametre Duyarlilik Grafikleri
- Populasyon boyutu vs en iyi fitness
- Nesil sayisi vs en iyi fitness
- CX/MUT orani vs en iyi fitness

#### 6. Pareto Cephesi (Multi-Objective icin)
- Sharpe Ratio vs Total Return vs Max Drawdown

#### 7. Trade Dagilim Grafigi
- Kar/zarar dagilimi (histogram)
- Trade sureleri dagilimi

#### 8. Korelasyon Matrisi
- Optimize edilmis parametreler arasi korelasyon (Hall of Fame bireylerinden)

```python
# Ornek gorsellesitirme kodu
import matplotlib.pyplot as plt

def plot_fitness_evolution(logbook):
    """Nesil bazli fitness evrimi grafigi."""
    gen = logbook.select("gen")
    avg = logbook.select("avg")
    max_val = logbook.select("max")
    min_val = logbook.select("min")

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(gen, avg, label='Ortalama Fitness', color='blue')
    ax.plot(gen, max_val, label='En Iyi Fitness', color='green')
    ax.plot(gen, min_val, label='En Kotu Fitness', color='red')
    ax.fill_between(gen, min_val, max_val, alpha=0.1)
    ax.set_xlabel('Nesil')
    ax.set_ylabel('Fitness (Sharpe Ratio)')
    ax.set_title('GA Fitness Evrimi')
    ax.legend()
    plt.savefig('results/figures/fitness_evolution.png', dpi=300, bbox_inches='tight')

def plot_equity_curve(portfolio_ga, portfolio_bh, data):
    """Portfoy degeri karsilastirma grafigi."""
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(portfolio_ga.index, portfolio_ga['total'], label='GA Stratejisi', color='blue')
    ax.plot(portfolio_bh.index, portfolio_bh['total'], label='Buy & Hold', color='orange')
    ax.set_xlabel('Tarih')
    ax.set_ylabel('Portfoy Degeri ($)')
    ax.set_title('GA Stratejisi vs Buy & Hold')
    ax.legend()
    plt.savefig('results/figures/equity_curve.png', dpi=300, bbox_inches='tight')
```

### 5.7 Sonuc Tablolari
**Sorumlu:** Gorkem Ege
**Sure:** 1 saat (17:30-18:30)

- [ ] Asagidaki sonuc tablolarinin olusturulmasi:

#### Tablo 1: Temel Deney Sonuclari
- Train ve test performans metrikleri

#### Tablo 2: Parametre Duyarlilik Analizi
- Her deney konfigurasyonu icin sonuclar

#### Tablo 3: Performans Iyilestirme Karsilastirmasi
- Her iyilestirme yonteminin etkisi

#### Tablo 4: Walk-Forward Validation Sonuclari
- Donem bazli train/test performanslari

#### Tablo 5: Benchmark Karsilastirmasi
- Tum stratejilerin yan yana karsilastirmasi

#### Tablo 6: En Iyi Birey Parametreleri
- GA'nin buldugu optimal parametre degerleri

- [ ] Tablolarin hem CSV hem LaTeX/Markdown formatinda kaydedilmesi

---

## Sonuc Kayit Yapisi

```
results/
├── figures/
│   ├── fitness_evolution.png
│   ├── equity_curve.png
│   ├── signals_on_price.png
│   ├── drawdown.png
│   ├── param_sensitivity_population.png
│   ├── param_sensitivity_generations.png
│   ├── trade_distribution.png
│   └── correlation_matrix.png
├── tables/
│   ├── baseline_results.csv
│   ├── parameter_sensitivity.csv
│   ├── improvement_comparison.csv
│   ├── walkforward_results.csv
│   ├── benchmark_comparison.csv
│   └── best_individual_params.csv
└── logs/
    ├── ga_run_1.log
    ├── ga_run_2.log
    └── ...
```

---

## Ciktilar

| Cikti | Aciklama |
|-------|----------|
| Deney Sonuclari | Tum deneylerin metrikleri (CSV) |
| Grafikler | En az 8 farkli gorsellesitirme (PNG) |
| Analiz Notebook'u | `notebooks/experiments.ipynb` |
| Gorsellesitirme Modulu | `src/visualization/plots.py` |

---

## Basari Kriterleri

- [ ] En az 15 farkli deney konfigurasyonu calistirilmis
- [ ] Her deney en az 3 kez tekrarlanmis (farkli seed)
- [ ] GA optimize stratejisi buy-and-hold'dan daha iyi Sharpe Ratio uretmis (ideal)
- [ ] Walk-forward validation yapilmis, overfitting analiz edilmis
- [ ] En az 6 farkli grafik/gorsellesitirme olusturulmus
- [ ] Tum sonuclar tablolarda dokumante edilmis
- [ ] Performans iyilestirme calismalari yapilmis ve etkileri olculmus

---

## Notlar

- Her deney rastgele tohum (random seed) ile calistirilmali ki sonuclar tekrarlanabilir olsun
- Deneylerin calistırılma suresi uzun olabilir - paralel calistirma veya gece calistirma dusunulebilir
- Overfitting en buyuk risk: train seti performansi cok iyi ama test seti kotu ise problem var
- Sonuclar ne olursa olsun (iyi veya kotu) raporlanmali - negatif sonuclar da degerli
- "GA neden ise yarar/yaramaz" sorusuna cevap verebilecek derinlikte analiz yapilmali
