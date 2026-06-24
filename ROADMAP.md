# Genetik Algoritmalar ile Bitcoin Fiyat Tahmini ve Trading Stratejisi Optimizasyonu

## Proje Yol Haritasi (Roadmap)

**Ders:** Yapay Zeka Yontemleri (3+0) - 2025-2026 Bahar Donemi
**Universite:** Ege Universitesi - Bilgisayar Muhendisligi Bolumu
**Proje:** Proje 2 - Optimizasyon Alaninda Yapay Zeka Projesi

---

## Takim Uyeleri

| Uye | Rol |
|-----|-----|
| **Mert** | Genetik Algoritma Cekirdek Gelistirme & Fitness Fonksiyonu Tasarimi |
| **Yigit** | Veri Toplama, On Isleme & Feature Engineering |
| **Mert Kerem** | Trading Stratejisi Tasarimi & Backtesting Altyapisi |
| **Gorkem Ege** | Deneysel Calismalar, Gorselletirme & Rapor/Dokumantasyon |

---

## Proje Ozeti

Bu projede, Genetik Algoritmalar (GA) kullanilarak Bitcoin (BTC) fiyat hareketlerini tahmin eden ve trading stratejilerini optimize eden bir sistem gelistirilecektir. Sistem, tarihsel BTC fiyat verilerinden teknik indikatörler cikararak, GA ile en uygun trading parametrelerini (aliş/satiş sinyalleri, stop-loss/take-profit seviyeleri, indikatör agirliklarini) bulmayi hedefler. Proje, klasik makine ogrenimi yontemlerinden farkli olarak, meta-sezgisel optimizasyon yaklasimi kullanarak trading stratejilerini evrimsel sureclere tabi tutacaktir.

---

## Faz Ozeti ve Takvim

**Toplam Sure:** 4 gun (1.5 hafta)

| Gun | Tarih | Fazlar | Sorumlular |
|-----|-------|--------|------------|
| **Gun 1 (Cumartesi)** | 31 Mayis 2026 | Faz 1 + Faz 2 | Tum Takim |
| **Gun 2 (Pazar)** | 1 Haziran 2026 | Faz 3 + Faz 4 | Tum Takim |
| **Gun 3 (Pazartesi)** | 2 Haziran 2026 | Faz 5 | Tum Takim |
| **Gun 4 (Gelecek Hafta)** | 5-6 Haziran 2026 | Faz 6 | Tum Takim |

| Faz | Baslik | Gun | Durum |
|-----|--------|-----|-------|
| [Faz 1](./fazlar/FAZ_1_ARASTIRMA_VE_PLANLAMA.md) | Arastirma ve Planlama | Gun 1 - Sabah | Tamamlandi |
| [Faz 2](./fazlar/FAZ_2_VERI_TOPLAMA_VE_ON_ISLEME.md) | Veri Toplama ve On Isleme | Gun 1 - Ogleden Sonra | Tamamlandi |
| [Faz 3](./fazlar/FAZ_3_GENETIK_ALGORITMA_TASARIMI.md) | Genetik Algoritma Tasarimi ve Gelistirme | Gun 2 - Sabah | Tamamlandi |
| [Faz 4](./fazlar/FAZ_4_TRADING_STRATEJISI_VE_ENTEGRASYON.md) | Trading Stratejisi ve Entegrasyon | Gun 2 - Ogleden Sonra | Tamamlandi |
| [Faz 5](./fazlar/FAZ_5_DENEYSEL_CALISMALAR.md) | Deneysel Calismalar ve Optimizasyon | Gun 3 (Tam Gun) | Tamamlandi |
| [Faz 6](./fazlar/FAZ_6_RAPOR_VE_SUNUM.md) | Rapor Yazimi ve Demo Video | Gun 4 (Tam Gun) | Devam Ediyor |

---

## Mimari Genel Bakis

```
+--------------------------------------------------+
|              VERI KATMANI                         |
|  BTC Tarihsel Veri (API) --> On Isleme -->        |
|  Teknik Indikatörler (RSI, MACD, BB, EMA, vb.)   |
+--------------------------------------------------+
                      |
                      v
+--------------------------------------------------+
|          GENETIK ALGORITMA MOTORU                 |
|  Kromozom: [ind_agirliklari, esik_degerleri,      |
|             stop_loss, take_profit, periyotlar]   |
|  Fitness: Toplam Kar / Sharpe Orani / Max DD      |
|  Operatörler: Secim, Caprazlama, Mutasyon         |
+--------------------------------------------------+
                      |
                      v
+--------------------------------------------------+
|          TRADING STRATEJI MOTORU                  |
|  Aliş/Satiş Sinyal Uretimi --> Backtesting -->    |
|  Performans Degerlendirme                         |
+--------------------------------------------------+
                      |
                      v
+--------------------------------------------------+
|          SONUC VE GORSELLESITIRME                  |
|  Kar/Zarar Grafikleri, Equity Curve,              |
|  Nesil Bazli Fitness Evrimi, Karsilastirma        |
+--------------------------------------------------+
```

---

## Teknoloji Yigini

| Kategori | Teknoloji |
|----------|-----------|
| **Programlama Dili** | Python 3.10+ |
| **Veri Toplama** | `yfinance` |
| **Veri Isleme** | `pandas`, `numpy` |
| **Teknik Analiz** | Ozel implementasyon (`pandas` / `numpy` ile sifirdan kodlandi) |
| **Genetik Algoritma** | Ozel GA motoru (`chromosome.py`, `operators.py`, `engine.py`, `fitness.py`) |
| **Gorsellesitirme** | `matplotlib` |
| **Backtesting** | Ozel vektorel backtesting motoru (`backtest.py`) |
| **Versiyon Kontrolu** | Git & GitHub |
| **Gelistirme Ortami** | VS Code / PyCharm |

---

## Teslimatlar

### Ara Rapor (Son ders haftasi - ders gununde)
- [x] Projenin mevcut durumunu gosteren kod
- [x] Ara rapor dokumani
- [x] Gelecekte yapilacak isler plani

### Final Teslimati
1. **Kaynak Kod** - Tum proje kodlari (GitHub reposu)
2. **Proje Raporu** - Template'e uygun, 5-13 sayfa (Ekler dahil)
3. **Demo Videosu** - Proje tanitimi, kod aciklamasi, is bolumu (5-10 dk)

---

## Rapor Bolumleri Eslestirmesi

| Rapor Bolumu | Ilgili Faz |
|--------------|------------|
| Problem Tanimi | Faz 1 |
| Arastirma | Faz 1 |
| Araclar, Yontemler ve Kutuphaneler | Faz 1, 2, 3 |
| Onerilen Yontem | Faz 3, 4 |
| Deneysel Calismalar | Faz 5 |
| Sonuc | Faz 5, 6 |
| Ek 1: Performans Iyilestirme | Faz 5 |
| Ek 2: Literatur Katkisi | Faz 1, 6 |
| Ek 3: Kullanilan Kaynaklar | Faz 1 |
| Ek 4: Is Paketleri ve Is Bolumu | Faz 6 |
| Arastirma Sorusu (Uretici YZ) | Faz 6 |

---

## Is Bolumu Detayi

### Mert - Genetik Algoritma Cekirdek Gelistirici
- GA motorunun tasarimi ve kodlanmasi (ozel implementasyon)
- Kromozom yapisinin belirlenmesi
- Fitness fonksiyonunun tasarimi ve implementasyonu
- Secim, caprazlama ve mutasyon operatörlerinin implementasyonu
- GA parametre ayarlama (populasyon boyutu, nesil sayisi, oranlar)

### Yigit - Veri Muhendisi
- BTC tarihsel veri toplama (API entegrasyonu)
- Veri temizleme ve normalizasyon
- Teknik indikatörlerin hesaplanmasi (RSI, MACD, Bollinger Bands, EMA, vb.)
- Feature engineering ve ozellik secimi
- Veri pipeline'inin olusturulmasi

### Mert Kerem - Trading Strateji Gelistirici
- Trading sinyal uretim mekanizmasi
- Alis/Satis kurallarinin tanimlanmasi
- Backtesting framework'unun gelistirilmesi
- Risk yonetimi (stop-loss, take-profit) implementasyonu
- Strateji performans metriklerinin hesaplanmasi

### Gorkem Ege - Analiz & Dokumantasyon
- Deneysel calismalarin yurutulmesi ve sonuclarin kaydedilmesi
- Performans karsilastirma tablolarinin olusturulmasi
- Gorsellesitirme (grafikler, tablolar, equity curve)
- Proje raporunun yazilmasi
- Demo videonun hazirlanmasi

---

## Risk ve Onlemler

| Risk | Onem | Onlem |
|------|------|-------|
| Yetersiz veri kalitesi | Yuksek | Birden fazla veri kaynagi kullanma, veri dogrulama |
| Overfitting | Yuksek | Train/Test ayirimi, walk-forward validation |
| GA'nin yakinsamamasi | Orta | Farkli operatör kombinasyonlari deneme, parametre ayarlama |
| Hesaplama suresi | Orta | Populasyon ve nesil sayisini optimize etme |
| Takim koordinasyonu | Dusuk | Gunluk senkronizasyon, GitHub issue tracking |

---

## Basarı Kriterleri

1. GA'nin nesiller boyunca fitness degerini tutarli sekilde iyilestirmesi
2. Optimize edilmis stratejinin buy-and-hold stratejisinden daha iyi performans gostermesi
3. Backtesting sonuclarinin anlamli kar/zarar metrikleri uretmesi
4. Farkli parametre kombinasyonlarinin karsilastirilmasi ve en iyisinin belirlenmesi
5. Raporun tum gereksinimleri karsilamasi ve profesyonel kalitede olmasi
