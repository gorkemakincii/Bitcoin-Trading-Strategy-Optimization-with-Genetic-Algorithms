# Faz 6: Rapor Yazimi ve Demo Video

**Sure:** Gun 4 - Tam Gun (5-6 Haziran, Gelecek Hafta)
**Tarih:** 5-6 Haziran 2026
**Durum:** Devam Ediyor
**Sorumlu:** Gorkem Ege (Ana), Tum Takim
**Bagimlilik:** Faz 5 tamamlanmis olmali (onceki hafta)

---

## Hedef

Proje raporunu template'e uygun sekilde yazmak, demo videoyu hazirlamak ve tum teslimatları tamamlamak.

---

## Gorevler

### 6.1 Rapor Yazimi
**Sorumlu:** Gorkem Ege (koordinatör), Ilgili faz sorumlulari (icerik)

#### Kapak Sayfasi
**Sorumlu:** Gorkem Ege
- [ ] Ders adi, yil ve donem, proje numarasi (2)
- [ ] Proje konusu: "Genetik Algoritmalar ile Bitcoin Fiyat Tahmini ve Trading Stratejisi Optimizasyonu"
- [ ] Ogrenci numaralari ve isimler:
  - Mert
  - Yigit
  - Mert Kerem
  - Gorkem Ege

#### Bolum 1: Problem Tanimi (1 sayfa)
**Sorumlu:** Mert
- [x] Bitcoin fiyat tahmini ve trading stratejisi optimizasyonu probleminin tanimi
- [x] Neden bu problem onemli?
- [x] GA'nin bu probleme uygunlugu
- [x] Projenin amaci ve kapsami

**Mert notu:** Bu bolum icin hazir metin `docs/mert_rapor_notlari.md` dosyasina eklendi.

#### Bolum 2: Arastirma - On Calisma (1 sayfa)
**Sorumlu:** Gorkem Ege
- [ ] Ders materyallerinden ilgili konularin ozeti (optimizasyon, GA)
- [ ] Internet'ten incelenen kaynaklar ve linkler
- [ ] Okunan makalelerden ogrenilen bilgiler
- [ ] Izlenen videolar ve elde edilen bilgiler
- [ ] Hocanin BTC makalesi hakkinda notlar

#### Bolum 3: Araclar, Yontemler ve Kutuphaneler (0.5 sayfa)
**Sorumlu:** Yigit
- [ ] Kullanilan yapay zeka yontemi: Genetik Algoritmalar
- [ ] Gelistirme ortami: Python 3.10+, VS Code/PyCharm
- [ ] Kutuphaneler ve versiyonlar:
  - Özel GA Motoru (custom, numpy/pandas tabanli - DEAP kullanilmadi)
  - pandas, numpy (veri isleme, teknik indikatörler ve vektörel hesaplama)
  - yfinance (veri toplama)
  - matplotlib (gorsellesitirme)

#### Bolum 4: Onerilen Yontem (1.5 sayfa)
**Sorumlu:** Mert & Mert Kerem
- [x] GA yapisinin aciklamasi:
  - Kromozom temsili ve gen yapisi
  - Fitness fonksiyonu
  - Secim, caprazlama, mutasyon operatörleri
- [ ] Trading strateji yapisinin aciklamasi:
  - Indikatör bazli sinyal uretimi
  - Agirlikli bilesik sinyal
  - Backtesting mekanizmasi
- [ ] Sistem diyagrami/akis semasi (mimari sema)
- [ ] Algoritma pseudocode'u

```
Onerilen Yontem Diyagrami:

[BTC Veri] -> [Teknik Indikatörler] -> [GA Optimizasyonu] -> [Optimal Parametreler]
                                              |
                                    [Fitness Degerlendirme]
                                              |
                                    [Trading Sinyalleri] -> [Backtesting] -> [Performans]
```

#### Bolum 5: Deneysel Calismalar (2-3 sayfa)
**Sorumlu:** Gorkem Ege & Yigit
- [ ] Veri setinin tanitimi:
  - Kaynak ve link
  - Ozellikler (feature sayisi, tarih araligi, veri sayisi)
- [ ] Deney ayarlari:
  - GA parametreleri
  - Train/test ayirimi
  - Walk-forward validation
- [ ] Sonuc tablolari:
  - Temel deney sonuclari
  - Parametre duyarlilik analizi
  - Benchmark karsilastirmasi
- [ ] Grafikler:
  - Fitness evrimi
  - Equity curve
  - Alis/satis sinyalleri
- [ ] Sonuc analizi ve yorumlar

#### Bolum 6: Sonuc (0.5-1 sayfa)
**Sorumlu:** Tum Takim (herkes 1 paragraf)
- [ ] Projenin basarisinin degerlendirmesi
- [ ] Gercek hayatta kullanilabilirligi ve uygulama alanlari
- [ ] Insanlara/topluma faydasi (ekonomik etki, karar destek)
- [ ] Her takim uyesinin ogrendikleri ve kazandiklari deneyimler (her kisi kendi paragrafini yazar)

#### Referanslar
**Sorumlu:** Gorkem Ege
- [ ] En az 10 kaynak (makale, kitap, web sitesi)
- [ ] APA veya IEEE formati

#### Ek 1: Performans Iyilestirme
**Sorumlu:** Mert
- [x] Hangi iyilestirme yontemleri denendi
- [x] Her yontemin ne kadar iyilestirme sagladi
- [x] Tablo ile karsilastirma

**Mert notu:** Performans iyilestirme tablosu ve GA entegrasyon ozeti `docs/mert_rapor_notlari.md` dosyasina eklendi.

#### Ek 2: Literatur Katkisi
**Sorumlu:** Mert Kerem
- [ ] Dunya'da ve Turkiye'de yapilan benzer calismalar
- [ ] Bu projenin farki ve ozgunlugu:
  - Coklu indikatör agirliklama yaklasimi
  - GA ile hem indikatör periyotlari hem trading parametreleri optimizasyonu
  - Kapsamli parametre duyarlilik analizi

#### Ek 3: Kullanilan Kaynaklar
**Sorumlu:** Yigit
- [ ] Faydalanilan calismalar, hazir kodlar, linkler
- [ ] Bu kaynaklara kiyasla farkliliklar

#### Ek 4: Is Paketleri ve Is Bolumu
**Sorumlu:** Gorkem Ege
- [ ] Is paketleri tablosu:

| Is Paketi | Gorev | Sorumlu | Durum |
|-----------|-------|---------|-------|
| IP-1 | Literatur Taramasi | Gorkem Ege & Mert | Tamamlandi |
| IP-2 | Veri Toplama & On Isleme | Yigit | Tamamlandi |
| IP-3 | GA Tasarimi & Kodlama | Mert | Tamamlandi |
| IP-4 | Trading Stratejisi & Backtesting | Mert Kerem | Tamamlandi |
| IP-5 | Deneysel Calismalar | Gorkem Ege | Tamamlandi |
| IP-6 | Gorsellesitirme | Gorkem Ege & Yigit | Tamamlandi |
| IP-7 | Rapor Yazimi | Tum Takim | Tamamlandi |
| IP-8 | Demo Video | Tum Takim | Tamamlandi |

#### Arastirma Sorusu (Uretici Yapay Zeka)
**Sorumlu:** Tum Takim (bolunmus)

**12.a) Uretici YZ'nin Faydalari/Avantajlari (10 madde)**
**Sorumlu:** Gorkem Ege
- [ ] 10 avantaj maddesi (arastirmaya dayali)

**12.b) Uretici YZ Kullanim Deneyimi**
**Sorumlu:** Yigit
- [ ] Projede kullanilan Gen AI araclari
- [ ] Nasil kullanildigi ve saglanan faydalar (kalite, hiz, inovasyon)

**12.c) Uretici YZ'nin Riskleri/Tehlikeleri**
**Sorumlu:** Mert & Mert Kerem
- [ ] Riskler ve tehlikeler
- [ ] Kisisel gozlemler

#### Oz Degerlendirme Tablosu
**Sorumlu:** Gorkem Ege
- [ ] Her bolum icin puan tahmini
- [ ] Aciklama bolumune eksiklik/basari bilgisi

### 6.2 Rapor Formatlama
**Sorumlu:** Gorkem Ege
**Sure:** 2 saat (14:00-16:00)

- [ ] Minimum 5, maksimum 13 sayfa (ekler dahil)
- [ ] Alinti yapilan yerlere kaynak gosterilmesi
- [ ] Tablo ve sekil numaralandirmasi
- [ ] Sayfa numaralari
- [ ] Tutarli yazi tipi ve boyutu
- [ ] Icindekiler tablosu

### 6.3 Demo Video Hazirlanmasi
**Sorumlu:** Tum Takim
**Sure:** 2 saat (16:00-18:00)

- [ ] Video suresi: 5-10 dakika
- [ ] Video icerigi:

| Bolum | Sure | Sunan | Icerik |
|-------|------|-------|--------|
| Giris | 1 dk | Gorkem Ege | Proje tanitimi, problem tanimi |
| Veri & On Isleme | 1.5 dk | Yigit | Veri toplama, indikatörler |
| GA Yapisi | 2 dk | Mert | Kromozom, fitness, operatörler, kod aciklamasi |
| Trading Stratejisi | 1.5 dk | Mert Kerem | Sinyal uretimi, backtesting, kod aciklamasi |
| Sonuclar | 2 dk | Gorkem Ege | Deneysel sonuclar, grafikler |
| Kapanis | 1 dk | Gorkem Ege | Sonuc, is bolumu ozeti |

- [ ] Ekran kaydi ile kodun calismasi gosterilecek
- [ ] Grafiklerin ve tablolarin gosterimi
- [ ] Is bolumunun videonun sonunda aciklanmasi

### 6.4 Kod Temizligi ve Dokumantasyon
**Sorumlu:** Yigit & Mert
**Sure:** 1 saat (09:00-10:00, Sabah ilk is)

- [ ] Kodlarin temizlenmesi ve duzenli hale getirilmesi
- [ ] Gereksiz yorum ve debug kodlarinin kaldirilmasi
- [x] `README.md` guncellenmesi:
  - Proje aciklamasi
  - Kurulum talimatlari
  - Kullanim kilavuzu
  - Proje yapisi
- [ ] `requirements.txt` guncellenmesi (final versiyonlari)

### 6.5 Final Teslimati
**Sorumlu:** Gorkem Ege (koordinasyon)
**Sure:** 1 saat (18:00-19:00, Gunun sonu)

- [ ] Teslim kontrol listesi:
  - [ ] Proje raporu (Word/PDF) - sisteme yuklenmis
  - [ ] Kaynak kodlar - sisteme yuklenmis
  - [ ] Demo videosu - sisteme yuklenmis
  - [ ] GitHub reposu guncel
  - [ ] Tum dosyalar tam ve eksiksiz

---

## Rapor Sayfa Dagilimi (Tavsiye)

| Bolum | Sayfa |
|-------|-------|
| Kapak | 0.5 |
| Problem Tanimi | 0.5 |
| Arastirma | 1 |
| Araclar ve Yontemler | 0.5 |
| Onerilen Yontem | 1.5 |
| Deneysel Calismalar | 2.5 |
| Sonuc | 0.5 |
| Referanslar | 0.5 |
| Ek 1-4 | 2 |
| Arastirma Sorusu | 1.5 |
| Oz Degerlendirme | 0.5 |
| **Toplam** | **~11** |

---

## Ciktilar

| Cikti | Format |
|-------|--------|
| Proje Raporu | .docx / .pdf |
| Demo Videosu | .mp4 (5-10 dk) |
| Temiz Kod | GitHub reposu |
| README | Markdown |

---

## Basari Kriterleri

- [ ] Rapor 5-13 sayfa araliginda
- [ ] Tum rapor bolumleri eksiksiz doldurulmus
- [ ] En az 10 referans eklenmis
- [ ] Tablo ve sekiller numaralandirilmis
- [ ] Demo video 5-10 dakika arasinda
- [ ] Videoda tum takim uyeleri konusmus
- [ ] Is bolumu videoda aciklanmis
- [ ] Kod temiz ve calisir durumda
- [ ] README ile kurulum ve calistirma talimatları mevcut
- [ ] Tum teslimatlar sisteme yuklenmis

---

## Notlar

- Rapor kalitesi notun onemli bir kismini olusturuyor - ozen gosterilmeli
- "Sadece hazir yontem uygulayip sonuc gostermek yeterli degil" - ekstra caba gostermek puan getirir
- Oz degerlendirme tablosu gercekci olmali, abartili puan yazmayin
- Arastirma sorusu (Uretici YZ) ayri bir bolum, unutulmasin
- Video'da kodun canli calismasi (demo) gosterilmeli
- Raporda alinti yapilan her sey icin referans verilmeli
