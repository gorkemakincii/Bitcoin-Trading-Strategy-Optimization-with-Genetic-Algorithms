### 3) Tools, Methods, and Libraries Used (Kullanılan Araçlar, Yöntemler ve Kütüphaneler)
Projede strateji testi (backtesting) aşamasında hazır bir kütüphane (örneğin backtrader) kullanmak yerine, **pandas** ve **numpy** tabanlı özel (*custom*) vektörel bir backtesting altyapısı geliştirilmiştir.

Genetik Algoritma (GA) sürecinde fitness (uyumluluk) değerlendirmesinin her nesilde popülasyon boyutu kadar tekrarlanması gerektiğinden, sistemin çalışma hızı kritik bir öneme sahiptir. backtrader gibi olay-yönelimli (*event-driven*) kütüphaneler komisyon hesabı gibi araçları otomatik sağlasa da, on binlerce simülasyon gerektiren GA'nın ihtiyaç duyduğu işlem hızını karşılayamamaktadır.

Bu nedenle;
* **Yüksek Hız:** Hızlı matris işlemleri sunan numpy ve pandas mimarisi tercih edilmiştir.
* **Esneklik:** Projeye özel geliştirilen Genetik Algoritma motoru ile doğrudan uyum ve esneklik avantajı sağlanmıştır.

---

### 4) Proposed (Developed/Used) Method (Önerilen / Geliştirilen Metot)

##### Backtesting Mekanizması
Geliştirilen özel backtesting motoru, üretilen ticaret stratejilerinin geçmiş veriler üzerindeki performansını gerçekçi piyasa koşullarında simüle etmektedir. Simülasyonun gerçekçiliğini ve güvenilirliğini en üst düzeye çıkarmak amacıyla sisteme şu kısıtlar ve kurallar entegre edilmiştir:

* **Gerçekçi Piyasa Koşulları:** Binance spot piyasası referans alınarak her bir işleme **%0.1 komisyon oranı** ve olası fiyat kayıplarını yansıtmak için **%0.05 kayma (slippage)** maliyeti eklenmiştir.
* **Veri Sızıntısı Önleme (Look-Ahead Bias):** Modelin gelecekteki verileri kullanarak yanıltıcı başarı elde etmesini engellemek amacıyla, vektörel kaydırma (*shift*) işlemleri kullanılarak katı önlemler alınmıştır.
* **Pozisyon ve Sermaye Yönetimi:** Sistem, sadece alım yönlü (*long-only*) pozisyonlar açacak şekilde tasarlanmış olup, risk yönetimi gereği **sermayenin %95'i ile tek pozisyon** kuralı uygulanmaktadır.
* **Performans Değerlendirme (Fitness Function):** Mekanizma, strateji başarısını Toplam Getiri (Total Return), Sharpe Oranı (Sharpe Ratio), Maksimum Düşüş (Maximum Drawdown) ve Kazanma Oranı (Win Rate) metrikleri üzerinden hesaplayarak Genetik Algoritma'nın uyumluluk (*fitness*) fonksiyonuna geri bildirim sağlamaktadır.

##### Modüller Arası Arayüz (Interface) Tasarımı
Genetik Algoritma (GA) motoru ile Backtesting motoru arasındaki parametre aktarımı; tip güvenliği (type safety), okunabilirlik ve geliştirme ortamlarında otomatik tamamlama desteği sağlaması amacıyla **Python DataClass** yapısı kullanılarak tasarlanmıştır. Bu mimari tercih sayesinde 16 farklı gen parametresinin modüller arası aktarımı sırasındaki yanlış indeksleme ve isim hataları riski sıfıra indirilmiştir.
