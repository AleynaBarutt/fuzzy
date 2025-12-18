# Kalp Hastalığı Risk Tahmin Sistemi - Kod Mantığı Açıklaması

Bu doküman, `heart_disease_fuzzy_system.py` dosyasındaki kodların ne işe yaradığını, satır satır ve blok blok, yazılımı veya bulanık mantığı hiç bilmeyen birine anlatır gibi açıklamaktadır.

---

## 1. Kütüphaneler ve Hazırlık

Kodun en başında, ihtiyacımız olan araçları yüklüyoruz:

- **numpy**: Sayısal hesaplamalar için (diziler, matematiksel işlemler).
- **pandas**: Veri okumak için (Excel/CSV dosyalarını tablo gibi açar).
- **skfuzzy**: Bulanık mantık (Fuzzy Logic) işlemleri için özel bir kütüphane. Üyelik fonksiyonlarını ve durulaştırmayı bu yapar.
- **sklearn.metrics**: Sistemin ne kadar doğru çalıştığını ölçmek için (Doğruluk oranı, karmaşıklık matrisi).

Ayrıca `RULES_FILE` ve `TEST_FILE` gibi sabitlerle, kuralların ve test verilerinin hangi dosyalarda olduğunu belirtiyoruz. `TYPO_CORRECTIONS` sözlüğü ise veri setindeki yazım hatalarını düzeltmek için (örneğin "HIgh" yazılmışsa, sistemin bunu "High" olarak anlaması için).

---

## 2. FuzzyVariable Sınıfı (Bulanık Değişken)

Bu sınıf, "Yaş", "Tansiyon" gibi her bir tıbbi değişkeni temsil eder.

### Ne Yapar?
Normal (keskin) bir sayıyı, bulanık kavramlara dönüştürür.
Örneğin: Yaş değişkeni için 65 sayısını alır, bunun ne kadar "Genç", ne kadar "Orta", ne kadar "Yaşlı" olduğunu hesaplar.

- **`__init__`**: Değişkenin adını, alabileceği değer aralığını (universe) ve bulanık kümelerini (mfs) tanımlar.
- **`fuzzify`**: Asıl işi yapan kısımdır. Gelen sayıyı alır, tanımlı her bir küme (örneğin Genç, Yaşlı) için "üyelik derecesi" hesaplar. Sonuç 0 ile 1 arasındadır.

---

## 3. MamdaniFIS Sınıfı (Ana Beyin)

Sistemin bütün mantığı bu sınıfın içindedir. "Mamdani", kullandığımız bulanık çıkarım yönteminin adıdır.

### `__init__` ve `_define_variables`
Sistem başlatıldığında çalışan kısımdır. Burada tıbbi bilgiler kodlanır:
- **Risk Çıkışı**: 0'dan 10'a kadar bir risk puanı tanımlanır. (0-3: Sağlıklı, 7-10: Yüksek Risk gibi).
- **Değişkenler**: Yaş, HbA1c, LDL gibi değişkenlerin sınırları belirlenir.
  - *Örnek*: "HbA1c" (Şeker) için 4 ile 14 arası bir aralık belirlenir. 6.5 altı "Çok Sağlıklı", 8 üstü "Yüksek" olarak tanımlanır. Bu sınırlar `fuzz.trimf` (üçgen fonksiyon) ile çizilir.

### `load_rules` (Kuralları Yükle)
`inference_rules_corrected.csv` dosyasını okur.
Bu dosyadaki 4000+ satırı tek tek okur ve bilgisayarın anlayacağı bir formata çevirir.
- *Örnek Kural*: "Eğer Göğüs Ağrısı Tipik VE Yaş Çok Yaşlı İSE -> Yüksek Risk".
Bu fonksiyon metinleri okur, "VE" (AND) ile ayrılmış şartları bulur ve bunları listeye ekler.

### `fuzzify_inputs` (Girdileri Bulanıklaştır)
Hastadan gelen sayısal verileri (Yaş: 55, Tansiyon: 140 vb.) alır ve `FuzzyVariable` sınıfını kullanarak bulanıklaştırır.
- Girdi: `{'Age': 55}`
- Çıktı: `{'Age': {'Young': 0.2, 'Mid': 0.8, 'Old': 0.0 ...}}` (Yani 55 yaş, %80 oranında "Orta", %20 oranında "Genç" grubuna giriyor).

### `evaluate_rules` (Kuralları Değerlendir) **[KRİTİK BÖLÜM]**
Burası karar mekanizmasıdır.
1. Hafızadaki tüm kuralları tek tek gezer.
2. Her kural için, hastanın durumu o kurala ne kadar uyuyor ona bakar.
3. Kuralda "VE" (AND) kullanıldığı için, tüm şartlar arasındaki **en küçük** uyum değerini (MIN) alır.
   - *Örnek*: Kural "Yaşlı VE Yüksek Tansiyon" diyor. Hasta %80 Yaşlı, %40 Yüksek Tansiyonlu ise, bu kuralın gücü %40'tır (0.4).

### `aggregate` (Sonuçları Birleştir)
Aktif olan tüm kuralların çıktılarını birleştirir.
- Bir kural "Yüksek Risk" diyor (%40 güçle), başka bir kural "Orta Risk" diyor (%20 güçle).
- Bu fonksiyon tüm bu önerileri üst üste koyar (MAX operatörü). Sonuçta elimizde yamuk yumuk bir şekil (bulanık alan) oluşur.

### `defuzzify_hybrid` (Durulaştırma - Sonuca Varma)
Oluşan bulanık şekli tek bir sayıya (Risk Puanı) çevirir. Tek bir yöntem yerine 3 yöntemi birleştirip (Hibrit) daha sağlam sonuç üretir:
1. **Centroid**: Şeklin ağırlık merkezini bulur.
2. **Bisector**: Alanı ikiye bölen çizgiyi bulur.
3. **MOM (Mean of Maximum)**: En yüksek noktaların ortalamasını alır.
Sonuçta bu üçünün ortalaması alınarak nihai risk puanı (örneğin 7.4) bulunur.

### `infer` (Çıkarım Yap)
Tüm süreci yöneten ana fonksiyondur:
1. `fuzzify_inputs`: Sayısal veriyi bulanığa çevir.
2. `evaluate_rules`: Kuralları çalıştır.
3. Eğer hiç kural çalışmazsa -> `_calculate_risk_score` ile Fallback (Düşüş) mekanizmasını çalıştır (aşağıda).
4. Kural çalışırsa -> Birleştir ve Durulaştır.
5. Son puanı kategoriye ayır (0-3 Sağlıklı, 7+ Yüksek Risk vb.).

### `_calculate_risk_score` (Fallback / B Planı)
Eğer hastanın verileri o kadar nadir bir kombinasyon ki, 4000 kuraldan hiçbiri uymuyorsa bu devreye girer.
- Burada manuel puanlama vardır.
- Yaşlıysa +2 puan, Sigara varsa +1 puan gibi basit toplama mantığıyla yaklaşık bir risk hesaplar. Bu sayede sistem asla "Cevap veremiyorum" demez.

---

## 4. Evaluate ve Main (Test Kısmı)

Dosyanın en altındaki `evaluate` fonksiyonu sistemi test etmek içindir.
- `testing_data_set.csv` dosyasındaki 260 hastayı tek tek sisteme sokar.
- Sistemin tahmini ile doktorun gerçek kararını karşılaştırır.
- Sonunda "Doğruluk: %97.31" gibi bir rapor basar.

## Özet: Veri Nasıl Akar?

1. **Hasta Verisi** (Sayılar)
   ⬇️
2. **Bulanıklaştırma** (Sayılar -> Dereceler: %80 Yaşlı, %20 Genç)
   ⬇️
3. **Kural Motoru** (Kuralları Tara, Uygun Olanları Seç)
   ⬇️
4. **Birleştirme** (Farklı kuralların çıktılarını üst üste koy)
   ⬇️
5. **Durulaştırma** (Ortaya çıkan şekilden tek bir Risk Puanı üret)
   ⬇️
6. **Sonuç** (Yüksek Risk)
