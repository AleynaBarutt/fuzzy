# Kalp Hastalığı Risk Tahmin Sistemi

Mamdani Bulanık Çıkarım Sistemi ve Hibrit Durulaştırma kullanılarak geliştirilen kalp hastalığı risk tahmini uzman sistemi.

## Sistem Performansı

| Metrik | Değer |
|--------|-------|
| Doğruluk | %97.31 |
| Test Sayısı | 260 |
| Kural Sayısı | 4,057 |

## Giriş Değişkenleri

| Değişken | Birim | Aralık | Bulanık Kümeler |
|----------|-------|--------|-----------------|
| Yaş | yıl | 20-100 | Young, Mid, Old, VeryOld |
| Tansiyon | mmHg | 80-200 | Medium, High, VeryHigh |
| HbA1c | % | 4-14 | VeryHealthy, Healthy, High |
| LDL | mg/dL | 50-250 | VeryHealthy, Healthy, High, VeryHigh, XHigh |
| HDL | mg/dL | 20-100 | Low, Healthy |
| Nabız | bpm | 50-180 | VeryHealthy, Healthy, High |
| Göğüs Ağrısı | - | 0-3 | NoPain, NonAnginal, Atypical, Typical |

## Çıkış Değişkeni

| Kategori | Skor | Açıklama |
|----------|------|----------|
| Healthy | 0-3 | Sağlıklı |
| LowRisk | 3-5 | Düşük Risk |
| MediumRisk | 5-7 | Orta Risk |
| HighRisk | 7-10 | Yüksek Risk |

## Kullanım

### Kurulum
```bash
pip install numpy pandas scikit-fuzzy scikit-learn
```

### Konsol Testi
```bash
python heart_disease_fuzzy_system.py
```

### GUI Arayüzü
```bash
python gui.py
```

## Dosyalar

| Dosya | Açıklama |
|-------|----------|
| heart_disease_fuzzy_system.py | Mamdani FIS ana modülü |
| gui.py | Tkinter tabanlı kullanıcı arayüzü |
| inference_rules_corrected.csv | 4,057 IF-THEN kuralı |
| testing_data_set.csv | 260 test vakası |
| SISTEM_ACIKLAMASI.md | Teknik dokümantasyon |

## Metodoloji

1. **Bulanıklaştırma**: Üçgensel üyelik fonksiyonları (trimf)
2. **Kural Değerlendirme**: MIN operatörü (AND)
3. **Çıktı Birleştirme**: MAX operatörü
4. **Hibrit Durulaştırma**: Centroid + Bisector + MOM ortalaması
5. **Fallback**: Kural bulunamazsa risk faktörü ağırlıklandırması
