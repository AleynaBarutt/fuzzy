# Kalp Hastalığı Risk Tahmin Sistemi - Detaylı Teknik Dokümantasyon

Bu doküman, Mamdani Bulanık Çıkarım Sistemi'nin tüm bileşenlerini, çalışma prensiplerini ve kod implementasyonunu detaylı olarak açıklamaktadır.

---

## İçindekiler

1. [Bulanık Mantık Nedir?](#1-bulanık-mantık-nedir)
2. [Mamdani FIS Yapısı](#2-mamdani-fis-yapısı)
3. [Adım 1: Bulanıklaştırma](#3-adım-1-bulanıklaştırma-fuzzification)
4. [Adım 2: Kural Değerlendirme](#4-adım-2-kural-değerlendirme)
5. [Adım 3: Çıktı Birleştirme](#5-adım-3-çıktı-birleştirme-aggregation)
6. [Adım 4: Durulaştırma](#6-adım-4-durulaştırma-defuzzification)
7. [Hibrit Durulaştırma](#7-hibrit-durulaştırma)
8. [Fallback Mekanizması](#8-fallback-mekanizması)
9. [Kod Yapısı](#9-kod-yapısı)
10. [Örnek Senaryo](#10-örnek-senaryo-tam-çıkarım-süreci)

---

## 1. Bulanık Mantık Nedir?

### Klasik Mantık vs Bulanık Mantık

**Klasik (Boolean) Mantık:**
- Bir değer ya 0 ya da 1'dir
- "Hasta ya sağlıklı ya da hasta"
- Keskin sınırlar

**Bulanık (Fuzzy) Mantık:**
- Bir değer 0 ile 1 arasında herhangi bir değer alabilir
- "Hasta %60 sağlıklı, %40 riskli olabilir"
- Yumuşak, gerçek hayata uygun geçişler

### Neden Bulanık Mantık?

Tıbbi verilerde keskin sınırlar yoktur:
- Tansiyon 139 mmHg → Normal mi?
- Tansiyon 140 mmHg → Yüksek mi?

Bulanık mantık bu belirsizliği modeller:
- Tansiyon 139 mmHg → Normal(%60), Yüksek(%40)
- Tansiyon 140 mmHg → Normal(%50), Yüksek(%50)

---

## 2. Mamdani FIS Yapısı

Mamdani Bulanık Çıkarım Sistemi 4 adımdan oluşur:

```
┌─────────────────┐
│  SAYISAL GİRDİ  │  Yaş=65, Tansiyon=145, HbA1c=8.5, ...
└────────┬────────┘
         ↓
┌─────────────────┐
│ 1. BULANIKLAŞ-  │  Age→Old(0.8), BP→High(0.7), HbA1c→High(0.6)
│    TIRMA        │  Sayısal değerler → Üyelik dereceleri
└────────┬────────┘
         ↓
┌─────────────────┐
│ 2. KURAL        │  4057 IF-THEN kuralı değerlendirilir
│    DEĞERLENDİRME│  Her kuralın aktivasyon derecesi hesaplanır
└────────┬────────┘
         ↓
┌─────────────────┐
│ 3. ÇIKTI        │  Aktive olan kuralların çıktıları
│    BİRLEŞTİRME  │  MAX operatörü ile birleştirilir
└────────┬────────┘
         ↓
┌─────────────────┐
│ 4. DURULASTIRMA │  Bulanık çıktı → Net sayısal değer
│    (Hibrit)     │  Centroid + Bisector + MOM ortalaması
└────────┬────────┘
         ↓
┌─────────────────┐
│     SONUÇ       │  Risk Skoru: 7.2 → Kategori: HighRisk
└─────────────────┘
```

---

## 3. Adım 1: Bulanıklaştırma (Fuzzification)

### Üçgensel Üyelik Fonksiyonu (Triangular MF)

Üç parametre ile tanımlanır: [a, b, c]
- a: Sol sınır (üyelik = 0)
- b: Tepe noktası (üyelik = 1)
- c: Sağ sınır (üyelik = 0)

```
Üyelik
Derecesi (μ)
    1.0          •
               / \
              /   \
             /     \
    0.0 ----+---+---+----→ Değer
            a   b   c
```

### Matematiksel Formül

```
         ⎧ 0,           x ≤ a
         ⎪ (x-a)/(b-a), a < x ≤ b
μ(x) =   ⎨ (c-x)/(c-b), b < x < c
         ⎪ 0,           x ≥ c
         ⎩
```

### Kod Implementasyonu

```python
class FuzzyVariable:
    def __init__(self, name, universe, mfs):
        self.name = name
        self.universe = universe  # Değer aralığı (örn: 20-100 yaş)
        # Her terim için üyelik fonksiyonu oluştur
        self.mfs = {term: fuzz.trimf(universe, params) 
                    for term, params in mfs.items()}
    
    def fuzzify(self, value):
        """Bir sayısal değerin tüm kümelerdeki üyelik derecelerini hesaplar"""
        return {term: float(fuzz.interp_membership(self.universe, mf, value)) 
                for term, mf in self.mfs.items()}
```

### Sistemdeki Değişken Tanımları

```python
# Yaş değişkeni: 20-100 yıl aralığında
self.variables['Age'] = FuzzyVariable('Age', np.arange(20, 101, 1), {
    'Young':   [20, 32, 45],   # Genç: 20-45 arası, tepe 32
    'Mid':     [40, 52, 65],   # Orta: 40-65 arası, tepe 52
    'Old':     [60, 72, 85],   # Yaşlı: 60-85 arası, tepe 72
    'VeryOld': [80, 95, 100]   # Çok yaşlı: 80-100 arası, tepe 95
})

# HbA1c (Şeker) değişkeni: 4-14% aralığında
self.variables['HbA1c'] = FuzzyVariable('HbA1c', np.arange(4, 14.1, 0.1), {
    'VeryHealthy': [4, 5, 6.5],    # Çok sağlıklı: <6.5%
    'Healthy':     [6, 7.5, 9],    # Sağlıklı: 6-9%
    'High':        [8, 11, 14]     # Yüksek: >8%
})
```

### Örnek: HbA1c = 7.5 için Bulanıklaştırma

```
HbA1c = 7.5

VeryHealthy [4, 5, 6.5]:
  7.5 > 6.5 → μ = 0.0

Healthy [6, 7.5, 9]:
  7.5 = b (tepe noktası) → μ = 1.0

High [8, 11, 14]:
  7.5 < 8 → μ = 0.0

Sonuç: {'VeryHealthy': 0.0, 'Healthy': 1.0, 'High': 0.0}
```

### Örtüşen Üyelik Fonksiyonları

```
Üyelik                Young        Mid         Old       VeryOld
    1.0                 •           •           •           •
                       / \         / \         / \         / \
                      /   \       /   \       /   \       /   \
    0.0 ─────────────+─────+─────+─────+─────+─────+─────+─────→ Yaş
                    20    45    40    65    60    85    80   100

Örnek: Yaş = 55
  - Young: 0.0
  - Mid: (65-55)/(65-52) = 0.77  ← En yüksek
  - Old: (55-60)/(72-60) = -0.42 → 0.0 (negatif olmaz)
  - VeryOld: 0.0
```

---

## 4. Adım 2: Kural Değerlendirme

### IF-THEN Kuralları

Sistemde 4,057 adet kural bulunur. Her kural şu formattadır:

```
IF ChestPain=Typical AND HbA1c=High AND LDL=VeryHigh 
   AND HDL=Low AND HeartRate=High AND BloodPressure=VeryHigh AND Age=Old 
THEN Risk=HighRisk
```

### Kural Dosyası Formatı (CSV)

```csv
"Antecedent (If Condition)","Consequent (Output)"
"ChestPain = NoPain AND HbA1c = VeryHealthy AND LDL = Healthy AND HDL = Healthy AND HeartRate = VeryHealthy AND BloodPressure = Medium AND Age = Young","Healthy"
"ChestPain = Typical AND HbA1c = High AND LDL = XHigh AND HDL = Low AND HeartRate = High AND BloodPressure = VeryHigh AND Age = VeryOld","HighRisk"
```

### Kural Aktivasyonu: MIN Operatörü

AND bağlacı için MIN (minimum) operatörü kullanılır:

```
Aktivasyon = MIN(μ_ChestPain, μ_HbA1c, μ_LDL, μ_HDL, μ_HeartRate, μ_BP, μ_Age)
```

### Kod Implementasyonu

```python
def evaluate_rules(self, fuzzified):
    """Her kuralın aktivasyon derecesini hesaplar"""
    activations = []
    
    for rule in self.rules:
        activation = 1.0  # Başlangıç değeri (tam aktif)
        
        for var, term in rule['antecedent'].items():
            if var in fuzzified and term in fuzzified[var]:
                # MIN operatörü: En düşük üyelik değerini al
                activation = min(activation, fuzzified[var][term])
            else:
                activation = 0  # Terim bulunamazsa kural devre dışı
                break
        
        if activation > 0:
            activations.append({
                'activation': activation,
                'consequent': rule['consequent']
            })
    
    return activations
```

### Örnek: Kural Aktivasyonu Hesaplama

```
Girdiler:
  Age=65 → Old(0.8), Mid(0.2)
  HbA1c=8.5 → High(0.5), Healthy(0.5)
  LDL=160 → VeryHigh(0.7), High(0.3)
  HDL=35 → Low(0.8), Healthy(0.2)
  HeartRate=90 → High(0.4), Healthy(0.6)
  BloodPressure=150 → VeryHigh(0.6), High(0.4)
  ChestPain=2 → Atypical(1.0)

Kural: IF Age=Old AND HbA1c=High AND LDL=VeryHigh AND HDL=Low 
          AND HeartRate=High AND BloodPressure=VeryHigh AND ChestPain=Atypical 
       THEN Risk=HighRisk

Aktivasyon = MIN(0.8, 0.5, 0.7, 0.8, 0.4, 0.6, 1.0) = 0.4
```

---

## 5. Adım 3: Çıktı Birleştirme (Aggregation)

### Implication (Kesme)

Her kuralın çıktı üyelik fonksiyonu, aktivasyon seviyesinde kesilir:

```
Orijinal MF:                    Kesilen MF (aktivasyon=0.6):
    1.0    /\                       0.6    ____
          /  \                            /    \
         /    \                          /      \
    0.0 +------+                    0.0 +--------+
```

### MAX ile Birleştirme

Tüm kesilen çıktılar MAX operatörü ile birleştirilir:

```python
def aggregate(self, activations):
    """Aktive olan kuralların çıktılarını birleştirir"""
    aggregated = np.zeros_like(self.risk_universe)  # Boş çıktı
    
    for rule in activations:
        # Kuralın çıktı MF'sini aktivasyon seviyesinde kes
        clipped = np.minimum(rule['activation'], 
                            self.risk_mfs[rule['consequent']])
        # MAX ile birleştir
        aggregated = np.maximum(aggregated, clipped)
    
    return aggregated
```

### Görsel Örnek

```
Kural 1: HighRisk, aktivasyon=0.6     Kural 2: MediumRisk, aktivasyon=0.3
        
    0.6  ____                              0.3  ____
        /    \                                 /    \
       /      \                               /      \
   0  +--------+----                      0  +--------+----
      6      8.5    10                       4   6    8

Birleştirilmiş Çıktı (MAX):

   0.6        ____
             /    \
   0.3  ____/      \
       /            \
   0  +----+----+----+----
      4    6   8.5   10
```

---

## 6. Adım 4: Durulaştırma (Defuzzification)

Bulanık çıktıyı net bir sayısal değere dönüştürür.

### Centroid (Ağırlık Merkezi) Yöntemi

En yaygın kullanılan yöntem. Alanın ağırlık merkezini bulur.

```
        ∫ x · μ(x) dx
COA = ─────────────────
          ∫ μ(x) dx
```

### Bisector (İkiye Bölen) Yöntemi

Alanı iki eşit parçaya bölen dikey çizginin x koordinatı.

### MOM (Mean of Maximum) Yöntemi

Maksimum üyelik değerlerine sahip noktaların ortalaması.

### Kod Implementasyonu

```python
def defuzzify_hybrid(self, aggregated):
    """Üç yöntemin ortalamasını alır"""
    if np.sum(aggregated) == 0:
        return 5.0  # Varsayılan değer
    
    results = []
    for method in ['centroid', 'bisector', 'mom']:
        try:
            r = fuzz.defuzz(self.risk_universe, aggregated, method)
            if not np.isnan(r):
                results.append(r)
        except:
            pass
    
    return np.mean(results) if results else 5.0
```

---

## 7. Hibrit Durulaştırma

### Neden Hibrit?

Her durulaştırma yöntemi farklı avantajlara sahiptir:

| Yöntem | Avantaj | Dezavantaj |
|--------|---------|------------|
| Centroid | Kararlı, sağlam | Uç değerlere duyarsız |
| Bisector | Asimetriyi yakalar | Hesaplama maliyeti |
| MOM | Hızlı, basit | Çoklu tepe noktalarında belirsiz |

### Hibrit Yaklaşım

Üç yöntemin ortalamasını alarak dezavantajları minimize ederiz:

```python
final_score = (centroid + bisector + mom) / 3
```

---

## 8. Fallback Mekanizması

### Problem

4,057 kural 7 değişken için tüm kombinasyonları kapsamaz:
- Teorik kombinasyon: 4 × 3 × 5 × 2 × 3 × 3 × 4 = 4,320
- Mevcut kural: 4,057
- Bazı girdiler için eşleşen kural bulunamayabilir

### Çözüm: Risk Faktörü Ağırlıklandırması

Kural bulunamadığında, her değişkenin bağımsız risk katkısı hesaplanır:

```python
def _calculate_risk_score(self, fuzzified):
    """Kural bulunamadığında risk faktörlerine göre skor hesapla"""
    
    risk_weights = {
        'Age': {'Young': 0, 'Mid': 1, 'Old': 2, 'VeryOld': 3},
        'BloodPressure': {'Medium': 0, 'High': 1.5, 'VeryHigh': 3},
        'HbA1c': {'VeryHealthy': 0, 'Healthy': 1, 'High': 2.5},
        'LDL': {'VeryHealthy': 0, 'Healthy': 0.5, 'High': 1.5, 
                'VeryHigh': 2, 'XHigh': 2.5},
        'HDL': {'Healthy': 0, 'Low': 1.5},
        'HeartRate': {'VeryHealthy': 0, 'Healthy': 0.5, 'High': 1.5},
        'ChestPain': {'NoPain': 0, 'NonAnginal': 1, 'Atypical': 2, 'Typical': 3}
    }
    
    total_score = 0
    for var, memberships in fuzzified.items():
        if memberships and var in risk_weights:
            dominant = max(memberships, key=memberships.get)
            if dominant in risk_weights[var]:
                total_score += risk_weights[var][dominant]
    
    # 0-10 aralığına normalize et
    max_possible = 17  # 3+3+2.5+2.5+1.5+1.5+3
    normalized = (total_score / max_possible) * 10
    return min(normalized, 10)
```

### Ağırlık Tablosu

| Faktör | Düşük Risk | Orta Risk | Yüksek Risk |
|--------|------------|-----------|-------------|
| Age | Young: 0 | Mid: 1, Old: 2 | VeryOld: 3 |
| BloodPressure | Medium: 0 | High: 1.5 | VeryHigh: 3 |
| HbA1c | VeryHealthy: 0 | Healthy: 1 | High: 2.5 |
| LDL | VeryHealthy: 0 | Healthy: 0.5, High: 1.5 | VeryHigh: 2, XHigh: 2.5 |
| HDL | Healthy: 0 | - | Low: 1.5 |
| HeartRate | VeryHealthy: 0 | Healthy: 0.5 | High: 1.5 |
| ChestPain | NoPain: 0 | NonAnginal: 1, Atypical: 2 | Typical: 3 |

---

## 9. Kod Yapısı

### Sınıf Diyagramı

```
┌─────────────────────────────────────────────────────────────┐
│                      MamdaniFIS                             │
├─────────────────────────────────────────────────────────────┤
│ - variables: dict[str, FuzzyVariable]                       │
│ - rules: list[dict]                                         │
│ - risk_universe: np.array                                   │
│ - risk_mfs: dict[str, np.array]                             │
├─────────────────────────────────────────────────────────────┤
│ + load_rules(filepath) → int                                │
│ + fuzzify_inputs(numeric_inputs) → dict                     │
│ + evaluate_rules(fuzzified) → list                          │
│ + aggregate(activations) → np.array                         │
│ + defuzzify_hybrid(aggregated) → float                      │
│ + infer(numeric_inputs) → tuple                             │
│ + infer_categorical(categorical_inputs) → tuple             │
│ - _calculate_risk_score(fuzzified) → float                  │
└─────────────────────────────────────────────────────────────┘
                            │
                            │ uses
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                     FuzzyVariable                           │
├─────────────────────────────────────────────────────────────┤
│ - name: str                                                 │
│ - universe: np.array                                        │
│ - mfs: dict[str, np.array]                                  │
├─────────────────────────────────────────────────────────────┤
│ + fuzzify(value) → dict[str, float]                         │
└─────────────────────────────────────────────────────────────┘
```

### Ana Çıkarım Fonksiyonu

```python
def infer(self, numeric_inputs):
    """
    Tam Mamdani çıkarım süreci
    
    Args:
        numeric_inputs: {'Age': 65, 'HbA1c': 8.5, ...}
    
    Returns:
        score: float (0-10 arası risk skoru)
        category: str ('Healthy', 'LowRisk', 'MediumRisk', 'HighRisk')
        fuzzified: dict (bulanıklaştırma sonuçları)
        activations: list (aktive olan kurallar)
    """
    # 1. Bulanıklaştırma
    fuzzified = self.fuzzify_inputs(numeric_inputs)
    
    # 2. Kural değerlendirme
    activations = self.evaluate_rules(fuzzified)
    
    # 3. Fallback kontrolü
    if not activations:
        score = self._calculate_risk_score(fuzzified)
    else:
        # 4. Çıktı birleştirme
        aggregated = self.aggregate(activations)
        # 5. Hibrit durulaştırma
        score = self.defuzzify_hybrid(aggregated)
    
    # 6. Sınıflandırma
    if score < 3: category = 'Healthy'
    elif score < 5: category = 'LowRisk'
    elif score < 7: category = 'MediumRisk'
    else: category = 'HighRisk'
    
    return score, category, fuzzified, activations
```

---

## 10. Örnek Senaryo: Tam Çıkarım Süreci

### Hasta Verileri

```
Yaş: 70 yıl
Tansiyon: 155 mmHg
HbA1c: 9.0%
LDL: 170 mg/dL
HDL: 35 mg/dL
Nabız: 95 bpm
Göğüs Ağrısı: Atypical (2)
```

### Adım 1: Bulanıklaştırma

```
Age = 70:
  - Young [20,32,45]: 0.0
  - Mid [40,52,65]: 0.0
  - Old [60,72,85]: (85-70)/(85-72) = 1.15 → 1.0 ✓
  - VeryOld [80,95,100]: 0.0
  → Dominant: Old (1.0)

BloodPressure = 155:
  - Medium [80,100,120]: 0.0
  - High [110,135,160]: (160-155)/(160-135) = 0.2
  - VeryHigh [150,175,200]: (155-150)/(175-150) = 0.2
  → Dominant: High (0.2) veya VeryHigh (0.2)

HbA1c = 9.0:
  - VeryHealthy [4,5,6.5]: 0.0
  - Healthy [6,7.5,9]: (9-9)/(9-7.5) = 0.0
  - High [8,11,14]: (9-8)/(11-8) = 0.33
  → Dominant: High (0.33)

LDL = 170:
  - VeryHigh [140,165,190]: (190-170)/(190-165) = 0.8
  - XHigh [180,215,250]: 0.0
  → Dominant: VeryHigh (0.8)

HDL = 35:
  - Low [20,30,45]: (45-35)/(45-30) = 0.67
  - Healthy [40,65,100]: 0.0
  → Dominant: Low (0.67)

HeartRate = 95:
  - VeryHealthy [50,60,75]: 0.0
  - Healthy [65,82,100]: (100-95)/(100-82) = 0.28
  - High [90,135,180]: (95-90)/(135-90) = 0.11
  → Dominant: Healthy (0.28)

ChestPain = 2:
  - Atypical [1.5,2,2.5]: 1.0 (tam tepe)
  → Dominant: Atypical (1.0)
```

### Adım 2: Kural Eşleştirme

```
Örnek Kural: 
IF Age=Old AND HbA1c=High AND LDL=VeryHigh AND HDL=Low 
   AND HeartRate=Healthy AND BloodPressure=High AND ChestPain=Atypical
THEN Risk=HighRisk

Aktivasyon = MIN(1.0, 0.33, 0.8, 0.67, 0.28, 0.2, 1.0)
           = 0.2
```

### Adım 3-4: Birleştirme ve Durulaştırma

```
Aktive kuralların HighRisk çıktısı 0.2 seviyesinde kesilir.

Hibrit Durulaştırma:
- Centroid: 7.8
- Bisector: 7.5
- MOM: 8.5

Final Skor = (7.8 + 7.5 + 8.5) / 3 = 7.93
```

### Sonuç

```
Risk Skoru: 7.93
Kategori: HighRisk
Öneri: Acil tıbbi değerlendirme gerekli
```

---

## Özet: Sunum İçin Anahtar Noktalar

1. **Bulanık Mantık** belirsizliği modeller, keskin sınırlar yerine yumuşak geçişler sağlar

2. **Mamdani FIS** 4 adımdan oluşur: Bulanıklaştırma → Kural Değerlendirme → Birleştirme → Durulaştırma

3. **Üçgensel MF** ile her değer 0-1 arası üyelik derecesi alır

4. **MIN operatörü** AND için, **MAX operatörü** birleştirme için kullanılır

5. **Hibrit Durulaştırma** üç yöntemin ortalamasıdır (Centroid + Bisector + MOM)

6. **Fallback mekanizması** kural bulunamadığında risk faktörü ağırlıklandırması yapar

7. **%97.31 doğruluk** oranı elde edilmiştir
