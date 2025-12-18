# === KÜTÜPHANELER VE TEMEL AYARLAR ===

# Gerekli kütüphaneler içe aktarılıyor.
import numpy as np  # Sayısal hesaplamalar ve dizi işlemleri için (örn: np.arange, np.minimum).
import pandas as pd  # Veri işleme ve CSV dosyalarını okumak için (örn: pd.read_csv).
import skfuzzy as fuzz  # Bulanık mantık fonksiyonları için (örn: fuzz.trimf, fuzz.defuzz).
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score  # Model performansını ölçmek için.
import warnings  # Kod çalışırken çıkan önemsiz uyarıları bastırmak için.

# Çalışma zamanı uyarılarını (runtime warnings) görmezden gel. Bu, çıktıyı daha temiz tutar.
warnings.filterwarnings('ignore')

# === SABİTLER VE VERİ TEMİZLEME ===

# Bulanık çıkarım kurallarının bulunduğu dosyanın adı.
RULES_FILE = "inference_rules_corrected.csv"
# Test verilerinin bulunduğu dosyanın adı.
TEST_FILE = "testing_data_set.csv"

# Veri setindeki olası yazım hatalarını veya tutarsızlıkları düzeltmek için bir sözlük.
# Bu, verinin standartlaşmasını sağlar (örn: "HIgh" -> "High").
TYPO_CORRECTIONS = {
    "HIgh": "High", "VeryHIgh": "VeryHigh", "XHIgh": "XHigh",
    "Normal": "Medium", "A typical": "Atypical", 
    "Very Healthy": "VeryHealthy", "Very Old": "VeryOld", "ExtraHigh": "XHigh"
}


# === FUZZYVARIABLE SINIFI: BİR BULANIK DEĞİŞKENİ TEMSİL EDER ===

class FuzzyVariable:
    """Bir bulanık değişkeni (örn: Yaş) ve üyelik fonksiyonlarını ('Genç', 'Orta', 'Yaşlı') yönetir."""
    
    # Sınıfın yapıcı metodu: Değişken oluşturulurken çağrılır.
    def __init__(self, name, universe, mfs):
        self.name = name  # Değişkenin adı (örn: 'Age').
        self.universe = universe  # Değişkenin alabileceği sayısal değer aralığı (evren kümesi).
        # Üyelik fonksiyonlarını (mfs) bir sözlük olarak oluşturur.
        # Her bir sözel terim ('Young') için bir üçgen üyelik fonksiyonu (trimf) yaratır.
        self.mfs = {term: fuzz.trimf(universe, params) for term, params in mfs.items()}
    
    # Bulanıklaştırma metodu: Net bir sayısal değeri bulanık bir değere dönüştürür.
    def fuzzify(self, value):
        """Verilen net bir değerin her bir bulanık kümeye olan üyelik derecesini hesaplar."""
        # Gelen değerin (örn: yaş=48) her bir üyelik fonksiyonuna göre üyelik derecesini (0-1 arası) hesaplar.
        return {term: float(fuzz.interp_membership(self.universe, mf, value)) 
                for term, mf in self.mfs.items()}


# === MAMDANIFIS SINIFI: ANA BULANIK ÇIKARIM SİSTEMİ ===

class MamdaniFIS:
    """Mamdani tipi bulanık çıkarım sisteminin tüm adımlarını yönetir."""
    
    # Sistemin yapıcı metodu: Başlangıç ayarlarını yapar.
    def __init__(self):
        self.variables = {}  # Girdi bulanık değişkenlerini saklamak için sözlük.
        self.rules = []  # Çıkarım kurallarını saklamak için liste.
        self.risk_universe = np.arange(0, 10.01, 0.01)  # Çıktı değişkeni 'Risk' için evren kümesi (0-10 arası).
        # Çıktı değişkeninin sözel terimleri ('Healthy', 'LowRisk' vb.) ve üyelik fonksiyonları.
        self.risk_mfs = {
            'Healthy': fuzz.trimf(self.risk_universe, [0, 1.5, 3]),
            'LowRisk': fuzz.trimf(self.risk_universe, [2, 4, 6]),
            'MediumRisk': fuzz.trimf(self.risk_universe, [4, 6, 8]),
            'HighRisk': fuzz.trimf(self.risk_universe, [6, 8.5, 10])
        }
        self._define_variables()  # Girdi değişkenlerini tanımlamak için yardımcı metodu çağırır.
    
    # Girdi değişkenlerini ve üyelik fonksiyonlarını tanımlar.
    def _define_variables(self):
        # 'Age' değişkeni ve sözel terimleri ('Young', 'Mid', 'Old', 'VeryOld').
        self.variables['Age'] = FuzzyVariable('Age', np.arange(20, 101, 1),
            {'Young': [20, 32, 45], 'Mid': [40, 52, 65], 'Old': [60, 72, 85], 'VeryOld': [80, 95, 100]})
        
        # 'HbA1c' (kan şekeri) değişkeni ve sözel terimleri.
        self.variables['HbA1c'] = FuzzyVariable('HbA1c', np.arange(4, 14.1, 0.1),
            {'VeryHealthy': [4, 5, 6.5], 'Healthy': [6, 7.5, 9], 'High': [8, 11, 14]})
        
        # 'LDL' (kötü kolesterol) değişkeni ve sözel terimleri.
        self.variables['LDL'] = FuzzyVariable('LDL', np.arange(50, 251, 1),
            {'VeryHealthy': [50, 65, 80], 'Healthy': [70, 90, 110], 'High': [100, 125, 150],
             'VeryHigh': [140, 165, 190], 'XHigh': [180, 215, 250]})
        
        # 'HDL' (iyi kolesterol) değişkeni ve sözel terimleri.
        self.variables['HDL'] = FuzzyVariable('HDL', np.arange(20, 101, 1),
            {'Low': [20, 30, 45], 'Healthy': [40, 65, 100]})
        
        # 'HeartRate' (kalp atış hızı) değişkeni ve sözel terimleri.
        self.variables['HeartRate'] = FuzzyVariable('HeartRate', np.arange(50, 181, 1),
            {'VeryHealthy': [50, 60, 75], 'Healthy': [65, 82, 100], 'High': [90, 135, 180]})
        
        # 'BloodPressure' (kan basıncı) değişkeni ve sözel terimleri.
        self.variables['BloodPressure'] = FuzzyVariable('BloodPressure', np.arange(80, 201, 1),
            {'Medium': [80, 100, 120], 'High': [110, 135, 160], 'VeryHigh': [150, 175, 200]})
        
        # 'ChestPain' (göğüs ağrısı tipi) değişkeni ve sözel terimleri.
        self.variables['ChestPain'] = FuzzyVariable('ChestPain', np.arange(0, 4, 0.1),
            {'NoPain': [0, 0, 0.5], 'NonAnginal': [0.5, 1, 1.5], 
             'Atypical': [1.5, 2, 2.5], 'Typical': [2.5, 3, 3.5]})
    
    # CSV dosyasından çıkarım kurallarını yükler.
    def load_rules(self, filepath):
        # Pandas ile CSV dosyasını oku. Hatalı satırları atla.
        df = pd.read_csv(filepath, engine="python", on_bad_lines="skip", quoting=3)
        
        # Dosyadaki her bir satır (kural) için döngü.
        for idx in range(len(df)):
            row0, row1 = df.iloc[idx, 0], df.iloc[idx, 1]  # Kuralın 'EĞER' ve 'İSE' kısımlarını al.
            if pd.isna(row0) or pd.isna(row1):  # Eğer satır boşsa atla.
                continue
            
            antecedent = {}  # Kuralın 'EĞER' (antecedent) kısmını saklamak için sözlük.
            # 'EĞER' kısmını ' AND ' ile ayırarak her bir koşulu işle.
            for p in str(row0).replace('"', '').split(' AND '):
                if '=' in p:  # Koşul 'degisken=deger' formatında mı kontrol et.
                    k, v = p.split('=')  # Değişkeni ve değeri ayır.
                    # Değeri, yazım hatası düzeltme sözlüğünü kullanarak standartlaştır.
                    antecedent[k.strip()] = TYPO_CORRECTIONS.get(v.strip(), v.strip())
            
            # Kuralın 'İSE' (consequent) kısmını al ve standartlaştır.
            consequent = TYPO_CORRECTIONS.get(str(row1).replace('"', '').strip(), 
                                               str(row1).replace('"', '').strip())
            # Tamamlanan kuralı listeye ekle.
            self.rules.append({'antecedent': antecedent, 'consequent': consequent})
        
        return len(self.rules)  # Yüklenen toplam kural sayısını döndür.
    
    # Sayısal girdileri bulanıklaştırır.
    def fuzzify_inputs(self, numeric_inputs):
        # Gelen her bir sayısal girdi için ilgili FuzzyVariable'ın fuzzify metodunu çağır.
        return {var: self.variables[var].fuzzify(val) 
                for var, val in numeric_inputs.items() if var in self.variables}
    
    # Kuralları değerlendirerek aktivasyon derecelerini hesaplar.
    def evaluate_rules(self, fuzzified):
        activations = []  # Aktif olan kuralları ve aktivasyon derecelerini saklamak için liste.
        for rule in self.rules:  # Her bir kural için döngü.
            activation = 1.0  # Kuralın başlangıç aktivasyon derecesi.
            # Kuralın 'EĞER' kısmındaki her bir koşul için döngü.
            for var, term in rule['antecedent'].items():
                if var in fuzzified and term in fuzzified[var]:
                    # 'VE' işlemi için, koşulların üyelik derecelerinin minimum olanını al.
                    activation = min(activation, fuzzified[var][term])
                else:
                    activation = 0  # Eğer bir koşul sağlanmıyorsa kural aktif değildir.
                    break
            if activation > 0:  # Eğer kural aktifse listeye ekle.
                activations.append({'activation': activation, 'consequent': rule['consequent']})
        return activations
    
    # Aktif kuralların çıktılarını birleştirir (aggregation).
    def aggregate(self, activations):
        aggregated = np.zeros_like(self.risk_universe)  # Boş bir birleşik çıktı alanı oluştur.
        for rule in activations:  # Her aktif kural için.
            # Kuralın çıktısındaki üyelik fonksiyonunu, kuralın aktivasyon derecesiyle kırp.
            clipped = np.minimum(rule['activation'], self.risk_mfs[rule['consequent']])
            # Kırpılmış çıktıyı, genel birleşik çıktıya ekle (maksimum operatörü ile).
            aggregated = np.maximum(aggregated, clipped)
        return aggregated
    
    # Birleşik bulanık çıktıyı net bir sayısal değere dönüştürür (defuzzification).
    def defuzzify_hybrid(self, aggregated):
        if np.sum(aggregated) == 0:  # Eğer hiç aktivasyon yoksa varsayılan bir değer döndür.
            return 5.0
        results = []
        # Üç farklı durulaştırma metodunu dene: centroid, bisector, mom.
        for method in ['centroid', 'bisector', 'mom']:
            try:
                r = fuzz.defuzz(self.risk_universe, aggregated, method)
                if not np.isnan(r):  # Eğer sonuç geçerliyse listeye ekle.
                    results.append(r)
            except:  # Hata olursa görmezden gel.
                pass
        # Geçerli sonuçların ortalamasını alarak hibrit bir sonuç elde et.
        return np.mean(results) if results else 5.0
    
    # Tüm çıkarım sürecini yöneten ana metot.
    def infer(self, numeric_inputs):
        fuzzified = self.fuzzify_inputs(numeric_inputs)  # 1. Adım: Bulanıklaştırma
        activations = self.evaluate_rules(fuzzified)  # 2. Adım: Kural Değerlendirme
        
        # Eğer hiçbir kural aktif olmadıysa, yedek bir risk hesaplama metodu kullan.
        if not activations:
            score = self._calculate_risk_score(fuzzified)  # Yedek skor hesapla.
            # Skora göre kategoriyi belirle.
            if score < 3: category = 'Healthy'
            elif score < 5: category = 'LowRisk'
            elif score < 7: category = 'MediumRisk'
            else: category = 'HighRisk'
            return score, category, fuzzified, []
        
        aggregated = self.aggregate(activations)  # 3. Adım: Birleştirme
        score = self.defuzzify_hybrid(aggregated)  # 4. Adım: Durulaştırma
        
        # Nihai skora göre kategoriyi belirle.
        if score < 3: category = 'Healthy'
        elif score < 5: category = 'LowRisk'
        elif score < 7: category = 'MediumRisk'
        else: category = 'HighRisk'
        
        # Sonuçları döndür.
        return score, category, fuzzified, activations
    
    # Yedek risk skoru hesaplama metodu.
    def _calculate_risk_score(self, fuzzified):
        """Hiçbir kural uymadığında, risk faktörlerine basit ağırlıklar vererek skor hesaplar."""
        risk_weights = {  # Her bir sözel terim için basit risk ağırlıkları.
            'Age': {'Young': 0, 'Mid': 1, 'Old': 2, 'VeryOld': 3},
            'BloodPressure': {'Medium': 0, 'High': 1.5, 'VeryHigh': 3},
            'HbA1c': {'VeryHealthy': 0, 'Healthy': 1, 'High': 2.5},
            'LDL': {'VeryHealthy': 0, 'Healthy': 0.5, 'High': 1.5, 'VeryHigh': 2, 'XHigh': 2.5},
            'HDL': {'Healthy': 0, 'Low': 1.5},
            'HeartRate': {'VeryHealthy': 0, 'Healthy': 0.5, 'High': 1.5},
            'ChestPain': {'NoPain': 0, 'NonAnginal': 1, 'Atypical': 2, 'Typical': 3}
        }
        
        total_score = 0
        # Her bir değişken için.
        for var, memberships in fuzzified.items():
            if memberships and var in risk_weights:
                # En yüksek üyelik derecesine sahip terimi (dominant terim) bul.
                dominant = max(memberships, key=memberships.get)
                if dominant in risk_weights[var]:
                    total_score += risk_weights[var][dominant]  # Ağırlığı toplam skora ekle.
        
        # Skoru 0-10 aralığına normalize et.
        max_possible = 3 + 3 + 2.5 + 2.5 + 1.5 + 1.5 + 3  # Olası maksimum skor (17).
        normalized = (total_score / max_possible) * 10
        return min(normalized, 10)  # Skorun 10'u geçmediğinden emin ol.
    
    # Kategorik girdilerle çıkarım yapar.
    def infer_categorical(self, categorical_inputs):
        """Sözel girdileri (örn: 'Age'='Old') sayısal değerlere çevirip çıkarım yapar."""
        # Her sözel terimin temsil ettiği sayısal merkez noktaları.
        centers = {
            'Age': {'Young': 35, 'Mid': 55, 'Old': 72, 'VeryOld': 90},
            'HbA1c': {'VeryHealthy': 5.2, 'Healthy': 7.5, 'High': 10},
            'LDL': {'VeryHealthy': 65, 'Healthy': 90, 'High': 125, 'VeryHigh': 165, 'XHigh': 210},
            'HDL': {'Low': 32, 'Healthy': 60},
            'HeartRate': {'VeryHealthy': 62, 'Healthy': 82, 'High': 130},
            'BloodPressure': {'Medium': 100, 'High': 135, 'VeryHigh': 175},
            'ChestPain': {'NoPain': 0, 'NonAnginal': 1, 'Atypical': 2, 'Typical': 3}
        }
        # Kategorik girdileri sayısal merkez değerlerine dönüştür.
        numeric = {var: centers[var][term] for var, term in categorical_inputs.items() 
                   if var in centers and term in centers[var]}
        # Sayısal değerlerle ana çıkarım metodunu çağır.
        return self.infer(numeric)


# === DEĞERLENDİRME FONKSİYONU: SİSTEMİ TEST EDER ===

def evaluate():
    """Bulanık sistemi test verileriyle çalıştırır ve performansını değerlendirir."""
    print("=" * 60)
    print("Kalp Hastaligi Risk Tahmin Sistemi")
    print("Mamdani FIS + Hibrit Durulaştirma")
    print("=" * 60)
    
    fis = MamdaniFIS()  # Bulanık çıkarım sisteminden bir nesne oluştur.
    print(f"\nKural sayisi: {fis.load_rules(RULES_FILE)}")  # Kuralları yükle ve sayısını yazdır.
    
    df = pd.read_csv(TEST_FILE, engine="python", on_bad_lines="skip", quoting=3)  # Test verisini yükle.
    print(f"Test verisi: {len(df)}")
    
    y_true, y_pred = [], []  # Gerçek ve tahmin edilen sonuçları saklamak için listeler.
    
    # Test verisindeki her bir örnek için döngü.
    for idx in range(len(df)):
        row0, row1 = df.iloc[idx, 0], df.iloc[idx, 1]  # Girdileri ve beklenen çıktıyı al.
        if pd.isna(row0) or pd.isna(row1):  # Boş satırları atla.
            continue
        
        inputs = {}  # Girdileri saklamak için sözlük.
        # Girdi satırını işle ve standartlaştır.
        for p in str(row0).replace('"', '').split(' AND '):
            if '=' in p:
                k, v = p.split('=')
                inputs[k.strip()] = TYPO_CORRECTIONS.get(v.strip(), v.strip())
        
        # Beklenen çıktıyı (gerçek sonuç) al ve standartlaştır.
        expected = TYPO_CORRECTIONS.get(str(row1).replace('"', '').strip(), 
                                         str(row1).replace('"', '').strip())
        # Kategorik girdilerle sistemden bir tahmin yap.
        _, prediction, _, _ = fis.infer_categorical(inputs)
        
        y_true.append(expected)  # Gerçek sonucu listeye ekle.
        y_pred.append(prediction)  # Tahmin edilen sonucu listeye ekle.
    
    # Modelin doğruluğunu hesapla ve yüzde olarak yazdır.
    accuracy = accuracy_score(y_true, y_pred) * 100
    print(f"\nDogruluk: %{accuracy:.2f}")
    # Sınıflandırma raporunu (precision, recall, f1-score) yazdır.
    print("\n" + classification_report(y_true, y_pred, zero_division=0))
    
    # Karmaşıklık matrisini (confusion matrix) oluştur ve yazdır.
    labels = ["Healthy", "LowRisk", "MediumRisk", "HighRisk"]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    print("Karisiklik Matrisi:")
    print(f"{"":12} {"Healthy":>8} {"LowRisk":>8} {"MediumRisk":>10} {"HighRisk":>8}")
    for i, l in enumerate(labels):
        print(f"{l:12} {cm[i,0]:>8} {cm[i,1]:>8} {cm[i,2]:>10} {cm[i,3]:>8}")
    
    return fis, accuracy


# === ANA ÇALIŞTIRMA BLOĞU ===

# Bu betik doğrudan çalıştırıldığında aşağıdaki kod bloğu çalışır.
if __name__ == "__main__":
    evaluate()  # Değerlendirme fonksiyonunu çağır.
    print("\n" + "=" * 60)
    print("Test Tamamlandi")
    print("=" * 60)