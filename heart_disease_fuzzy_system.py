"""
Kalp Hastaligi Risk Tahmin Sistemi
Mamdani Bulanik Cikarim + Hibrit Durulaştirma
"""

import numpy as np
import pandas as pd
import skfuzzy as fuzz
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import warnings

warnings.filterwarnings('ignore')

RULES_FILE = "inference_rules_corrected.csv"
TEST_FILE = "testing_data_set.csv"

TYPO_CORRECTIONS = {
    "HIgh": "High", "VeryHIgh": "VeryHigh", "XHIgh": "XHigh",
    "Normal": "Medium", "A typical": "Atypical", 
    "Very Healthy": "VeryHealthy", "Very Old": "VeryOld", "ExtraHigh": "XHigh"
}


class FuzzyVariable:
    def __init__(self, name, universe, mfs):
        self.name = name
        self.universe = universe
        self.mfs = {term: fuzz.trimf(universe, params) for term, params in mfs.items()}
    
    def fuzzify(self, value):
        return {term: float(fuzz.interp_membership(self.universe, mf, value)) 
                for term, mf in self.mfs.items()}


class MamdaniFIS:
    def __init__(self):
        self.variables = {}
        self.rules = []
        self.risk_universe = np.arange(0, 10.01, 0.01)
        self.risk_mfs = {
            'Healthy': fuzz.trimf(self.risk_universe, [0, 1.5, 3]),
            'LowRisk': fuzz.trimf(self.risk_universe, [2, 4, 6]),
            'MediumRisk': fuzz.trimf(self.risk_universe, [4, 6, 8]),
            'HighRisk': fuzz.trimf(self.risk_universe, [6, 8.5, 10])
        }
        self._define_variables()
    
    def _define_variables(self):
        self.variables['Age'] = FuzzyVariable('Age', np.arange(20, 101, 1),
            {'Young': [20, 32, 45], 'Mid': [40, 52, 65], 'Old': [60, 72, 85], 'VeryOld': [80, 95, 100]})
        
        self.variables['HbA1c'] = FuzzyVariable('HbA1c', np.arange(4, 14.1, 0.1),
            {'VeryHealthy': [4, 5, 6.5], 'Healthy': [6, 7.5, 9], 'High': [8, 11, 14]})
        
        self.variables['LDL'] = FuzzyVariable('LDL', np.arange(50, 251, 1),
            {'VeryHealthy': [50, 65, 80], 'Healthy': [70, 90, 110], 'High': [100, 125, 150],
             'VeryHigh': [140, 165, 190], 'XHigh': [180, 215, 250]})
        
        self.variables['HDL'] = FuzzyVariable('HDL', np.arange(20, 101, 1),
            {'Low': [20, 30, 45], 'Healthy': [40, 65, 100]})
        
        self.variables['HeartRate'] = FuzzyVariable('HeartRate', np.arange(50, 181, 1),
            {'VeryHealthy': [50, 60, 75], 'Healthy': [65, 82, 100], 'High': [90, 135, 180]})
        
        self.variables['BloodPressure'] = FuzzyVariable('BloodPressure', np.arange(80, 201, 1),
            {'Medium': [80, 100, 120], 'High': [110, 135, 160], 'VeryHigh': [150, 175, 200]})
        
        self.variables['ChestPain'] = FuzzyVariable('ChestPain', np.arange(0, 4, 0.1),
            {'NoPain': [0, 0, 0.5], 'NonAnginal': [0.5, 1, 1.5], 
             'Atypical': [1.5, 2, 2.5], 'Typical': [2.5, 3, 3.5]})
    
    def load_rules(self, filepath):
        df = pd.read_csv(filepath, engine="python", on_bad_lines="skip", quoting=3)
        
        for idx in range(len(df)):
            row0, row1 = df.iloc[idx, 0], df.iloc[idx, 1]
            if pd.isna(row0) or pd.isna(row1):
                continue
            
            antecedent = {}
            for p in str(row0).replace('"', '').split(' AND '):
                if '=' in p:
                    k, v = p.split('=')
                    antecedent[k.strip()] = TYPO_CORRECTIONS.get(v.strip(), v.strip())
            
            consequent = TYPO_CORRECTIONS.get(str(row1).replace('"', '').strip(), 
                                               str(row1).replace('"', '').strip())
            self.rules.append({'antecedent': antecedent, 'consequent': consequent})
        
        return len(self.rules)
    
    def fuzzify_inputs(self, numeric_inputs):
        return {var: self.variables[var].fuzzify(val) 
                for var, val in numeric_inputs.items() if var in self.variables}
    
    def evaluate_rules(self, fuzzified):
        activations = []
        for rule in self.rules:
            activation = 1.0
            for var, term in rule['antecedent'].items():
                if var in fuzzified and term in fuzzified[var]:
                    activation = min(activation, fuzzified[var][term])
                else:
                    activation = 0
                    break
            if activation > 0:
                activations.append({'activation': activation, 'consequent': rule['consequent']})
        return activations
    
    def aggregate(self, activations):
        aggregated = np.zeros_like(self.risk_universe)
        for rule in activations:
            clipped = np.minimum(rule['activation'], self.risk_mfs[rule['consequent']])
            aggregated = np.maximum(aggregated, clipped)
        return aggregated
    
    def defuzzify_hybrid(self, aggregated):
        if np.sum(aggregated) == 0:
            return 5.0
        results = []
        for method in ['centroid', 'bisector', 'mom']:
            try:
                r = fuzz.defuzz(self.risk_universe, aggregated, method)
                if not np.isnan(r):
                    results.append(r)
            except:
                pass
        return np.mean(results) if results else 5.0
    
    def infer(self, numeric_inputs):
        fuzzified = self.fuzzify_inputs(numeric_inputs)
        activations = self.evaluate_rules(fuzzified)
        
        if not activations:
            # Fallback: Risk faktorlerine gore hesapla
            score = self._calculate_risk_score(fuzzified)
            if score < 3: category = 'Healthy'
            elif score < 5: category = 'LowRisk'
            elif score < 7: category = 'MediumRisk'
            else: category = 'HighRisk'
            return score, category, fuzzified, []
        
        aggregated = self.aggregate(activations)
        score = self.defuzzify_hybrid(aggregated)
        
        if score < 3: category = 'Healthy'
        elif score < 5: category = 'LowRisk'
        elif score < 7: category = 'MediumRisk'
        else: category = 'HighRisk'
        
        return score, category, fuzzified, activations
    
    def _calculate_risk_score(self, fuzzified):
        """Kural bulunamadiginda risk faktorlerine gore skor hesapla"""
        risk_weights = {
            'Age': {'Young': 0, 'Mid': 1, 'Old': 2, 'VeryOld': 3},
            'BloodPressure': {'Medium': 0, 'High': 1.5, 'VeryHigh': 3},
            'HbA1c': {'VeryHealthy': 0, 'Healthy': 1, 'High': 2.5},
            'LDL': {'VeryHealthy': 0, 'Healthy': 0.5, 'High': 1.5, 'VeryHigh': 2, 'XHigh': 2.5},
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
        
        # 0-10 araligina normalize et
        max_possible = 3 + 3 + 2.5 + 2.5 + 1.5 + 1.5 + 3  # 17
        normalized = (total_score / max_possible) * 10
        return min(normalized, 10)
    
    def infer_categorical(self, categorical_inputs):
        centers = {
            'Age': {'Young': 35, 'Mid': 55, 'Old': 72, 'VeryOld': 90},
            'HbA1c': {'VeryHealthy': 5.2, 'Healthy': 7.5, 'High': 10},
            'LDL': {'VeryHealthy': 65, 'Healthy': 90, 'High': 125, 'VeryHigh': 165, 'XHigh': 210},
            'HDL': {'Low': 32, 'Healthy': 60},
            'HeartRate': {'VeryHealthy': 62, 'Healthy': 82, 'High': 130},
            'BloodPressure': {'Medium': 100, 'High': 135, 'VeryHigh': 175},
            'ChestPain': {'NoPain': 0, 'NonAnginal': 1, 'Atypical': 2, 'Typical': 3}
        }
        numeric = {var: centers[var][term] for var, term in categorical_inputs.items() 
                   if var in centers and term in centers[var]}
        return self.infer(numeric)


def evaluate():
    print("=" * 60)
    print("Kalp Hastaligi Risk Tahmin Sistemi")
    print("Mamdani FIS + Hibrit Durulaştirma")
    print("=" * 60)
    
    fis = MamdaniFIS()
    print(f"\nKural sayisi: {fis.load_rules(RULES_FILE)}")
    
    df = pd.read_csv(TEST_FILE, engine="python", on_bad_lines="skip", quoting=3)
    print(f"Test verisi: {len(df)}")
    
    y_true, y_pred = [], []
    
    for idx in range(len(df)):
        row0, row1 = df.iloc[idx, 0], df.iloc[idx, 1]
        if pd.isna(row0) or pd.isna(row1):
            continue
        
        inputs = {}
        for p in str(row0).replace('"', '').split(' AND '):
            if '=' in p:
                k, v = p.split('=')
                inputs[k.strip()] = TYPO_CORRECTIONS.get(v.strip(), v.strip())
        
        expected = TYPO_CORRECTIONS.get(str(row1).replace('"', '').strip(), 
                                         str(row1).replace('"', '').strip())
        _, prediction, _, _ = fis.infer_categorical(inputs)
        
        y_true.append(expected)
        y_pred.append(prediction)
    
    accuracy = accuracy_score(y_true, y_pred) * 100
    print(f"\nDogruluk: %{accuracy:.2f}")
    print("\n" + classification_report(y_true, y_pred, zero_division=0))
    
    labels = ["Healthy", "LowRisk", "MediumRisk", "HighRisk"]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    print("Karisiklik Matrisi:")
    print(f"{'':12} {'Healthy':>8} {'LowRisk':>8} {'MediumRisk':>10} {'HighRisk':>8}")
    for i, l in enumerate(labels):
        print(f"{l:12} {cm[i,0]:>8} {cm[i,1]:>8} {cm[i,2]:>10} {cm[i,3]:>8}")
    
    return fis, accuracy


if __name__ == "__main__":
    evaluate()
    print("\n" + "=" * 60)
    print("Test Tamamlandi")
    print("=" * 60)
