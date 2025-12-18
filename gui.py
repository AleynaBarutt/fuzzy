"""
Kalp Hastalığı Risk Tahmin Sistemi - GUI
Gerçek Mamdani FIS + Hibrit Durulaştırma
"""

import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import pandas as pd
import skfuzzy as fuzz

RULES_FILE = "inference_rules_corrected.csv"

TYPO_CORRECTIONS = {
    "HIgh": "High", "VeryHIgh": "VeryHigh", "XHIgh": "XHigh",
    "Normal": "Medium", "A typical": "Atypical", 
    "Very Healthy": "VeryHealthy", "Very Old": "VeryOld", "ExtraHigh": "XHigh"
}

VARIABLES = {
    "Age": {
        "label": "Yaş",
        "unit": "yıl",
        "range": (20, 100),
        "default": 45,
        "mfs": {"Young": [20, 32, 45], "Mid": [40, 52, 65], "Old": [60, 72, 85], "VeryOld": [80, 95, 100]}
    },
    "BloodPressure": {
        "label": "Sistolik Tansiyon",
        "unit": "mmHg",
        "range": (80, 200),
        "default": 120,
        "mfs": {"Medium": [80, 100, 120], "High": [110, 135, 160], "VeryHigh": [150, 175, 200]}
    },
    "HbA1c": {
        "label": "HbA1c (Şeker)",
        "unit": "%",
        "range": (4.0, 14.0),
        "default": 5.5,
        "mfs": {"VeryHealthy": [4, 5, 6.5], "Healthy": [6, 7.5, 9], "High": [8, 11, 14]}
    },
    "LDL": {
        "label": "LDL Kolesterol",
        "unit": "mg/dL",
        "range": (50, 250),
        "default": 100,
        "mfs": {"VeryHealthy": [50, 65, 80], "Healthy": [70, 90, 110], "High": [100, 125, 150], 
                "VeryHigh": [140, 165, 190], "XHigh": [180, 215, 250]}
    },
    "HDL": {
        "label": "HDL Kolesterol",
        "unit": "mg/dL",
        "range": (20, 100),
        "default": 50,
        "mfs": {"Low": [20, 30, 45], "Healthy": [40, 65, 100]}
    },
    "HeartRate": {
        "label": "Nabız",
        "unit": "bpm",
        "range": (50, 180),
        "default": 75,
        "mfs": {"VeryHealthy": [50, 60, 75], "Healthy": [65, 82, 100], "High": [90, 135, 180]}
    },
    "ChestPain": {
        "label": "Göğüs Ağrısı",
        "unit": "",
        "range": (0, 3),
        "default": 0,
        "type": "combo",
        "options": ["NoPain", "NonAnginal", "Atypical", "Typical"],
        "mfs": {"NoPain": [0, 0, 0.5], "NonAnginal": [0.5, 1, 1.5], "Atypical": [1.5, 2, 2.5], "Typical": [2.5, 3, 3.5]}
    }
}

RISK_INFO = {
    "Healthy": {"label": "SAĞLIKLI", "color": "#27AE60", "score": "0-3", "desc": "Risk faktörü düşük"},
    "LowRisk": {"label": "DÜŞÜK RİSK", "color": "#F39C12", "score": "3-5", "desc": "Yaşam tarzı düzenlemesi önerilir"},
    "MediumRisk": {"label": "ORTA RİSK", "color": "#E67E22", "score": "5-7", "desc": "Uzman kontrolü gerekli"},
    "HighRisk": {"label": "YÜKSEK RİSK", "color": "#E74C3C", "score": "7-10", "desc": "Acil tıbbi değerlendirme"}
}


class MamdaniFIS:
    """Gerçek Mamdani Bulanık Çıkarım Sistemi"""
    
    def __init__(self):
        self.rules = []
        self.risk_universe = np.arange(0, 10.01, 0.01)
        self.risk_mfs = {
            'Healthy': fuzz.trimf(self.risk_universe, [0, 1.5, 3]),
            'LowRisk': fuzz.trimf(self.risk_universe, [2, 4, 6]),
            'MediumRisk': fuzz.trimf(self.risk_universe, [4, 6, 8]),
            'HighRisk': fuzz.trimf(self.risk_universe, [6, 8.5, 10])
        }
        self.load_rules()
    
    def load_rules(self):
        try:
            df = pd.read_csv(RULES_FILE, engine="python", on_bad_lines="skip", quoting=3)
            for idx in range(len(df)):
                row0 = df.iloc[idx, 0]
                row1 = df.iloc[idx, 1]
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
        except Exception as e:
            messagebox.showerror("Hata", f"Kural dosyası yüklenemedi: {e}")
    
    def fuzzify(self, var_name, value):
        """Sayısal değerin tüm kümelerdeki üyelik derecelerini hesaplar"""
        if var_name not in VARIABLES or "mfs" not in VARIABLES[var_name]:
            return {}
        
        mfs = VARIABLES[var_name]["mfs"]
        min_val, max_val = VARIABLES[var_name]["range"]
        universe = np.arange(min_val, max_val + 0.1, 0.1)
        
        memberships = {}
        for term, params in mfs.items():
            mf = fuzz.trimf(universe, params)
            memberships[term] = float(fuzz.interp_membership(universe, mf, value))
        
        return memberships
    
    def infer(self, numeric_inputs):
        """Tam Mamdani çıkarım süreci"""
        
        # 1. Bulanıklaştırma
        fuzzified = {}
        for var_name, value in numeric_inputs.items():
            fuzzified[var_name] = self.fuzzify(var_name, value)
        
        # 2. Kural değerlendirme
        rule_activations = []
        for rule in self.rules:
            activation = 1.0
            
            for var_name, term in rule['antecedent'].items():
                if var_name in fuzzified and term in fuzzified[var_name]:
                    activation = min(activation, fuzzified[var_name][term])
                else:
                    activation = 0
                    break
            
            if activation > 0:
                rule_activations.append({'activation': activation, 'consequent': rule['consequent']})
        
        # 3. Çıktı birleştirme veya fallback
        if len(rule_activations) == 0:
            # Fallback: Risk faktorlerine gore hesapla
            score = self._calculate_risk_score(fuzzified)
        else:
            aggregated = np.zeros_like(self.risk_universe)
            for rule in rule_activations:
                clipped = np.minimum(rule['activation'], self.risk_mfs[rule['consequent']])
                aggregated = np.maximum(aggregated, clipped)
            
            # 4. Hibrit durulaştırma
            if np.sum(aggregated) == 0:
                score = self._calculate_risk_score(fuzzified)
            else:
                results = []
                for method in ['centroid', 'bisector', 'mom']:
                    try:
                        r = fuzz.defuzz(self.risk_universe, aggregated, method)
                        if not np.isnan(r):
                            results.append(r)
                    except:
                        pass
                score = np.mean(results) if results else self._calculate_risk_score(fuzzified)
        
        # Sınıflandırma
        if score < 3:
            category = 'Healthy'
        elif score < 5:
            category = 'LowRisk'
        elif score < 7:
            category = 'MediumRisk'
        else:
            category = 'HighRisk'
        
        return score, category, fuzzified, len(rule_activations)
    
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
        max_possible = 17  # Sum of max weights
        normalized = (total_score / max_possible) * 10
        return min(normalized, 10)


class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Kalp Hastalığı Risk Tahmin Sistemi")
        self.root.geometry("620x820")
        self.root.resizable(False, False)
        self.root.configure(bg="#ECF0F1")
        
        self.fis = MamdaniFIS()
        self.inputs = {}
        
        self.create_ui()
    
    def create_ui(self):
        # Başlık
        header = tk.Frame(self.root, bg="#2C3E50", height=70)
        header.pack(fill="x")
        header.pack_propagate(False)
        
        tk.Label(header, text="Kalp Hastalığı Risk Tahmini", 
                font=("Segoe UI", 18, "bold"), bg="#2C3E50", fg="white").pack(pady=12)
        tk.Label(header, text="Mamdani FIS + Hibrit Durulaştırma", 
                font=("Segoe UI", 9), bg="#2C3E50", fg="#BDC3C7").pack()
        
        # Giriş bölümü
        input_frame = tk.LabelFrame(self.root, text=" Hasta Verileri ", 
                                   font=("Segoe UI", 10, "bold"), bg="#ECF0F1", 
                                   fg="#2C3E50", padx=15, pady=10)
        input_frame.pack(fill="x", padx=20, pady=10)
        
        for var, info in VARIABLES.items():
            row = tk.Frame(input_frame, bg="#ECF0F1")
            row.pack(fill="x", pady=5)
            
            label = f"{info['label']}" + (f" ({info['unit']})" if info['unit'] else "")
            tk.Label(row, text=label, font=("Segoe UI", 10), bg="#ECF0F1", 
                    fg="#34495E", width=18, anchor="w").pack(side="left")
            
            if info.get("type") == "combo":
                combo = ttk.Combobox(row, values=info["options"], state="readonly", 
                                    width=12, font=("Segoe UI", 10))
                combo.current(0)
                combo.pack(side="right", padx=5)
                self.inputs[var] = ("combo", combo)
            else:
                frame = tk.Frame(row, bg="#ECF0F1")
                frame.pack(side="right")
                
                entry = tk.Entry(frame, width=7, font=("Segoe UI", 10), justify="center")
                entry.insert(0, str(info["default"]))
                entry.pack(side="left", padx=3)
                
                min_val, max_val = info["range"]
                slider = ttk.Scale(frame, from_=min_val, to=max_val, length=100,
                                  command=lambda v, e=entry: self.update_entry(e, v))
                slider.set(info["default"])
                slider.pack(side="left")
                
                self.inputs[var] = ("numeric", entry, slider)
        
        # Tahmin butonu
        btn = tk.Button(self.root, text="RİSK ANALİZİ YAP", font=("Segoe UI", 12, "bold"),
                       bg="#3498DB", fg="white", padx=30, pady=10, 
                       relief="flat", cursor="hand2", command=self.predict)
        btn.pack(pady=12)
        btn.bind("<Enter>", lambda e: btn.config(bg="#2980B9"))
        btn.bind("<Leave>", lambda e: btn.config(bg="#3498DB"))
        
        # Bulanıklaştırma sonuçları
        fuzzy_frame = tk.LabelFrame(self.root, text=" Bulanıklaştırma (Fuzzification) ", 
                                   font=("Segoe UI", 10, "bold"), bg="#ECF0F1", 
                                   fg="#2C3E50", padx=10, pady=8)
        fuzzy_frame.pack(fill="x", padx=20, pady=5)
        
        self.fuzzy_text = tk.Text(fuzzy_frame, height=7, font=("Consolas", 9), 
                                  bg="white", fg="#2C3E50", relief="flat")
        self.fuzzy_text.pack(fill="x", pady=3)
        self.fuzzy_text.insert("1.0", "Değerler girildiğinde üyelik dereceleri gösterilecek...")
        self.fuzzy_text.config(state="disabled")
        
        # Sonuç bölümü
        result_frame = tk.LabelFrame(self.root, text=" Sonuç ", 
                                    font=("Segoe UI", 10, "bold"), bg="#ECF0F1", 
                                    fg="#2C3E50", padx=15, pady=12)
        result_frame.pack(fill="x", padx=20, pady=10)
        
        self.result_box = tk.Frame(result_frame, bg="#BDC3C7", height=70)
        self.result_box.pack(fill="x", pady=5)
        self.result_box.pack_propagate(False)
        
        self.result_label = tk.Label(self.result_box, text="Analiz bekleniyor...", 
                                    font=("Segoe UI", 18, "bold"), bg="#BDC3C7", fg="#7F8C8D")
        self.result_label.pack(expand=True)
        
        # Detay
        detail_frame = tk.Frame(result_frame, bg="#ECF0F1")
        detail_frame.pack(fill="x", pady=8)
        
        self.score_lbl = self.create_box(detail_frame, "Risk Skoru", "-")
        self.rules_lbl = self.create_box(detail_frame, "Aktif Kural", "-")
        self.desc_lbl = self.create_box(detail_frame, "Öneri", "-")
        
        # Footer
        tk.Label(self.root, text=f"{len(self.fis.rules)} Kural | Centroid + Bisector + MOM", 
                font=("Segoe UI", 8), bg="#ECF0F1", fg="#95A5A6").pack(side="bottom", pady=8)
    
    def create_box(self, parent, title, value):
        box = tk.Frame(parent, bg="white", padx=8, pady=6)
        box.pack(side="left", expand=True, fill="x", padx=3)
        tk.Label(box, text=title, font=("Segoe UI", 8), bg="white", fg="#7F8C8D").pack()
        lbl = tk.Label(box, text=value, font=("Segoe UI", 10, "bold"), bg="white", fg="#2C3E50")
        lbl.pack()
        return lbl
    
    def update_entry(self, entry, value):
        entry.delete(0, tk.END)
        entry.insert(0, f"{float(value):.1f}")
    
    def predict(self):
        try:
            numeric_inputs = {}
            
            for var, widgets in self.inputs.items():
                if widgets[0] == "combo":
                    # Kategorik → Sayısal
                    idx = widgets[1].current()
                    numeric_inputs[var] = idx
                else:
                    numeric_inputs[var] = float(widgets[1].get())
            
            score, category, fuzzified, rule_count = self.fis.infer(numeric_inputs)
            info = RISK_INFO.get(category, RISK_INFO["MediumRisk"])
            
            # Bulanıklaştırma sonuçlarını göster
            self.fuzzy_text.config(state="normal")
            self.fuzzy_text.delete("1.0", tk.END)
            
            for var, memberships in fuzzified.items():
                if memberships:
                    dominant = max(memberships, key=memberships.get)
                    line = f"{var:15} → {dominant:12} (μ = {memberships[dominant]:.2f})\n"
                    self.fuzzy_text.insert(tk.END, line)
            
            self.fuzzy_text.config(state="disabled")
            
            # Sonuç
            self.result_box.config(bg=info["color"])
            self.result_label.config(text=info["label"], bg=info["color"], fg="white")
            self.score_lbl.config(text=f"{score:.2f}")
            self.rules_lbl.config(text=str(rule_count))
            self.desc_lbl.config(text=info["desc"])
            
        except ValueError:
            messagebox.showerror("Hata", "Lütfen geçerli sayısal değerler girin!")


if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
