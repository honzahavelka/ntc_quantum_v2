import os
import csv
import datetime
import numpy as np
import matplotlib.pyplot as plt
from qiskit.visualization import plot_histogram

class DataHandler:
    def __init__(self, base_dir="../results/oaa", file_name="result"):
        """Vytvoří strukturu složek a inicializuje soubory pro aktuální běh."""
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)
        
        # Unikátní časové razítko pro propojení všech souborů z jednoho běhu
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_path = os.path.join(self.base_dir, f"{file_name}.csv")
        self.txt_path = os.path.join(self.base_dir, f"{file_name}.txt")
        self.plot_path = os.path.join(self.base_dir, f"{file_name}.png")
        
        # 1. Inicializace CSV hlavičky
        with open(self.csv_path, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["k", "target_prob", "measured_success_prob", "backend"])
            
        # 2. Inicializace TXT protokolu s hlavičkou
        with open(self.txt_path, mode='w', encoding='utf-8') as f:
            f.write(f"=== OAA EXPERIMENT SUMMARY ({self.timestamp}) ===\n")
            f.write("Tento soubor obsahuje sloučená data pro všechny iterace k.\n\n")
            
        # 3. Paměť pro finální vygenerování grafů
        self.history = []

    def save_experiment_data(self, k, target_prob, optimal_k, theoretical_angle, 
                             prob_good_exact, measured_good_prob, counts, shots, backend_name):
        """
        Zápis dat iterace do CSV a TXT, a uložení do paměti pro finální graf.
        """
        # A. Uložení řádku do CSV
        with open(self.csv_path, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([k, target_prob, measured_good_prob, backend_name])
            
        # B. Připsání bloku do sloučeného TXT s jasným oddělovačem
        with open(self.txt_path, mode='a', encoding='utf-8') as f:
            f.write("="*60 + "\n")
            f.write(f"OAA EXPERIMENT REPORT - Iteration k = {k}\n")
            f.write("="*60 + "\n")
            f.write("[HARDWARE & PARAMETERS]\n")
            f.write(f"Backend used:          {backend_name}\n")
            f.write(f"Shots:                 {shots}\n")
            f.write(f"Target Initial Prob:   {target_prob * 100:.2f} %\n")
            f.write(f"Optimal Iterations:    {optimal_k}\n\n")
            f.write("[RESULTS & METRICS]\n")
            f.write(f"Theoretical Angle:        {theoretical_angle:.2f}°\n")
            f.write(f"Mathematical Probability: {prob_good_exact * 100:.2f} % (Success)\n")
            f.write(f"Measured Probability:     {measured_good_prob * 100:.2f} %\n")
            f.write(f"Absolute Error:           {abs(prob_good_exact - measured_good_prob) * 100:.2f} %\n")
            f.write(f"Raw Counts:               {counts}\n\n")

        # C. Uložení do paměti pro finální spojitý graf
        self.history.append({
            'k': k, 'optimal_k': optimal_k, 'theoretical_angle': theoretical_angle,
            'prob_good_exact': prob_good_exact, 'measured_good_prob': measured_good_prob,
            'counts': counts, 'shots': shots
        })
        
        print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Zpracována iterace k={k}")

    def generate_comparison_plot(self):
        """
        Vykreslí jeden vertikálně roztáhlý PNG soubor obsahující grafy
        (Vektor + Histogram) pro všechny testované iterace k pod sebou.
        """
        num_k = len(self.history)
        if num_k == 0:
            return

        # Dynamická výška obrázku (5 palců na každý řádek k)
        fig, axes = plt.subplots(num_k, 2, figsize=(14, 5 * num_k))
        
        # Pojistka pro případ, kdy testujeme jen jedno k
        if num_k == 1:
            axes = np.expand_dims(axes, axis=0)

        for idx, data in enumerate(self.history):
            ax1 = axes[idx, 0]
            ax2 = axes[idx, 1]
            
            k = data['k']
            optimal_k = data['optimal_k']
            measured_good_prob = data['measured_good_prob']
            prob_good_exact = data['prob_good_exact']
            shots = data['shots']
            theoretical_angle = data['theoretical_angle']
            counts = data['counts']

            # ==========================================
            # 1. Quiver Plot (State Vector) - Levé pole
            # ==========================================
            ax1.set_xlim(-1.1, 1.1)
            ax1.set_ylim(-1.1, 1.1)
            ax1.set_aspect('equal', adjustable='box') 
            
            ax1.axhline(0, color='black', linewidth=1)
            ax1.axvline(0, color='black', linewidth=1)
            ax1.grid(True, linestyle='--', alpha=0.5)
            circle = plt.Circle((0, 0), 1.0, color='gray', fill=False, linestyle=':')
            ax1.add_patch(circle)
            
            # TEORETICKÝ VEKTOR
            x_theory = np.sin(np.radians(theoretical_angle))
            y_theory = np.cos(np.radians(theoretical_angle))
            
            # REÁLNÝ VEKTOR
            x_real = np.sqrt(measured_good_prob)
            y_real = np.sqrt(1.0 - measured_good_prob)
            
            is_overshoot = k > optimal_k
            real_color = 'crimson' if is_overshoot else '#d62728'
            theory_color = '#1f77b4'
            
            # Vykreslení obou vektorů
            ax1.quiver(0, 0, x_theory, y_theory, angles='xy', scale_units='xy', scale=1, 
                       color=theory_color, width=0.012, alpha=0.4, label="Teorie (Ideál)")
            ax1.quiver(0, 0, x_real, y_real, angles='xy', scale_units='xy', scale=1, 
                       color=real_color, width=0.012, label="Realita (QPU)")
            
            ax1.set_xlabel("Success (Ancilla |0>)", fontweight='bold')
            ax1.set_ylabel("Failure (Ancilla |1>)", fontweight='bold')
            
            title_vec = f"State Vector (k={k})"
            if k == optimal_k:
                title_vec += " - OPTIMAL"
            elif is_overshoot:
                title_vec += " - OVERSHOOT"
            ax1.set_title(title_vec)
            ax1.legend(loc='upper right')
            
            # ==========================================
            # 2. Histogram (Teorie vs Realita) - Pravé pole
            # ==========================================
            theory_counts = {
                '0': int(prob_good_exact * shots),
                '1': int((1.0 - prob_good_exact) * shots)
            }
            
            real_counts = {
                '0': counts.get('0', 0),
                '1': counts.get('1', 0)
            }

            plot_histogram(
                [theory_counts, real_counts], 
                ax=ax2, 
                color=[theory_color, real_color], 
                legend=['Teorie', 'Realita'],
                title=f"Distribuce výsledků (k={k})"
            )
            ax2.set_xlabel("Measured Ancilla State")

        # Finální začištění a uložení super-obrázku
        plt.tight_layout()
        plt.savefig(self.plot_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        print(f"\nVýstupy sloučeny a uloženy:")
        print(f" -> TXT log: {self.txt_path}")
        print(f" -> Grafy (PNG): {self.plot_path}")