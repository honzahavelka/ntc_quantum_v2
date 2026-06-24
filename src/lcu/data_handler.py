"""
Data handling module for LCU experiments.
Responsible for saving raw experimental data, formatted text reports, 
and generating dual-plot comparison charts into a unified experiment folder.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def save_experiment_results(experiment_name, results_list, folder_path):
    """
    Saves experimental data to both a machine-readable CSV and a human-readable TXT report.
    """
    os.makedirs(folder_path, exist_ok=True)
    
    safe_name = experiment_name.replace(" ", "_").replace("/", "_")
    csv_filepath = os.path.join(folder_path, f"{safe_name}.csv")
    txt_filepath = os.path.join(folder_path, f"{safe_name}.txt")
    
    # 1. Save machine-readable CSV
    df = pd.DataFrame(results_list)
    df.to_csv(csv_filepath, index=False)
    
    # 2. Save human-readable TXT report
    total_shots = df['Total_Shots'].iloc[0]
    total_successes = df['Total_Successes'].iloc[0]
    backend_name = df.get('Backend_Used', pd.Series(['Simulator'])).iloc[0]
    success_rate = (total_successes / total_shots * 100) if total_shots > 0 else 0
    
    with open(txt_filepath, 'w') as f:
        f.write(f"=========================================================\n")
        f.write(f" EXPERIMENT REPORT: {experiment_name}\n")
        f.write(f"=========================================================\n")
        f.write(f"Backend Used       : {backend_name}\n")
        f.write(f"Total Shots Sent   : {total_shots}\n")
        f.write(f"Successful Shots   : {total_successes} (Ancillas measured |0...0>)\n")
        f.write(f"Success Rate       : {success_rate:.2f} %\n\n")
        
        f.write(f"--- Probability Distribution Comparison ---\n")
        # Přidán sloupec Diff (%)
        f.write(f"{'State':<10} | {'Measured Prob':<15} | {'Analytical Prob':<15} | {'Diff (%)':<10} | {'Raw Counts'}\n")
        f.write("-" * 78 + "\n")
        
        for _, row in df.iterrows():
            state = f"|{row['Target_State']}⟩"
            meas_prob = row.get('Hardware_Prob', row.get('Simulator_Prob', 0.0))
            analyt_prob = row['Analytical_Prob']
            raw_counts = row.get('Measured_Counts', 0)
            
            # Výpočet absolutní odchylky v procentních bodech
            diff_pct = abs(meas_prob - analyt_prob) * 100
            
            f.write(f"{state:<10} | {meas_prob:<15.4f} | {analyt_prob:<15.4f} | {diff_pct:<10.2f} | {raw_counts}\n")
            
    print(f"[DataHandler] Report successfully saved to: {txt_filepath}")
    return df

def plot_experiment_results(df, experiment_name, folder_path):
    """
    Generates a figure with two subplots: probabilities and raw counts.
    """
    os.makedirs(folder_path, exist_ok=True)
    
    safe_name = experiment_name.replace(" ", "_").replace("/", "_")
    filepath = os.path.join(folder_path, f"{safe_name}.png")
    
    # Extract data
    states = df['Target_State'].tolist()
    labels = [f"|{s}⟩" for s in states]
    analytical_probs = df['Analytical_Prob'].tolist()
    raw_counts = df.get('Measured_Counts', [0]*len(states)).tolist()
    
    if 'Hardware_Prob' in df.columns:
        measured_probs = df['Hardware_Prob'].tolist()
        measured_label = "Measured (Hardware)"
        measured_color = '#e74c3c'
    else:
        measured_probs = df.get('Simulator_Prob', [0]*len(states)).tolist()
        measured_label = "Measured (Simulator)"
        measured_color = '#2ecc71'

    x = np.arange(len(states))
    width = 0.35
    
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(10, 10))
    
    # Subplot 1: Probabilities
    ax1.bar(x - width/2, analytical_probs, width, label='Analytical (Theory)', color='#3498db')
    ax1.bar(x + width/2, measured_probs, width, label=measured_label, color=measured_color)
    ax1.set_ylabel('Probability')
    ax1.set_title(f'Probability Distribution: {experiment_name}')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=45 if len(states) > 8 else 0)
    ax1.legend()
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Subplot 2: Raw Counts
    ax2.bar(x, raw_counts, width=0.6, color='#9b59b6', label='Raw Success Counts')
    ax2.set_ylabel('Number of Measurements (Shots)')
    ax2.set_xlabel('Quantum Target State |ψ⟩')
    ax2.set_title(f'Raw Count Distribution (Total Successes: {sum(raw_counts)})')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=45 if len(states) > 8 else 0)
    ax2.legend()
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    
    fig.tight_layout()
    plt.savefig(filepath, dpi=300)
    plt.close(fig)
    
    print(f"[DataHandler] Plot successfully saved to: {filepath}")

def process_and_save(experiment_name, results_list, folder_path):
    """
    Wrapper function to execute both saving and plotting in the specified folder.
    """
    df = save_experiment_results(experiment_name, results_list, folder_path)
    plot_experiment_results(df, experiment_name, folder_path)