from src.oaa.core import get_oaa_parameters, build_oaa_circuit, get_theoretical_probs
from src.oaa.data_handler import DataHandler

from qiskit import transpile
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as IBMSampler
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import SamplerV2 as AerSampler

def run_experiment_noise_degradation(use_simulator=False):
    """
    Experiment 1: Ztráta amplifikace vlivem šumu
    Měří úspěšnost OAA postupně pro k=0 až k=optimal_k+1 na reálném HW.
    """
    # --- NASTAVENÍ EXPERIMENTU ---
    target_prob = 0.05  # Výchozí pravděpodobnost, kterou chceme zesílit (5%)
    shots = 2048
    
    # 1. Výpočet základních parametrů
    theta_rad, theta_deg, optimal_k = get_oaa_parameters(target_prob)
    max_k = optimal_k + 1  # Spustíme i jeden krok navíc pro otestování "overshootu"
    
    # 2. Inicializace backendu
    # Skript se nejprve pokusí připojit k IBM Quantum. Pokud nemáš aktivní relaci,
    # elegantně se přepne na lokální simulátor, aby skript nespadl.
    # --- VÝBĚR BACKENDU PODLE ARGUMENTU ---
    # --- VÝBĚR BACKENDU PODLE ARGUMENTU ---
    if use_simulator:
        print("Spouštím na lokálním simulátoru (AerSimulator)...")
        backend = AerSimulator()
        sampler = AerSampler() # Lokální sampler nevyžaduje předání backendu
    else:
        try:
            print("Připojuji se k IBM Quantum...")
            service = QiskitRuntimeService()
            backend = service.least_busy(operational=True, simulator=False)
            print(f"Úspěšně připojeno. Používám QPU: {backend.name}")
            # V nejnovějších verzích používá IBM SamplerV2 argument 'mode' místo 'backend'
            sampler = IBMSampler(mode=backend) 
        except Exception as e:
            print(f"Nepodařilo se připojit k IBM: {e}")
            print("Padám zpět na lokální simulátor.")
            backend = AerSimulator()
            sampler = AerSampler()
            
    handler = DataHandler(base_dir="results/oaa/exp_a", file_name="exp_a")
    
    print("\n" + "="*50)
    print(f"SPOUŠTÍM EXPERIMENT: Ztráta amplifikace vlivem šumu")
    print(f"Cílová počáteční pravděpodobnost: {target_prob * 100:.2f} %")
    print(f"Optimal k: {optimal_k}")
    print("="*50 + "\n")

    # ... (inicializace backendu a parametrů zůstává stejná) ...

    print("\n1. PŘÍPRAVA A TRANSPILACE OBVODŮ...")
    circuits_to_run = []
    exact_probs_list = []
    theoretical_angles_list = []

    for k in range(max_k + 1):
        # A. Uložení teoretických hodnot
        qc_base = build_oaa_circuit(target_prob, k, measure=False)
        prob_good_exact, _ = get_theoretical_probs(qc_base)
        exact_probs_list.append(prob_good_exact)
        theoretical_angles_list.append((2 * k + 1) * theta_deg)
        
        # B. Tvorba obvodu s měřením
        qc_meas = build_oaa_circuit(target_prob, k, measure=True)
        circuits_to_run.append(qc_meas)

    # Transpilujeme celý list najednou
    transpiled_circuits = transpile(circuits_to_run, backend, optimization_level=3)

    print("\n2. ODESLÁNÍ DO FRONT (Čekáme pouze jednou!)...")
    # V Qiskit SamplerV2 posíláme list tzv. "pubs" (Primitive Unified Blocs)
    pubs = [(qc, None, shots) for qc in transpiled_circuits]
    job = sampler.run(pubs)
    
    print(f"Job odeslán! ID: {job.job_id()}")
    print("Čekám na provedení na QPU...")
    result = job.result() # Zde se skript pozastaví, dokud nedostane všechny výsledky
    print("Výsledky úspěšně staženy!\n")

    print("3. ZPRACOVÁNÍ A ULOŽENÍ VÝSLEDKŮ...")
    backend_name = backend.name if hasattr(backend, 'name') else "AerSimulator"

    for k in range(max_k + 1):
        # Result z V2 sampleru obsahuje výsledky pro každý pub ve stejném pořadí
        pub_result = result[k]
        
        # Extrakce counts
        classical_reg_name = list(pub_result.data.keys())[0]
        counts = getattr(pub_result.data, classical_reg_name).get_counts()
        measured_good_prob = counts.get('0', 0) / shots
        
        # Uložení přes handler
        handler.save_experiment_data(
            k=k,
            target_prob=target_prob,
            optimal_k=optimal_k,
            theoretical_angle=theoretical_angles_list[k],
            prob_good_exact=exact_probs_list[k],
            measured_good_prob=measured_good_prob,
            counts=counts,
            shots=shots,
            backend_name=backend_name
        )
    handler.generate_comparison_plot()
    print("\nExperiment dokončen! Všechna data najdeš ve složce results/oaa.")

if __name__ == "__main__":
    run_experiment_noise_degradation(False)