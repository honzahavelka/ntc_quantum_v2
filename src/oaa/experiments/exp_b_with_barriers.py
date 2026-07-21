import argparse
from qiskit import QuantumCircuit, transpile
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as IBMSampler
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import SamplerV2 as AerSampler

# Importujeme základní stavební bloky a pomocné funkce z tvého core
from src.oaa.core import get_oaa_parameters, build_W, build_Q, get_theoretical_probs
from src.oaa.data_handler import DataHandler

def build_oaa_circuit_with_barriers(target_prob, k, measure=True):
    """
    Sestaví OAA obvod, ale na rozdíl od core.py vkládá bariéry.
    Tím se zabrání transpileru extrémně optimalizovat obvod a zachová se jeho
    skutečná fyzická hloubka, což nám umožní reálně měřit vliv šumu.
    """
    theta_rad, _, _ = get_oaa_parameters(target_prob)
    
    qc = QuantumCircuit(2, 1) if measure else QuantumCircuit(2)
    
    # 1. Počáteční stav
    qc.append(build_W(theta_rad), [0, 1])
    qc.barrier()  # <--- BARIÉRA: Zabraňuje sloučení W s Q
    
    # 2. Aplikace OAA iterací
    for _ in range(k):
        qc.append(build_Q(theta_rad), [0, 1])
        qc.barrier()  # <--- BARIÉRA: Zabraňuje sloučení jednotlivých iterací Q do sebe
        
    if measure:
        qc.measure(0, 0)
        
    return qc

def run_experiment_noise_with_barriers(use_simulator=False):
    """
    Experiment 1b: Ztráta amplifikace vlivem šumu (s vynucenou hloubkou obvodu)
    Měří úspěšnost OAA pro k=0 až k=optimal_k+1.
    """
    # --- NASTAVENÍ EXPERIMENTU ---
    target_prob = 0.05  # Výchozí pravděpodobnost, kterou chceme zesílit (5%)
    shots = 2048
    
    theta_rad, theta_deg, optimal_k = get_oaa_parameters(target_prob)
    max_k = optimal_k + 1
    
    # --- VÝBĚR BACKENDU ---
    if use_simulator:
        print("Spouštím na lokálním simulátoru (AerSimulator)...")
        backend = AerSimulator()
        sampler = AerSampler()
    else:
        try:
            print("Připojuji se k IBM Quantum...")
            service = QiskitRuntimeService()
            backend = service.least_busy(operational=True, simulator=False)
            print(f"Úspěšně připojeno. Používám QPU: {backend.name}")
            sampler = IBMSampler(mode=backend) 
        except Exception as e:
            print(f"Nepodařilo se připojit k IBM: {e}")
            print("Padám zpět na lokální simulátor.")
            backend = AerSimulator()
            sampler = AerSampler()
            
    # Úprava složky pro ukládání dat tohoto experimentu
    handler = DataHandler(base_dir="results/oaa/exp_b", file_name="exp_b_with_barrieris")
    
    print("\n" + "="*50)
    print(f"SPOUŠTÍM EXPERIMENT 1B: Šum a dekoherence (Bariéry aktivní)")
    print(f"Cílová počáteční pravděpodobnost: {target_prob * 100:.2f} %")
    print(f"Optimal k: {optimal_k}")
    print("="*50 + "\n")

    print("1. PŘÍPRAVA A TRANSPILACE OBVODŮ...")
    circuits_to_run = []
    exact_probs_list = []
    theoretical_angles_list = []

    for k in range(max_k + 1):
        # A. Uložení teoretických hodnot (používáme náš nový builder)
        qc_base = build_oaa_circuit_with_barriers(target_prob, k, measure=False)
        prob_good_exact, _ = get_theoretical_probs(qc_base)
        exact_probs_list.append(prob_good_exact)
        theoretical_angles_list.append((2 * k + 1) * theta_deg)
        
        # B. Tvorba obvodu s měřením
        qc_meas = build_oaa_circuit_with_barriers(target_prob, k, measure=True)
        circuits_to_run.append(qc_meas)

    # I s optimization_level=3 se díky bariérám hloubka obvodu zachová
    transpiled_circuits = transpile(circuits_to_run, backend, optimization_level=3)

    print("\n2. ODESLÁNÍ DO FRONT (Batching)...")
    pubs = [(qc, None, shots) for qc in transpiled_circuits]
    job = sampler.run(pubs)
    
    if hasattr(job, 'job_id'):
        print(f"Job odeslán! ID: {job.job_id()}")
    print("Čekám na provedení...")
    result = job.result()
    print("Výsledky úspěšně staženy!\n")

    print("3. ZPRACOVÁNÍ A ULOŽENÍ VÝSLEDKŮ...")
    backend_name = backend.name if hasattr(backend, 'name') else "AerSimulator"

    for k in range(max_k + 1):
        pub_result = result[k]
        classical_reg_name = list(pub_result.data.keys())[0]
        counts = getattr(pub_result.data, classical_reg_name).get_counts()
        measured_good_prob = counts.get('0', 0) / shots
        
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
    print("\nExperiment B dokončen! Všechna data najdeš ve složce results/oaa/exp_b.")

if __name__ == "__main__":
    run_experiment_noise_with_barriers(True)