"""
Experiment B: The Stress Test (Dense Complex Matrix on NISQ Hardware)
This experiment tests a fully dense 4x4 complex matrix that decomposes into 
all 16 possible Pauli terms. This requires 4 ancilla qubits and creates a very 
deep circuit with 16 multi-controlled Pauli operations. 
The goal is to observe how physical hardware noise degrades the theoretical 
probability distribution compared to an ideal simulator.
"""

import numpy as np

# Import our custom modules
from src.lcu.core import generate_lcu_circuit
from src.lcu.data_handler import process_and_save

from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

# ==========================================
# 1. EXPERIMENT CONFIGURATION
# ==========================================
EXPERIMENT_NAME = "exp_b_hard_4x4"
SHOTS = 100000
TARGET_FOLDER = "results/lcu/exp_b"

# Define a completely dense, asymmetric, complex 4x4 matrix.
# This will force Qiskit to use all 16 Pauli basis matrices.
A = np.array([
    [ 0.12-0.5j, -3.4,       1.1j,      0.8  ],
    [ 2.2,        0.0+0.8j, -0.4,       1.55j],
    [-0.9j,       1.1,      -2.2-0.1j,  0.7  ],
    [ 5.0,       -1.3j,      0.9,       3.14 ]
], dtype=complex)

# Initial data state: Random complex amplitudes
PSI = np.array([-0.5, 1.2j, 3.3-1j, -0.1], dtype=complex)
PSI = PSI / np.linalg.norm(PSI)

# ==========================================
# 2. MAIN EXECUTION
# ==========================================
def main():
    print(f"\n{'='*55}")
    print(f" STARTING EXPERIMENT: {EXPERIMENT_NAME}")
    print(f"{'='*55}\n")

    # 1. Generate Logical Circuit
    print("[1/5] Generating logical LCU circuit and analytical theory...")
    qc, num_target, num_ancilla = generate_lcu_circuit(A, PSI)
    
    analytical_result = A @ PSI
    analytical_result_norm = analytical_result / np.linalg.norm(analytical_result)
    analytical_probs = np.abs(analytical_result_norm) ** 2

    # 2. Connect to IBM Quantum
    print("[2/5] Connecting to IBM Quantum...")
    service = QiskitRuntimeService(channel="ibm_quantum_platform")
    backend = service.least_busy(simulator=False, operational=True)
    print(f"      -> Selected backend: {backend.name}")

    # 3. Transpile for Hardware
    print("[3/5] Transpiling circuit for physical hardware...")
    pass_manager = generate_preset_pass_manager(optimization_level=3, target=backend.target)
    isa_qc = pass_manager.run(qc)

    # 4. Execute on Hardware
    print(f"[4/5] Submitting job ({SHOTS} shots)...")
    sampler = Sampler(mode=backend)
    sampler.options.default_shots = SHOTS
    
    job = sampler.run([isa_qc])
    print(f"      -> Job submitted! ID: {job.job_id()}")
    print("      -> Waiting for execution in queue (this may take a while)...")
    
    result = job.result()[0]

    # 5. Process Results
    print("[5/5] Job complete! Processing and saving results...")
    anc_shots = result.data.meas_anc.get_bitstrings()
    tgt_shots = result.data.meas_tgt.get_bitstrings()

    target_ancilla_state = '0' * num_ancilla
    filtered_counts = {}
    total_successes = 0

    for anc_val, tgt_val in zip(anc_shots, tgt_shots):
        if anc_val == target_ancilla_state:
            total_successes += 1
            filtered_counts[tgt_val] = filtered_counts.get(tgt_val, 0) + 1

    results_to_save = []
    for i in range(2**num_target):
        state_bin = format(i, f'0{num_target}b')
        measured_count = filtered_counts.get(state_bin, 0)
        hw_prob = measured_count / total_successes if total_successes > 0 else 0
        
        results_to_save.append({
            "Experiment": EXPERIMENT_NAME,
            "Target_State": state_bin,
            "Analytical_Prob": analytical_probs[i],
            "Hardware_Prob": hw_prob,
            "Measured_Counts": measured_count,
            "Total_Successes": total_successes,
            "Total_Shots": len(anc_shots),
            "Backend_Used": backend.name
        })

    # Save data to unified folder
    process_and_save(EXPERIMENT_NAME, results_to_save, folder_path=TARGET_FOLDER)
    
    print(f"\n{'='*55}")
    print(f" EXPERIMENT FINISHED SUCCESSFULLY")
    print(f" Post-selection success rate: {(total_successes/SHOTS)*100:.2f} %")
    print(f"{'='*55}\n")

if __name__ == "__main__":
    main()