"""
Experiment D: Diagonal and Corners Matrix (Boundary Interaction Test)
This experiment tests a 4x4 matrix with entries only on the main diagonal 
and in the far corners. This structure represents periodic boundary conditions 
often found in physical spin chains. Because it decomposes into only 6 Pauli terms, 
the circuit depth is moderate, offering a good balance between complexity and NISQ performance.
"""

import numpy as np

# Import our custom modules from the src folder
from src.lcu.core import generate_lcu_circuit
from src.lcu.data_handler import process_and_save

from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

# ==========================================
# 1. EXPERIMENT CONFIGURATION
# ==========================================
EXPERIMENT_NAME = "exp_d_diag_corners_4x4"
SHOTS = 100000
TARGET_FOLDER = "results/lcu/exp_d"

# Define a 4x4 matrix with custom diagonal elements and complex values in the corners
A = np.array([
    [ 2.0,  0.0,  0.0,  1.0j],
    [ 0.0,  1.5,  0.0,  0.0 ],
    [ 0.0,  0.0,  1.0,  0.0 ],
    [ 1.0j, 0.0,  0.0,  2.5 ]
], dtype=complex)

# Initial data state: Superposition with a phase shift
PSI = np.array([1.0, 1.0j, -1.0, -1.0j], dtype=complex)
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
    
    print(f"      -> Target qubits: {num_target}")
    print(f"      -> Ancilla qubits: {num_ancilla} (required for Pauli decomposition)")
    
    # Calculate classical analytical probabilities
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
    print("      -> Waiting for execution in queue...")
    
    result = job.result()[0]

    # 5. Process Results
    print("[5/5] Job complete! Processing and saving results...")
    anc_shots = result.data.meas_anc.get_bitstrings()
    tgt_shots = result.data.meas_tgt.get_bitstrings()

    target_ancilla_state = '0' * num_ancilla
    filtered_counts = {}
    total_successes = 0

    # Filter for successful post-selection runs
    for anc_val, tgt_val in zip(anc_shots, tgt_shots):
        if anc_val == target_ancilla_state:
            total_successes += 1
            filtered_counts[tgt_val] = filtered_counts.get(tgt_val, 0) + 1

    # Format data for the updated data_handler (including Measured_Counts)
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

    # Save to the specific folder structure requested
    process_and_save(EXPERIMENT_NAME, results_to_save, folder_path=TARGET_FOLDER)
    
    print(f"\n{'='*55}")
    print(f" EXPERIMENT FINISHED SUCCESSFULLY")
    print(f" Post-selection success rate: {(total_successes/SHOTS)*100:.2f} %")
    print(f"{'='*55}\n")

if __name__ == "__main__":
    main()