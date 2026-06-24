"""
Experiment B: Large Sparse Matrix (Scalability Test)
This experiment tests a 16x16 matrix (4 data qubits) that is highly structured.
Because it decomposes into only 4 Pauli terms, it requires only 2 ancilla qubits
and creates a very shallow circuit. 
The goal is to prove that quantum computers can handle large state spaces well, 
as long as the operator is sparse in the Pauli basis.
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
EXPERIMENT_NAME = "exp_c_nice_16x16"
SHOTS = 100000
TARGET_FOLDER = "results/lcu/exp_c"

# Base 2x2 Pauli matrices
I = np.array([[1, 0], [0, 1]], dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)

def tensor_prod(*args):
    """Helper function to calculate the Kronecker product of multiple matrices."""
    res = args[0]
    for mat in args[1:]:
        res = np.kron(res, mat)
    return res

# Construct a large 16x16 matrix using only 4 Pauli terms.
# This represents a simplified physics model (e.g., spin interactions).
A = (
    2.0 * tensor_prod(I, I, I, I) +  # Base energy
    1.5 * tensor_prod(Z, Z, I, I) +  # Interaction between qubits 0 and 1
    1.5 * tensor_prod(I, I, Z, Z) +  # Interaction between qubits 2 and 3
    1.0 * tensor_prod(X, I, I, X)    # Long-range flipping between qubits 0 and 3
)

# Initial data state: Uniform superposition of all 16 states
PSI = np.ones(16, dtype=complex)
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
    
    print(f"      -> Matrix size: {A.shape[0]}x{A.shape[1]}")
    print(f"      -> Target qubits: {num_target}")
    print(f"      -> Ancilla qubits: {num_ancilla}")
    
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
    # Loop over all 16 possible states
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