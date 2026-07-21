"""
Core module for the Linear Combination of Unitaries (LCU) algorithm.
Contains functions to validate inputs and generate dynamic LCU quantum circuits
for arbitrary 2^n x 2^n matrices.
"""

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import StatePreparation, PauliGate
from qiskit.quantum_info import SparsePauliOp

def validate_lcu_inputs(A, psi):
    """
    Validates the target matrix A and initial state vector PSI for the LCU algorithm.
    
    Parameters:
        A (numpy.ndarray): The target non-unitary matrix.
        psi (numpy.ndarray): The initial quantum state vector.
        
    Raises:
        ValueError: If the matrix is not square, not a power of 2, or if the vector
                    does not match the matrix dimensions.
    """
    # 1. Check if the matrix is square
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"ERROR: Matrix A must be square (2D array). Given shape: {A.shape}.")
    
    dim_A = A.shape[0]
    
    # 2. Check if the matrix dimension is a power of 2 (2^n)
    if not np.log2(dim_A).is_integer():
        raise ValueError(f"ERROR: Dimension of matrix A must be a power of 2 (e.g., 2x2, 4x4, 8x8). Given dimension is {dim_A}x{dim_A}.")
    
    # 3. Check if the size of vector PSI matches matrix A
    if psi.ndim != 1:
        raise ValueError(f"ERROR: State PSI must be a 1D vector. Given shape is {psi.shape}.")
        
    if psi.shape[0] != dim_A:
        raise ValueError(f"ERROR: Length of vector PSI ({psi.shape[0]}) does not match the dimension of matrix A ({dim_A}).")

    return True

def generate_lcu_circuit(A, psi):
    """
    Generates a complete LCU quantum circuit for a given matrix A and initial state psi.
    
    Parameters:
        A (numpy.ndarray): The target non-unitary matrix.
        psi (numpy.ndarray): The initial quantum state vector.
        
    Returns:
        tuple: (qc, num_target, num_ancilla)
            - qc (QuantumCircuit): The constructed LCU circuit ready for simulation/hardware.
            - num_target (int): Number of target (data) qubits.
            - num_ancilla (int): Number of ancilla qubits used.
    """
    # 1. Validate inputs
    validate_lcu_inputs(A, psi)
    
    # 2. Normalize the initial state vector
    psi_norm = psi / np.linalg.norm(psi)
    
    # Calculate the required number of target (data) qubits
    num_target = int(np.log2(A.shape[0]))
    
    # 3. Perform automatic Pauli decomposition of matrix A
    pauli_op = SparsePauliOp.from_operator(A)
    paulis = pauli_op.paulis
    coeffs = np.array(pauli_op.coeffs)
    num_terms = len(coeffs)
    
    # 4. Determine ancilla requirements and pad coefficients
    # We need enough qubits (k) to represent all non-zero terms, so 2^k >= num_terms
    num_ancilla = int(np.ceil(np.log2(num_terms))) if num_terms > 1 else 1
    
    # Qiskit's StatePreparation strictly requires the input array length to be exactly 2^k
    padded_coeffs = np.zeros(2**num_ancilla, dtype=complex)
    padded_coeffs[:num_terms] = coeffs
    
    # Calculate the LCU normalization factor lambda and the preparation amplitudes
    lambda_ = np.abs(padded_coeffs).sum()
    amplitudes = np.sqrt(padded_coeffs / lambda_)
    
    # 5. Initialize StatePreparation operators
    prep = StatePreparation(amplitudes, label='PREP')
    # Use the conjugate to avoid negating the phase during post-selection
    prep_transpose = StatePreparation(amplitudes.conjugate(), inverse=True, label='PREP_transpose')
    psi_prep = StatePreparation(psi_norm, label='|ψ⟩')
    
    # 6. Construct the Quantum Circuit
    anc = QuantumRegister(num_ancilla, name='anc')
    tgt = QuantumRegister(num_target, name='tgt')
    cr_anc = ClassicalRegister(num_ancilla, name='meas_anc')
    cr_tgt = ClassicalRegister(num_target, name='meas_tgt')
    
    qc = QuantumCircuit(anc, tgt, cr_anc, cr_tgt)
    
    # STEP 0: Prepare the initial data state |ψ⟩
    qc.append(psi_prep, tgt)
    qc.barrier(label="|Φ_0⟩")
    
    # STEP 1: Apply PREP to ancillas
    qc.append(prep, anc)
    qc.barrier(label="|Φ_1⟩")
    
    # STEP 2: Apply the Select (SEL) operation via controlled Paulis
    for idx, pauli_obj in enumerate(paulis):
        # Convert index to zero-padded binary string for the control state
        ctrl_state = format(idx, f'0{num_ancilla}b')
        pauli_str = pauli_obj.to_label()
        
        # Attach multi-qubit control
        pauli_gate = PauliGate(pauli_str)
        controlled_pauli = pauli_gate.control(num_ancilla, ctrl_state=ctrl_state)
        
        qc.append(controlled_pauli, anc[:] + tgt[:])
        
    qc.barrier(label="|Φ_2⟩")
    
    # STEP 3: Decouple registers using Transpose and measure
    qc.append(prep_transpose, anc)
    qc.measure(anc, cr_anc)
    qc.barrier(label="|Φ_3⟩")
    
    # Measure the target data qubits
    qc.measure(tgt, cr_tgt)
    
    return qc, num_target, num_ancilla