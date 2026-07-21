import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

def get_oaa_parameters(target_prob):
    """Vypočítá úhly a optimální počet iterací pro zadanou pravděpodobnost."""
    theta_rad = np.arcsin(np.sqrt(target_prob))
    theta_deg = np.degrees(theta_rad)
    optimal_k = int(round((np.pi / (4 * theta_rad)) - 0.5))
    return theta_rad, theta_deg, optimal_k

def build_W(theta_rad):
    """Příprava počátečního stavu (orákulum)."""
    qc = QuantumCircuit(2, name=" W ")
    qc.x(0)
    qc.ry(2 * theta_rad, 0)
    qc.cx(0, 1)
    return qc.to_instruction()

def build_W_dg(theta_rad):
    """Inverzní příprava počátečního stavu."""
    qc = QuantumCircuit(2, name=" W^dagger ")
    qc.cx(0, 1)
    qc.ry(-2 * theta_rad, 0)
    qc.x(0)
    return qc.to_instruction()

def build_Q(theta_rad):
    """Groverův/OAA operátor Q."""
    qc = QuantumCircuit(2, name=" Q ")
    qc.z(0)                  
    qc.append(build_W_dg(theta_rad), [0, 1])
    qc.z(0)                  
    qc.append(build_W(theta_rad), [0, 1])
    return qc.to_instruction()

def build_oaa_circuit(target_prob, k, measure=True):
    """
    Sestaví kompletní OAA obvod pro danou pravděpodobnost a počet iterací k.
    Vrací obvod. Pokud measure=True, přidá měření na ancilla qubit.
    """
    theta_rad, _, _ = get_oaa_parameters(target_prob)
    
    # 2 qubity, 1 klasický bit pro měření (pokud měříme)
    qc = QuantumCircuit(2, 1) if measure else QuantumCircuit(2)
    
    qc.append(build_W(theta_rad), [0, 1])
    
    for _ in range(k):
        qc.append(build_Q(theta_rad), [0, 1])
        
    if measure:
        qc.measure(0, 0)
        
    return qc

def get_theoretical_probs(qc_base):
    """Pomocná funkce pro získání přesných pravděpodobností pomocí Statevectoru."""
    sv = Statevector(qc_base)
    probs = sv.probabilities([0]) # Marginalizace pro izolaci ancilla qubitu
    return probs[0], probs[1]     # Vrací (prob_success, prob_failure)