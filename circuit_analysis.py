from qiskit.visualization import circuit_drawer

if len(results['quantum_history']) > 0:
    circuit = optimizer.circuit.circuit
    print("Quantum Circuit Structure:")
    print(circuit)
    print(f"\nCircuit depth: {circuit.depth()}")
    print(f"Number of gates: {len(circuit.data)}")
    print(f"Parameterized gates: {sum(1 for gate in circuit.data if hasattr(gate[0], 'params'))}")
    
    print(f"\nQuantum State Information (Final Iteration):")
    final_state = results['quantum_history'][-1]['quantum_state']
    print(f"State vector dimension: {len(final_state)}")
    print(f"Number of basis states: {2**optimizer.circuit.num_qubits}")
    print(f"State vector norm: {np.linalg.norm(final_state):.6f}")
    
    top_states = np.argsort(np.abs(final_state)**2)[-5:][::-1]
    print(f"\nTop 5 quantum basis states (by probability):")
    for state_idx in top_states:
        bitstring = format(state_idx, f'0{optimizer.circuit.num_qubits}b')
        prob = np.abs(final_state[state_idx])**2
        config = optimizer.encoder.index_to_config(state_idx)
        print(f"  |{bitstring}⟩: P={prob:.4f}, Config={config}")
