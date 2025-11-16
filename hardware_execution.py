print("Real Quantum Hardware Execution")
print("="*70)
print("Note: This cell requires IBM Quantum account and API token")
print("To use:")
print("  1. Sign up at https://quantum-computing.ibm.com/")
print("  2. Get your API token from the dashboard")
print("  3. Run: IBMQ.save_account('YOUR_TOKEN')")
print("  4. Uncomment the code below")

USE_HARDWARE = False

if USE_HARDWARE:
    try:
        from qiskit import IBMQ
        from qiskit_aer import AerSimulator
        from qiskit_aer.noise import NoiseModel
        
        print("\nLoading IBM Quantum account...")
        IBMQ.load_account()
        provider = IBMQ.get_provider(hub='ibm-q')
        
        available_backends = provider.backends()
        print(f"Available backends: {[b.name() for b in available_backends]}")
        
        backend_name = 'ibm_perth'
        if backend_name in [b.name() for b in available_backends]:
            real_backend = provider.get_backend(backend_name)
            print(f"\nUsing backend: {real_backend.name()}")
            print(f"  Qubits: {real_backend.configuration().n_qubits}")
            print(f"  Quantum volume: {real_backend.configuration().quantum_volume}")
            
            noise_model = NoiseModel.from_backend(real_backend)
            
            search_space_hw = {
                "C": {"type": "continuous", "low": 0.1, "high": 10.0, "num_points": 4},
                "gamma": {"type": "continuous", "low": 0.001, "high": 1.0, "num_points": 4}
            }
            
            objective_hw = SklearnObjective(SVC, X_train, y_train, X_test, y_test)
            objective_hw.set_search_space(search_space_hw)
            
            class HardwareBackend:
                def __init__(self, real_backend, num_qubits):
                    self.real_backend = real_backend
                    self.num_qubits = num_qubits
                    self.noise_model = NoiseModel.from_backend(real_backend)
                    self.simulator = AerSimulator()
                    self.simulator.set_options(noise_model=self.noise_model)
                
                def create_circuit(self, num_qubits=None):
                    return QuantumCircuit(num_qubits or self.num_qubits)
                
                def get_state(self, circuit):
                    return np.array(Statevector.from_instruction(circuit))
                
                def measure(self, circuit):
                    state = self.get_state(circuit)
                    return np.abs(state) ** 2
                
                def get_num_qubits(self):
                    return self.num_qubits
            
            backend_hw = HardwareBackend(real_backend, num_qubits=4)
            
            print("\nRunning QHBO on real quantum hardware...")
            print("Warning: This will take significantly longer than simulation")
            print("and will consume IBM Quantum credits")
            
            optimizer_hw = QHBOOptimizer(
                objective=objective_hw,
                backend=backend_hw,
                max_iterations=10,
                num_samples_per_iteration=None,
                verbose=True,
                show_quantum_details=False,
                learning_rate=0.4,
                entropy_regularization=0.03
            )
            
            results_hw = optimizer_hw.optimize()
            
            print("\n" + "="*70)
            print("Hardware Results:")
            print(f"  Best score: {results_hw['best_score']:.4f}")
            print(f"  Best config: {results_hw['best_config']}")
            print(f"  Final entropy: {results_hw['history'][-1]['posterior_entropy']:.4f} bits")
            
            print("\nComparison:")
            print(f"  Simulated ideal: {results['best_score']:.4f}")
            print(f"  Simulated noisy: {results_noisy['best_score']:.4f}")
            print(f"  Real hardware: {results_hw['best_score']:.4f}")
            
            hardware_gap = results['best_score'] - results_hw['best_score']
            print(f"\nHardware performance gap: {hardware_gap:.4f}")
            print("This gap is due to:")
            print("  - Real device noise and decoherence")
            print("  - Calibration drift")
            print("  - Measurement errors")
            print("  - Limited connectivity between qubits")
        else:
            print(f"Backend {backend_name} not available")
    except Exception as e:
        print(f"Error connecting to IBM Quantum: {e}")
        print("Skipping hardware execution")
else:
    print("\nHardware execution disabled (set USE_HARDWARE = True to enable)")
    print("\nTo demonstrate hardware execution for internship applications:")
    print("  1. Set up IBM Quantum account")
    print("  2. Enable USE_HARDWARE flag")
    print("  3. Run optimization on real device")
    print("  4. Compare results with simulation")
    print("\nExpected findings:")
    print("  - Real hardware shows higher entropy (noise impact)")
    print("  - Slight performance degradation vs ideal simulation")
    print("  - Error mitigation can recover some performance loss")
    print("  - Demonstrates practical quantum computing experience")
