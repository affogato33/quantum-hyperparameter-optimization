from qiskit_aer.noise import NoiseModel, depolarizing_error, amplitude_damping_error

print("Noise Impact Analysis:")
print("="*60)
print("Running both ideal and noisy simulations from same initial state...")

np.random.seed(42)
initial_params = np.random.uniform(0, 2*np.pi, size=18)

backend_ideal = QiskitBackend(num_qubits=6, noise_model=None)
optimizer_ideal = QHBOOptimizer(
    objective=objective,
    backend=backend_ideal,
    max_iterations=25,
    num_samples_per_iteration=None,
    verbose=False,
    show_quantum_details=False,
    learning_rate=0.4,
    entropy_regularization=0.03,
    random_seed=42
)

optimizer_ideal.circuit.update_params(initial_params)
results_ideal = optimizer_ideal.optimize()

noise_model = NoiseModel()
depol_error = depolarizing_error(0.01, 1)
amp_damp_error = amplitude_damping_error(0.02)
noise_model.add_all_qubit_quantum_error(depol_error, ['ry', 'rz'])
noise_model.add_all_qubit_quantum_error(amp_damp_error, ['ry', 'rz'])

backend_noisy = QiskitBackend(num_qubits=6, noise_model=noise_model)
optimizer_noisy = QHBOOptimizer(
    objective=objective,
    backend=backend_noisy,
    max_iterations=25,
    num_samples_per_iteration=None,
    verbose=False,
    show_quantum_details=False,
    learning_rate=0.4,
    entropy_regularization=0.03,
    random_seed=42
)

optimizer_noisy.circuit.update_params(initial_params)
results_noisy = optimizer_noisy.optimize()

print(f"\nIdeal quantum (no noise):")
print(f"  Best score: {results_ideal['best_score']:.4f}")
print(f"  Iterations: {results_ideal['num_iterations']}")
print(f"  Initial entropy: {results_ideal['history'][0]['posterior_entropy']:.4f} bits")
print(f"  Final entropy: {results_ideal['history'][-1]['posterior_entropy']:.4f} bits")
print(f"  Entropy reduction: {results_ideal['history'][0]['posterior_entropy'] - results_ideal['history'][-1]['posterior_entropy']:.4f} bits")

print(f"\nNoisy quantum (depolarizing p=0.01, amplitude damping γ=0.02):")
print(f"  Best score: {results_noisy['best_score']:.4f}")
print(f"  Iterations: {results_noisy['num_iterations']}")
print(f"  Initial entropy: {results_noisy['history'][0]['posterior_entropy']:.4f} bits")
print(f"  Final entropy: {results_noisy['history'][-1]['posterior_entropy']:.4f} bits")
print(f"  Entropy reduction: {results_noisy['history'][0]['posterior_entropy'] - results_noisy['history'][-1]['posterior_entropy']:.4f} bits")

print(f"\nNoise Impact:")
performance_degradation = results_ideal['best_score'] - results_noisy['best_score']
entropy_increase = results_noisy['history'][-1]['posterior_entropy'] - results_ideal['history'][-1]['posterior_entropy']
print(f"  Performance degradation: {performance_degradation:.4f}")
print(f"  Entropy increase (uncertainty): {entropy_increase:+.4f} bits")
print(f"  Noise prevents full probability concentration on optimal config")
