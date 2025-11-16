import time

print("Benchmarking: QHBO vs Classical Methods")
print("="*60)

def random_search(objective, search_space, max_iterations):
    best_score = -np.inf
    best_config = None
    start_time = time.time()
    for _ in range(max_iterations):
        config = {}
        for param_name, param_def in search_space.items():
            if param_def["type"] == "continuous":
                config[param_name] = np.random.uniform(param_def["low"], param_def["high"])
            else:
                config[param_name] = np.random.choice(param_def["values"])
        score = objective.evaluate(config)
        if score > best_score:
            best_score = score
            best_config = config
    return {"best_score": best_score, "best_config": best_config, "time": time.time() - start_time}

backend_fresh = QiskitBackend(num_qubits=6, noise_model=None)
optimizer_fresh = QHBOOptimizer(
    objective=objective,
    backend=backend_fresh,
    max_iterations=25,
    num_samples_per_iteration=None,
    verbose=False,
    show_quantum_details=False,
    learning_rate=0.4,
    entropy_regularization=0.03
)

qhbo_time = time.time()
results_qhbo = optimizer_fresh.optimize()
qhbo_time = time.time() - qhbo_time

num_evaluations = results_qhbo['num_iterations'] * optimizer_fresh.num_samples_per_iteration
random_time = time.time()
results_random = random_search(objective, search_space, num_evaluations)
random_time = time.time() - random_time

print(f"QHBO (Quantum):")
print(f"  Best score: {results_qhbo['best_score']:.4f}")
print(f"  Best config: {results_qhbo['best_config']}")
print(f"  Iterations: {results_qhbo['num_iterations']}")
print(f"  Time: {qhbo_time:.3f}s")
print(f"  Sample efficiency: {num_evaluations} evaluations")
if results_qhbo['history']:
    initial_entropy = results_qhbo['history'][0]['posterior_entropy']
    final_entropy = results_qhbo['history'][-1]['posterior_entropy']
    print(f"  Entropy reduction: {initial_entropy - final_entropy:.4f} bits")

print(f"\nRandom Search (Classical):")
print(f"  Best score: {results_random['best_score']:.4f}")
print(f"  Best config: {results_random['best_config']}")
print(f"  Evaluations: {num_evaluations}")
print(f"  Time: {random_time:.3f}s")

improvement = results_qhbo['best_score'] - results_random['best_score']
if improvement > 0:
    print(f"\nQHBO advantage: +{improvement:.4f} ({improvement/results_random['best_score']*100:.2f}% relative improvement)")
else:
    print(f"\nQHBO vs Random: {improvement:.4f} ({improvement/results_random['best_score']*100:.2f}% relative)")
    print("Note: Quantum method may need more iterations or different hyperparameters to show advantage")
