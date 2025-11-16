from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
import time

print("Scalability Analysis: QHBO Performance vs Problem Size")
print("="*70)

wine_data = load_wine()
X_wine, y_wine = wine_data.data, wine_data.target
X_train_w, X_test_w, y_train_w, y_test_w = train_test_split(X_wine, y_wine, test_size=0.2, random_state=42)

problem_sizes = [
    {"name": "Small (16 configs)", "C_points": 4, "gamma_points": 4, "max_iter": 20},
    {"name": "Medium (64 configs)", "C_points": 8, "gamma_points": 8, "max_iter": 30},
    {"name": "Large (256 configs)", "C_points": 16, "gamma_points": 16, "max_iter": 40},
]

scalability_results = []

for prob_size in problem_sizes:
    print(f"\nTesting {prob_size['name']}...")
    
    search_space_scaled = {
        "C": {"type": "continuous", "low": 0.1, "high": 10.0, "num_points": prob_size["C_points"]},
        "gamma": {"type": "continuous", "low": 0.001, "high": 1.0, "num_points": prob_size["gamma_points"]}
    }
    
    num_configs = prob_size["C_points"] * prob_size["gamma_points"]
    
    objective_scaled = SklearnObjective(SVC, X_train_w, y_train_w, X_test_w, y_test_w)
    objective_scaled.set_search_space(search_space_scaled)
    
    num_qubits_needed = int(np.ceil(np.log2(prob_size["C_points"]))) + int(np.ceil(np.log2(prob_size["gamma_points"])))
    
    backend_scaled = QiskitBackend(num_qubits=num_qubits_needed, noise_model=None)
    
    start_time = time.time()
    optimizer_scaled = QHBOOptimizer(
        objective=objective_scaled,
        backend=backend_scaled,
        max_iterations=prob_size["max_iter"],
        num_samples_per_iteration=None,
        verbose=False,
        show_quantum_details=False,
        learning_rate=0.4,
        entropy_regularization=0.03
    )
    
    results_scaled = optimizer_scaled.optimize()
    elapsed_time = time.time() - start_time
    
    num_evaluations = results_scaled['num_iterations'] * optimizer_scaled.num_samples_per_iteration
    
    scalability_results.append({
        "size": prob_size["name"],
        "num_configs": num_configs,
        "best_score": results_scaled['best_score'],
        "iterations": results_scaled['num_iterations'],
        "evaluations": num_evaluations,
        "time": elapsed_time,
        "samples_per_iter": optimizer_scaled.num_samples_per_iteration
    })
    
    print(f"  Configurations: {num_configs}")
    print(f"  Best score: {results_scaled['best_score']:.4f}")
    print(f"  Evaluations: {num_evaluations}")
    print(f"  Time: {elapsed_time:.2f}s")

print("\n" + "="*70)
print("Scalability Summary:")
print(f"{'Size':<20} {'Configs':<10} {'Score':<10} {'Evals':<10} {'Time (s)':<12} {'Eval/Config':<12}")
print("-"*70)
for res in scalability_results:
    eval_ratio = res['evaluations'] / res['num_configs']
    print(f"{res['size']:<20} {res['num_configs']:<10} {res['best_score']:<10.4f} {res['evaluations']:<10} {res['time']:<12.2f} {eval_ratio:<12.2f}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

config_counts = [r['num_configs'] for r in scalability_results]
evaluations = [r['evaluations'] for r in scalability_results]
times = [r['time'] for r in scalability_results]

axes[0].plot(config_counts, evaluations, 'o-', linewidth=2, markersize=8, color='#2E86AB')
axes[0].set_xlabel('Search Space Size (configurations)', fontsize=11)
axes[0].set_ylabel('Evaluations to Convergence', fontsize=11)
axes[0].set_title('Sample Efficiency vs Problem Size', fontsize=12, fontweight='bold')
axes[0].grid(True, alpha=0.3)
axes[0].set_xscale('log', base=2)

axes[1].plot(config_counts, times, 's-', linewidth=2, markersize=8, color='#A23B72')
axes[1].set_xlabel('Search Space Size (configurations)', fontsize=11)
axes[1].set_ylabel('Runtime (seconds)', fontsize=11)
axes[1].set_title('Computational Cost vs Problem Size', fontsize=12, fontweight='bold')
axes[1].grid(True, alpha=0.3)
axes[1].set_xscale('log', base=2)

plt.tight_layout()
plt.show()
