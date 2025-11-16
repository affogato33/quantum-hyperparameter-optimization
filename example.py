from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import matplotlib.pyplot as plt

X, y = make_classification(n_samples=200, n_features=5, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

search_space = {
    "C": {"type": "continuous", "low": 0.1, "high": 10.0, "num_points": 4},
    "gamma": {"type": "continuous", "low": 0.001, "high": 1.0, "num_points": 4}
}

objective = SklearnObjective(SVC, X_train, y_train, X_test, y_test)
objective.set_search_space(search_space)

backend = QiskitBackend(num_qubits=6, noise_model=None)
optimizer = QHBOOptimizer(
    objective=objective,
    backend=backend,
    max_iterations=25,
    num_samples_per_iteration=None,
    verbose=True,
    show_quantum_details=True,
    learning_rate=0.4,
    entropy_regularization=0.03
)

print(f"Auto-scaled samples per iteration: {optimizer.num_samples_per_iteration}")
print(f"Total search space size: {optimizer.encoder.get_num_configs()} configurations\n")

results = optimizer.optimize()

print(f"\nOptimization Results:")
print(f"Best score: {results['best_score']:.4f}")
print(f"Best config: {results['best_config']}")
print(f"Total iterations: {results['num_iterations']}")
if results['history']:
    initial_entropy = results['history'][0]['posterior_entropy']
    final_entropy = results['history'][-1]['posterior_entropy']
    print(f"Initial entropy: {initial_entropy:.4f} bits")
    print(f"Final entropy: {final_entropy:.4f} bits")
    print(f"Entropy reduction: {initial_entropy - final_entropy:.4f} bits")
