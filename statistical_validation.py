from scipy.stats import ttest_ind, mannwhitneyu

print("Statistical Significance Analysis")
print("="*70)
print("Running 20 independent trials of each method...")
print("(This may take several minutes)")

num_trials = 20
search_space_stats = {
    "C": {"type": "continuous", "low": 0.1, "high": 10.0, "num_points": 4},
    "gamma": {"type": "continuous", "low": 0.001, "high": 1.0, "num_points": 4}
}

objective_stats = SklearnObjective(SVC, X_train, y_train, X_test, y_test)
objective_stats.set_search_space(search_space_stats)

qhbo_scores = []
random_scores = []

for trial in range(num_trials):
    if (trial + 1) % 5 == 0:
        print(f"  Completed {trial + 1}/{num_trials} trials...")
    
    backend_trial = QiskitBackend(num_qubits=4, noise_model=None)
    optimizer_trial = QHBOOptimizer(
        objective=objective_stats,
        backend=backend_trial,
        max_iterations=20,
        num_samples_per_iteration=None,
        verbose=False,
        show_quantum_details=False,
        learning_rate=0.4,
        entropy_regularization=0.03,
        random_seed=42 + trial
    )
    
    results_trial = optimizer_trial.optimize()
    qhbo_scores.append(results_trial['best_score'])
    
    def evaluate_config_local(config_dict):
        try:
            model = SVC(**config_dict)
            model.fit(X_train, y_train)
            return -model.score(X_test, y_test)
        except:
            return 1.0
    
    def random_search_local(search_space, n_evaluations):
        best_score = -np.inf
        for _ in range(n_evaluations):
            config = {}
            for param_name, param_def in search_space.items():
                if param_def["type"] == "continuous":
                    config[param_name] = np.random.uniform(param_def["low"], param_def["high"])
                else:
                    config[param_name] = np.random.choice(param_def["values"])
            score = -evaluate_config_local(config)
            if score > best_score:
                best_score = score
        return best_score
    
    np.random.seed(42 + trial)
    num_evals = results_trial['num_iterations'] * optimizer_trial.num_samples_per_iteration
    random_score = random_search_local(search_space_stats, num_evals)
    random_scores.append(random_score)

qhbo_mean = np.mean(qhbo_scores)
qhbo_std = np.std(qhbo_scores)
qhbo_sem = qhbo_std / np.sqrt(num_trials)

random_mean = np.mean(random_scores)
random_std = np.std(random_scores)
random_sem = random_std / np.sqrt(num_trials)

print("\n" + "="*70)
print("Statistical Results (20 trials):")
print(f"{'Method':<20} {'Mean Score':<15} {'Std Dev':<12} {'95% CI':<20}")
print("-"*70)

qhbo_ci_lower = qhbo_mean - 1.96 * qhbo_sem
qhbo_ci_upper = qhbo_mean + 1.96 * qhbo_sem
print(f"{'QHBO':<20} {qhbo_mean:<15.4f} {qhbo_std:<12.4f} [{qhbo_ci_lower:.4f}, {qhbo_ci_upper:.4f}]")

random_ci_lower = random_mean - 1.96 * random_sem
random_ci_upper = random_mean + 1.96 * random_sem
print(f"{'Random Search':<20} {random_mean:<15.4f} {random_std:<12.4f} [{random_ci_lower:.4f}, {random_ci_upper:.4f}]")

t_stat, p_value_ttest = ttest_ind(qhbo_scores, random_scores)
u_stat, p_value_mw = mannwhitneyu(qhbo_scores, random_scores, alternative='two-sided')

print("\n" + "="*70)
print("Statistical Tests:")
print(f"  t-test (independent samples):")
print(f"    t-statistic: {t_stat:.4f}")
print(f"    p-value: {p_value_ttest:.4f}")
print(f"    Significant at α=0.05: {'Yes' if p_value_ttest < 0.05 else 'No'}")

print(f"\n  Mann-Whitney U test (non-parametric):")
print(f"    U-statistic: {u_stat:.4f}")
print(f"    p-value: {p_value_mw:.4f}")
print(f"    Significant at α=0.05: {'Yes' if p_value_mw < 0.05 else 'No'}")

effect_size = (qhbo_mean - random_mean) / np.sqrt((qhbo_std**2 + random_std**2) / 2)
print(f"\n  Effect size (Cohen's d): {effect_size:.4f}")
if abs(effect_size) < 0.2:
    effect_interpretation = "negligible"
elif abs(effect_size) < 0.5:
    effect_interpretation = "small"
elif abs(effect_size) < 0.8:
    effect_interpretation = "medium"
else:
    effect_interpretation = "large"
print(f"    Interpretation: {effect_interpretation} effect")

print("\n" + "="*70)
print("Conclusion:")
if p_value_ttest < 0.05:
    if qhbo_mean > random_mean:
        print("QHBO significantly outperforms Random Search (p < 0.05)")
    else:
        print("Random Search significantly outperforms QHBO (p < 0.05)")
else:
    print("No statistically significant difference between QHBO and Random Search")
    print("This suggests quantum advantage may require:")
    print("  - Larger problem sizes (more configurations)")
    print("  - Different problem structures (correlated hyperparameters)")
    print("  - More iterations or different hyperparameters")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].boxplot([qhbo_scores, random_scores], labels=['QHBO', 'Random Search'])
axes[0].set_ylabel('Best Score', fontsize=11)
axes[0].set_title('Score Distribution (20 trials)', fontsize=12, fontweight='bold')
axes[0].grid(True, alpha=0.3, axis='y')

axes[1].hist(qhbo_scores, bins=10, alpha=0.6, label='QHBO', color='#2E86AB', edgecolor='black')
axes[1].hist(random_scores, bins=10, alpha=0.6, label='Random Search', color='#A23B72', edgecolor='black')
axes[1].axvline(qhbo_mean, color='#2E86AB', linestyle='--', linewidth=2, label=f'QHBO mean: {qhbo_mean:.4f}')
axes[1].axvline(random_mean, color='#A23B72', linestyle='--', linewidth=2, label=f'Random mean: {random_mean:.4f}')
axes[1].set_xlabel('Best Score', fontsize=11)
axes[1].set_ylabel('Frequency', fontsize=11)
axes[1].set_title('Score Histogram (20 trials)', fontsize=12, fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()
