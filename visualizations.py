fig, axes = plt.subplots(2, 2, figsize=(14, 10))

scores = [h['best_score'] for h in results['history']]
entropies = [h['posterior_entropy'] for h in results['history']]

axes[0, 0].plot(scores, 'o-', linewidth=2, markersize=6, color='#2E86AB')
axes[0, 0].set_xlabel('Iteration', fontsize=11)
axes[0, 0].set_ylabel('Best Score', fontsize=11)
axes[0, 0].set_title('Optimization Convergence', fontsize=12, fontweight='bold')
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].plot(entropies, 's-', linewidth=2, markersize=6, color='#A23B72')
axes[0, 1].set_xlabel('Iteration', fontsize=11)
axes[0, 1].set_ylabel('Posterior Entropy (bits)', fontsize=11)
axes[0, 1].set_title('Quantum Posterior Entropy Evolution', fontsize=12, fontweight='bold')
axes[0, 1].grid(True, alpha=0.3)

selected_iterations = [0, len(results['quantum_history'])//2, len(results['quantum_history'])-1]
for idx, iter_num in enumerate(selected_iterations):
    if iter_num < len(results['quantum_history']):
        probs = results['quantum_history'][iter_num]['probability_distribution']
        top_indices = np.argsort(probs)[-8:][::-1]
        top_probs = probs[top_indices]
        config_labels = [f"{results['history'][iter_num]['configs'][0] if results['history'][iter_num].get('configs') else 'N/A'}"[:20] for _ in top_indices]
        axes[1, 0].bar(range(len(top_probs)), top_probs, alpha=0.6, label=f'Iter {iter_num}')
axes[1, 0].set_xlabel('Configuration Index', fontsize=11)
axes[1, 0].set_ylabel('Probability', fontsize=11)
axes[1, 0].set_title('Top Quantum Probabilities (Selected Iterations)', fontsize=12, fontweight='bold')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3, axis='y')

if len(results['quantum_history']) >= 3:
    initial_probs = results['quantum_history'][0]['probability_distribution']
    final_probs = results['quantum_history'][-1]['probability_distribution']
    axes[1, 1].plot(initial_probs[:16], 'o-', label='Initial (Iter 0)', alpha=0.7, linewidth=2)
    axes[1, 1].plot(final_probs[:16], 's-', label=f'Final (Iter {len(results["history"])-1})', alpha=0.7, linewidth=2)
    axes[1, 1].set_xlabel('Configuration Index', fontsize=11)
    axes[1, 1].set_ylabel('Probability', fontsize=11)
    axes[1, 1].set_title('Prior vs Posterior Distribution', fontsize=12, fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"Score improvement: {scores[-1] - scores[0]:.4f}")
print(f"Entropy reduction: {entropies[0] - entropies[-1]:.4f} bits (shows posterior concentration)")
