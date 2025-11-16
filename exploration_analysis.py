print("Exploration Analysis:")
print("="*60)

all_evaluated_configs = []
for h in results['history']:
    for config, score in zip(h['configs'], h['scores']):
        all_evaluated_configs.append({
            'config': config,
            'score': score,
            'iteration': h['iteration']
        })

all_evaluated_configs.sort(key=lambda x: x['score'], reverse=True)

print(f"Total unique configurations evaluated: {len(set(str(c['config']) for c in all_evaluated_configs))}")
print(f"Total evaluations: {len(all_evaluated_configs)}")
print(f"\nTop 5 configurations found:")
for i, eval_config in enumerate(all_evaluated_configs[:5], 1):
    print(f"  {i}. Score: {eval_config['score']:.4f}, Config: {eval_config['config']}, "
          f"Found at iteration: {eval_config['iteration']}")

print(f"\nBest configuration found:")
print(f"  Score: {results['best_score']:.4f}")
print(f"  Config: {results['best_config']}")

print(f"\nSearch space coverage:")
C_values = search_space['C']['values'] if 'values' in search_space['C'] else np.linspace(
    search_space['C']['low'], search_space['C']['high'], search_space['C']['num_points'])
gamma_values = search_space['gamma']['values'] if 'values' in search_space['gamma'] else np.linspace(
    search_space['gamma']['low'], search_space['gamma']['high'], search_space['gamma']['num_points'])

print(f"  C range: {C_values}")
print(f"  gamma range: {gamma_values}")

evaluated_C = [c['config']['C'] for c in all_evaluated_configs]
evaluated_gamma = [c['config']['gamma'] for c in all_evaluated_configs]

print(f"\n  Evaluated C values: {sorted(set([round(c, 3) for c in evaluated_C]))}")
print(f"  Evaluated gamma values: {sorted(set([round(g, 4) for g in evaluated_gamma]))}")

print(f"\nFinal quantum state analysis:")
final_probs = results['quantum_history'][-1]['probability_distribution']
top_indices = np.argsort(final_probs)[-5:][::-1]
print(f"  Top 5 most probable configurations in final quantum state:")
for idx in top_indices:
    config = optimizer.encoder.index_to_config(idx)
    prob = final_probs[idx]
    bitstring = format(idx, f'0{optimizer.circuit.num_qubits}b')
    print(f"    |{bitstring}⟩ (idx {idx}): P={prob:.4f}, Config={config}")
