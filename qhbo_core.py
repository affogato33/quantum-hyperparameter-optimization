import numpy as np
from typing import Dict, Any, Optional, List
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Statevector
from tqdm import tqdm

class CircuitEncoder:
    def __init__(self, search_space):
        self.search_space = search_space
        self.param_names = list(search_space.keys())
        self.param_configs = {}
        self.total_qubits = 0
        self._process_search_space()
    
    def _process_search_space(self):
        qubit_idx = 0
        for param_name, param_def in self.search_space.items():
            param_type = param_def.get("type", "continuous")
            if param_type == "continuous":
                num_points = param_def.get("num_points", 8)
                low, high = param_def["low"], param_def["high"]
                values = np.linspace(low, high, num_points)
                num_qubits = int(np.ceil(np.log2(num_points)))
            else:
                values = param_def["values"]
                num_qubits = int(np.ceil(np.log2(len(values))))
            self.param_configs[param_name] = {"values": values, "num_qubits": num_qubits}
            qubit_idx += num_qubits
        self.total_qubits = qubit_idx
    
    def config_to_state_index(self, config):
        indices, multipliers = [], []
        for param_name in self.param_names:
            value = config[param_name]
            values = self.param_configs[param_name]["values"]
            if isinstance(values, np.ndarray):
                idx = np.argmin(np.abs(values - value))
            else:
                idx = values.index(value) if value in values else 0
            indices.append(idx)
            multipliers.append(len(values))
        linear_idx = sum(idx * np.prod(multipliers[i+1:]) if i < len(multipliers)-1 else idx 
                        for i, idx in enumerate(indices))
        return linear_idx
    
    def index_to_config(self, linear_idx):
        config, multipliers = {}, []
        for param_name in self.param_names:
            multipliers.append(len(self.param_configs[param_name]["values"]))
        for i, param_name in enumerate(self.param_names):
            multiplier = np.prod(multipliers[i+1:]) if i < len(multipliers)-1 else 1
            idx = (linear_idx // multiplier) % multipliers[i]
            config[param_name] = self.param_configs[param_name]["values"][idx]
        return config
    
    def sample_configs_from_state(self, state, num_samples=1):
        probs = np.abs(state) ** 2
        probs = probs / np.sum(probs)
        indices = np.random.choice(len(probs), size=num_samples, p=probs)
        return [self.index_to_config(int(idx)) for idx in indices]
    
    def get_num_qubits(self):
        return self.total_qubits
    
    def get_num_configs(self):
        return int(np.prod([len(self.param_configs[p]["values"]) for p in self.param_names]))

class QiskitBackend:
    def __init__(self, num_qubits, noise_model=None):
        self.num_qubits = num_qubits
        self.noise_model = noise_model
        self.simulator = AerSimulator()
        if noise_model:
            self.simulator.set_options(noise_model=noise_model)
    
    def create_circuit(self, num_qubits=None):
        return QuantumCircuit(num_qubits or self.num_qubits)
    
    def get_state(self, circuit):
        return np.array(Statevector.from_instruction(circuit))
    
    def measure(self, circuit):
        state = self.get_state(circuit)
        return np.abs(state) ** 2
    
    def get_num_qubits(self):
        return self.num_qubits

class BayesianQuantumCircuit:
    def __init__(self, encoder, backend, num_layers=3):
        self.encoder = encoder
        self.backend = backend
        self.num_layers = num_layers
        self.num_qubits = encoder.get_num_qubits()
        num_params = num_layers * (2 * self.num_qubits + self.num_qubits - 1)
        self.params = np.random.uniform(0, 2*np.pi, size=num_params)
        self.circuit = None
        self._build_circuit()
    
    def _build_circuit(self):
        self.circuit = self.backend.create_circuit(self.num_qubits)
        param_idx = 0
        
        for layer in range(self.num_layers):
            for qubit in range(self.num_qubits):
                if param_idx < len(self.params):
                    self.circuit.ry(self.params[param_idx], qubit)
                    param_idx += 1
                if param_idx < len(self.params):
                    self.circuit.rz(self.params[param_idx], qubit)
                    param_idx += 1
            
            for qubit in range(self.num_qubits - 1):
                self.circuit.cx(qubit, qubit + 1)
    
    def get_state(self):
        return self.backend.get_state(self.circuit)
    
    def get_probability_distribution(self):
        return self.backend.measure(self.circuit)
    
    def sample_configs(self, num_samples=1):
        state = self.get_state()
        return self.encoder.sample_configs_from_state(state, num_samples)
    
    def update_params(self, new_params):
        self.params = np.array(new_params)
        self._build_circuit()
    
    def get_params(self):
        return self.params.copy()

class PosteriorUpdater:
    def __init__(self, circuit, learning_rate=0.4, entropy_regularization=0.03, exploration_temperature=1.0):
        self.circuit = circuit
        self.learning_rate = learning_rate
        self.entropy_regularization = entropy_regularization
        self.exploration_temperature = exploration_temperature
        self.observation_history = []
        self.baseline_score = None
        self.iteration = 0
        self.best_config_idx = None
    
    def update(self, configs, scores, best_score):
        self.iteration += 1
        for config, score in zip(configs, scores):
            self.observation_history.append({"config": config, "score": score})
        
        if self.baseline_score is None:
            self.baseline_score = np.mean(scores)
        else:
            self.baseline_score = 0.9 * self.baseline_score + 0.1 * np.mean(scores)
        
        best_observed = max(self.observation_history, key=lambda x: x["score"])
        self.best_config_idx = self.circuit.encoder.config_to_state_index(best_observed["config"])
        
        config_indices = np.array([self.circuit.encoder.config_to_state_index(c) for c in configs])
        scores_array = np.array(scores)
        
        current_probs = self.circuit.get_probability_distribution()
        current_params = self.circuit.get_params()
        current_entropy = self.get_posterior_entropy()
        
        reward_weights = scores_array - self.baseline_score
        if np.abs(reward_weights).max() > 1e-6:
            reward_weights = reward_weights / (np.abs(reward_weights).max() + 1e-6)
        
        reward_gradient = self._compute_gradient_improved(config_indices, reward_weights, current_probs)
        
        if current_entropy > 1.0:
            entropy_gradient = self._compute_entropy_gradient(current_entropy)
            total_gradient = reward_gradient + self.entropy_regularization * entropy_gradient
        else:
            total_gradient = reward_gradient
        
        gradient_norm = np.linalg.norm(total_gradient)
        if gradient_norm > 0:
            total_gradient = total_gradient / gradient_norm
        
        decay_factor = max(0.7, 1.0 - self.iteration * 0.005)
        adaptive_lr = self.learning_rate * decay_factor
        
        new_params = current_params + adaptive_lr * total_gradient
        
        if self.best_config_idx is not None and self.best_config_idx < len(current_probs):
            best_boost = self._compute_best_config_boost(current_probs, self.best_config_idx)
            boost_strength = 0.2 * (1.0 - min(1.0, current_entropy / np.log2(len(current_probs))))
            new_params = new_params + boost_strength * best_boost
        
        new_params = np.clip(new_params, 0, 2*np.pi)
        
        self.circuit.update_params(new_params)
    
    def _compute_best_config_boost(self, current_probs, best_idx):
        boost_gradient = np.zeros_like(self.circuit.params)
        epsilon = 0.05
        
        for i in range(len(self.circuit.params)):
            self.circuit.params[i] += epsilon
            self.circuit._build_circuit()
            perturbed_probs = self.circuit.get_probability_distribution()
            
            current_prob = current_probs[best_idx] if best_idx < len(current_probs) else 0
            perturbed_prob = perturbed_probs[best_idx] if best_idx < len(perturbed_probs) else 0
            
            boost_gradient[i] = (perturbed_prob - current_prob) / epsilon
            
            self.circuit.params[i] -= epsilon
            self.circuit._build_circuit()
        
        return boost_gradient
    
    def _compute_gradient_improved(self, config_indices, reward_weights, current_probs):
        gradient = np.zeros_like(self.circuit.params)
        epsilon = 0.1
        
        for i in range(len(self.circuit.params)):
            self.circuit.params[i] += epsilon
            self.circuit._build_circuit()
            perturbed_probs = self.circuit.get_probability_distribution()
            
            current_value = sum(current_probs[idx] * weight if idx < len(current_probs) else 0 
                               for idx, weight in zip(config_indices, reward_weights))
            perturbed_value = sum(perturbed_probs[idx] * weight if idx < len(perturbed_probs) else 0 
                                 for idx, weight in zip(config_indices, reward_weights))
            
            gradient[i] = (perturbed_value - current_value) / epsilon
            
            self.circuit.params[i] -= epsilon
            self.circuit._build_circuit()
        
        return gradient
    
    def _compute_entropy_gradient(self, current_entropy):
        gradient = np.zeros_like(self.circuit.params)
        epsilon = 0.1
        target_entropy = max(1.0, np.log2(len(self.circuit.get_probability_distribution())) * 0.3)
        
        for i in range(len(self.circuit.params)):
            self.circuit.params[i] += epsilon
            self.circuit._build_circuit()
            perturbed_probs = self.circuit.get_probability_distribution()
            perturbed_probs = perturbed_probs[perturbed_probs > 1e-10]
            if len(perturbed_probs) > 0:
                perturbed_entropy = -np.sum(perturbed_probs * np.log2(perturbed_probs))
                entropy_diff = (perturbed_entropy - target_entropy) - (current_entropy - target_entropy)
                gradient[i] = entropy_diff / epsilon
            else:
                gradient[i] = 0
            
            self.circuit.params[i] -= epsilon
            self.circuit._build_circuit()
        
        return gradient
    
    def suggest_next_configs(self, num_suggestions=1):
        probs = self.circuit.get_probability_distribution()
        probs = probs / np.sum(probs)
        
        current_entropy = self.get_posterior_entropy()
        max_entropy = np.log2(len(probs))
        exploration_factor = min(1.0, current_entropy / (max_entropy * 0.3))
        
        if len(self.observation_history) > 0 and exploration_factor < 0.5:
            best_observed = max(self.observation_history, key=lambda x: x["score"])
            best_idx = self.circuit.encoder.config_to_state_index(best_observed["config"])
            
            if best_idx < len(probs):
                boost_factor = 1.0 + (1.0 - exploration_factor) * 0.3
                probs[best_idx] *= boost_factor
                probs = probs / np.sum(probs)
        
        temperature = self.exploration_temperature * (1.0 + exploration_factor)
        log_probs = np.log(probs + 1e-10)
        tempered_probs = np.exp(log_probs / temperature)
        tempered_probs = tempered_probs / np.sum(tempered_probs)
        
        state = self.circuit.get_state()
        state_normalized = np.sqrt(tempered_probs) * np.exp(1j * np.angle(state))
        
        return self.circuit.encoder.sample_configs_from_state(state_normalized, num_suggestions)
    
    def get_posterior_entropy(self):
        probs = self.circuit.get_probability_distribution()
        probs = probs[probs > 1e-10]
        if len(probs) == 0:
            return 0
        return -np.sum(probs * np.log2(probs))

class ConvergenceChecker:
    def __init__(self, improvement_threshold=1e-4, plateau_length=7, min_iterations=15):
        self.improvement_threshold = improvement_threshold
        self.plateau_length = plateau_length
        self.min_iterations = min_iterations
        self.history = []
    
    def update(self, iteration, score, best_score):
        improvement = best_score - (self.history[-1]["best_score"] if self.history else best_score)
        self.history.append({"iteration": iteration, "score": score, "best_score": best_score, "improvement": improvement})
    
    def check_convergence(self):
        if len(self.history) < self.min_iterations:
            return {"converged": False, "reason": "min_iterations"}
        recent_best = [h["best_score"] for h in self.history[-self.plateau_length:]]
        if len(recent_best) >= self.plateau_length:
            improvement = max(recent_best) - min(recent_best)
            if improvement < self.improvement_threshold:
                return {"converged": True, "reason": "plateau"}
        return {"converged": False, "reason": "continuing"}
    
    def get_statistics(self):
        if not self.history:
            return {}
        best_scores = [h["best_score"] for h in self.history]
        return {"total_iterations": len(self.history), "initial_score": best_scores[0], 
                "final_score": best_scores[-1], "total_improvement": best_scores[-1] - best_scores[0]}

class SklearnObjective:
    def __init__(self, model_class, X_train, y_train, X_test=None, y_test=None, metric="accuracy", maximize=True):
        self.model_class = model_class
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.metric = metric
        self.maximize = maximize
        self.search_space = None
    
    def set_search_space(self, search_space):
        self.search_space = search_space
    
    def get_search_space(self):
        return self.search_space
    
    def evaluate(self, config):
        try:
            model = self.model_class(**config)
            if self.X_test is not None and self.y_test is not None:
                model.fit(self.X_train, self.y_train)
                return model.score(self.X_test, self.y_test)
            else:
                scores = cross_val_score(model, self.X_train, self.y_train, cv=5, scoring="accuracy")
                return np.mean(scores)
        except:
            return -np.inf if self.maximize else np.inf

class QHBOOptimizer:
    def __init__(self, objective, backend, max_iterations=50, num_samples_per_iteration=None, 
                 noise_mitigation=False, verbose=True, random_seed=None, show_quantum_details=False,
                 learning_rate=0.4, entropy_regularization=0.03):
        self.objective = objective
        self.backend = backend
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.show_quantum_details = show_quantum_details
        if random_seed:
            np.random.seed(random_seed)
        search_space = objective.get_search_space()
        if search_space is None:
            raise ValueError("Objective must have search space defined")
        self.encoder = CircuitEncoder(search_space)
        num_configs = self.encoder.get_num_configs()
        
        if num_samples_per_iteration is None:
            self.num_samples_per_iteration = max(3, int(np.log2(num_configs)))
        else:
            self.num_samples_per_iteration = num_samples_per_iteration
        
        self.circuit = BayesianQuantumCircuit(self.encoder, self.backend)
        self.updater = PosteriorUpdater(self.circuit, learning_rate=learning_rate, 
                                        entropy_regularization=entropy_regularization)
        self.convergence = ConvergenceChecker()
        self.history = []
        self.quantum_history = []
        self.best_config = None
        self.best_score = -np.inf if objective.maximize else np.inf
    
    def optimize(self):
        if self.verbose:
            print("Starting QHBO optimization...")
            print(f"Search space: {len(self.encoder.param_names)} parameters")
            print(f"\nQuantum Circuit Architecture:")
            print(f"  Qubits: {self.circuit.num_qubits} (encoding {len(self.encoder.param_names)} hyperparameters)")
            print(f"  Ansatz: Hardware-efficient with RY-RZ-CNOT layers")
            print(f"  Depth: {self.circuit.num_layers} layers")
            print(f"  Trainable parameters: {len(self.circuit.params)}")
            if self.backend.noise_model:
                print(f"  Noise model: Active")
            else:
                print(f"  Noise model: None (ideal simulation)")
        iterator = range(self.max_iterations)
        if self.verbose:
            iterator = tqdm(iterator, desc="Optimizing")
        for iteration in iterator:
            quantum_state = self.circuit.get_state()
            probs = self.circuit.get_probability_distribution()
            configs = self.updater.suggest_next_configs(self.num_samples_per_iteration)
            
            if self.show_quantum_details and iteration < 3:
                print(f"\nIteration {iteration} - Quantum State Analysis:")
                print(f"  State vector shape: {quantum_state.shape}")
                entropy = self.updater.get_posterior_entropy()
                print(f"  Probability distribution entropy: {entropy:.4f} bits")
                top_probs = np.argsort(probs)[-3:][::-1]
                print(f"  Top 3 most probable configurations:")
                for idx in top_probs:
                    config = self.encoder.index_to_config(idx)
                    bitstring = format(idx, f'0{self.circuit.num_qubits}b')
                    print(f"    |{bitstring}⟩ (idx {idx}): P={probs[idx]:.4f}, Config={config}")
                print(f"  Circuit parameters (first 5): {self.circuit.get_params()[:5]}")
            
            scores = []
            measurement_outcomes = []
            for config in configs:
                state_idx = self.encoder.config_to_state_index(config)
                measurement_outcomes.append(state_idx)
                score = self.objective.evaluate(config)
                scores.append(score)
                if (self.objective.maximize and score > self.best_score) or (not self.objective.maximize and score < self.best_score):
                    self.best_score = score
                    self.best_config = config.copy()
            
            if self.show_quantum_details and iteration < 3:
                print(f"  Sampled configurations (indices): {measurement_outcomes}")
                print(f"  Classical scores: {[f'{s:.4f}' for s in scores]}")
                print(f"  Best score so far: {self.best_score:.4f}")
            
            old_params = self.circuit.get_params().copy()
            old_entropy = self.updater.get_posterior_entropy()
            
            self.updater.update(configs, scores, self.best_score)
            
            new_params = self.circuit.get_params().copy()
            new_entropy = self.updater.get_posterior_entropy()
            
            if self.show_quantum_details and iteration < 3:
                param_change = np.linalg.norm(new_params - old_params)
                entropy_change = old_entropy - new_entropy
                print(f"  Parameter update magnitude: {param_change:.4f}")
                print(f"  Entropy change: {entropy_change:+.4f} bits")
                print(f"  New circuit params (first 5): {new_params[:5]}")
            self.convergence.update(iteration, np.mean(scores), self.best_score)
            
            quantum_info = {
                "quantum_state": quantum_state.copy(),
                "probability_distribution": probs.copy(),
                "posterior_entropy": self.updater.get_posterior_entropy(),
                "measurement_outcomes": measurement_outcomes,
                "circuit_params": self.circuit.get_params().copy()
            }
            
            self.history.append({"iteration": iteration, "configs": configs, "scores": scores, 
                                "best_score": self.best_score, "best_config": self.best_config.copy() if self.best_config else None,
                                "posterior_entropy": self.updater.get_posterior_entropy()})
            self.quantum_history.append(quantum_info)
            
            conv_result = self.convergence.check_convergence()
            if conv_result["converged"]:
                if self.verbose:
                    print(f"\nConverged at iteration {iteration}: {conv_result['reason']}")
                break
        return {"best_config": self.best_config, "best_score": self.best_score, 
                "num_iterations": len(self.history), "history": self.history,
                "quantum_history": self.quantum_history,
                "convergence_stats": self.convergence.get_statistics()}

print("All QHBO classes loaded successfully!")
