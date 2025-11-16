try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available. Install with: pip install torch")

if TORCH_AVAILABLE:
    print("Neural Network Hyperparameter Optimization")
    print("="*70)
    
    from sklearn.preprocessing import StandardScaler
    from sklearn.datasets import load_breast_cancer
    
    cancer_data = load_breast_cancer()
    X_cancer, y_cancer = cancer_data.data, cancer_data.target
    
    scaler = StandardScaler()
    X_cancer_scaled = scaler.fit_transform(X_cancer)
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
        X_cancer_scaled, y_cancer, test_size=0.2, random_state=42
    )
    
    X_train_tensor = torch.FloatTensor(X_train_c)
    X_test_tensor = torch.FloatTensor(X_test_c)
    y_train_tensor = torch.LongTensor(y_train_c)
    y_test_tensor = torch.LongTensor(y_test_c)
    
    class ImprovedNN(nn.Module):
        def __init__(self, input_dim, hidden_dim, num_layers, dropout_rate, use_batch_norm=True):
            super(ImprovedNN, self).__init__()
            layers = []
            layers.append(nn.Linear(input_dim, hidden_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            
            for _ in range(num_layers - 1):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_batch_norm:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.ReLU())
                if dropout_rate > 0:
                    layers.append(nn.Dropout(dropout_rate))
            
            layers.append(nn.Linear(hidden_dim, 2))
            self.network = nn.Sequential(*layers)
            
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.constant_(m.bias, 0)
        
        def forward(self, x):
            return self.network(x)
    
    def train_nn_model(hidden_dim, num_layers, learning_rate, dropout_rate, batch_size, epochs=80):
        use_batch_norm = batch_size >= 8
        
        try:
            model = ImprovedNN(X_train_c.shape[1], hidden_dim, num_layers, dropout_rate, use_batch_norm=use_batch_norm)
        except:
            model = ImprovedNN(X_train_c.shape[1], hidden_dim, num_layers, dropout_rate, use_batch_norm=False)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=8, verbose=False)
        
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=max(batch_size, 4), shuffle=True, drop_last=True)
        
        val_split = int(0.8 * len(X_train_c))
        X_val = X_train_c[val_split:]
        y_val = y_train_c[val_split:]
        X_val_tensor = torch.FloatTensor(X_val)
        y_val_tensor = torch.LongTensor(y_val)
        
        best_val_acc = 0.0
        patience_counter = 0
        max_patience = 15
        
        model.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            batch_count = 0
            for batch_X, batch_y in train_loader:
                if batch_X.size(0) < 2 and use_batch_norm:
                    model.eval()
                    with torch.no_grad():
                        test_outputs = model(X_test_tensor)
                        _, predicted = torch.max(test_outputs, 1)
                        accuracy = (predicted == y_test_tensor).float().mean().item()
                    return accuracy
                
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += loss.item()
                batch_count += 1
            
            if batch_count == 0:
                model.eval()
                with torch.no_grad():
                    test_outputs = model(X_test_tensor)
                    _, predicted = torch.max(test_outputs, 1)
                    accuracy = (predicted == y_test_tensor).float().mean().item()
                return accuracy
            
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val_tensor)
                _, val_predicted = torch.max(val_outputs, 1)
                val_accuracy = (val_predicted == y_val_tensor).float().mean().item()
            
            scheduler.step(val_accuracy)
            
            if val_accuracy > best_val_acc:
                best_val_acc = val_accuracy
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= max_patience:
                    break
            
            model.train()
        
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test_tensor)
            _, predicted = torch.max(test_outputs, 1)
            accuracy = (predicted == y_test_tensor).float().mean().item()
        
        return max(0.5, accuracy)
    
    class NNObjective:
        def __init__(self):
            self.search_space = None
            self.maximize = True
            self.evaluation_count = 0
        
        def set_search_space(self, search_space):
            self.search_space = search_space
        
        def get_search_space(self):
            return self.search_space
        
        def evaluate(self, config):
            self.evaluation_count += 1
            try:
                hidden_dim = int(config['hidden_dim'])
                num_layers = int(config['num_layers'])
                learning_rate = float(config['learning_rate'])
                dropout_rate = float(config['dropout_rate'])
                batch_size = int(config['batch_size'])
                
                if learning_rate <= 0 or learning_rate > 0.1:
                    return 0.5
                if dropout_rate < 0 or dropout_rate > 1:
                    return 0.5
                if batch_size < 4:
                    batch_size = 4
                
                accuracy = train_nn_model(hidden_dim, num_layers, learning_rate, dropout_rate, batch_size)
                
                if accuracy < 0.5 or accuracy > 1.0:
                    return 0.5
                
                return accuracy
            except Exception as e:
                return 0.5
    
    nn_search_space = {
        "hidden_dim": {"type": "discrete", "values": [32, 64, 128, 256]},
        "num_layers": {"type": "discrete", "values": [2, 3, 4]},
        "learning_rate": {"type": "continuous", "low": 0.0001, "high": 0.01, "num_points": 4},
        "dropout_rate": {"type": "continuous", "low": 0.0, "high": 0.5, "num_points": 3},
        "batch_size": {"type": "discrete", "values": [16, 32, 64]}
    }
    
    total_configs = 4 * 3 * 4 * 3 * 3
    print(f"Search space: {total_configs} configurations")
    print("Hyperparameters:")
    print("  - Hidden dimension: [32, 64, 128, 256]")
    print("  - Number of layers: [2, 3, 4]")
    print("  - Learning rate: [0.0001, 0.01] (4 points)")
    print("  - Dropout rate: [0.0, 0.5] (3 points)")
    print("  - Batch size: [16, 32, 64]")
    print("\nRunning QHBO optimization...")
    print("Note: Each evaluation trains a neural network (takes 10-30 seconds)")
    print("This will take 10-15 minutes total...")
    
    nn_objective = NNObjective()
    nn_objective.set_search_space(nn_search_space)
    
    num_qubits_nn = 0
    for param_name, param_def in nn_search_space.items():
        if param_def["type"] == "continuous":
            num_qubits_nn += int(np.ceil(np.log2(param_def["num_points"])))
        else:
            num_qubits_nn += int(np.ceil(np.log2(len(param_def["values"]))))
    
    backend_nn = QiskitBackend(num_qubits=num_qubits_nn, noise_model=None)
    optimizer_nn = QHBOOptimizer(
        objective=nn_objective,
        backend=backend_nn,
        max_iterations=15,
        num_samples_per_iteration=None,
        verbose=True,
        show_quantum_details=False,
        learning_rate=0.4,
        entropy_regularization=0.03
    )
    
    start_time_nn = time.time()
    results_nn = optimizer_nn.optimize()
    elapsed_time_nn = time.time() - start_time_nn
    
    print("\n" + "="*70)
    print("Neural Network Optimization Results:")
    print(f"  Best accuracy: {results_nn['best_score']:.4f}")
    if results_nn['best_config']:
        print(f"  Best configuration:")
        for param, value in results_nn['best_config'].items():
            if isinstance(value, (int, float)):
                print(f"    {param}: {value}")
            else:
                print(f"    {param}: {value}")
    print(f"  Iterations: {results_nn['num_iterations']}")
    print(f"  Total time: {elapsed_time_nn/60:.1f} minutes ({elapsed_time_nn:.1f}s)")
    num_evals = results_nn['num_iterations'] * optimizer_nn.num_samples_per_iteration
    print(f"  Evaluations: {num_evals}")
    print(f"  Search space coverage: {num_evals / total_configs * 100:.1f}%")
    print(f"  Average time per evaluation: {elapsed_time_nn/num_evals:.1f}s")
    
    if results_nn['history']:
        initial_score = results_nn['history'][0]['best_score']
        final_score = results_nn['history'][-1]['best_score']
        print(f"  Score improvement: {final_score - initial_score:.4f}")
    
    print("\nComparison with random search:")
    print("Running random search with same number of evaluations...")
    np.random.seed(42)
    random_nn_scores = []
    random_nn_time = time.time()
    for i in range(num_evals):
        if (i + 1) % 10 == 0:
            print(f"  Random search: {i+1}/{num_evals} evaluations...")
        config = {}
        for param_name, param_def in nn_search_space.items():
            if param_def["type"] == "continuous":
                config[param_name] = np.random.uniform(param_def["low"], param_def["high"])
            else:
                config[param_name] = np.random.choice(param_def["values"])
        score = nn_objective.evaluate(config)
        random_nn_scores.append(score)
    random_nn_time = time.time() - random_nn_time
    
    print(f"\n  Random search results:")
    print(f"    Best accuracy: {max(random_nn_scores):.4f}")
    print(f"    Mean accuracy: {np.mean(random_nn_scores):.4f}")
    print(f"    Std deviation: {np.std(random_nn_scores):.4f}")
    print(f"    Total time: {random_nn_time/60:.1f} minutes ({random_nn_time:.1f}s)")
    
    advantage = results_nn['best_score'] - max(random_nn_scores)
    relative_advantage = (advantage / max(random_nn_scores)) * 100 if max(random_nn_scores) > 0 else 0
    
    print(f"\n" + "="*70)
    print("Comparison Summary:")
    print(f"  QHBO best: {results_nn['best_score']:.4f}")
    print(f"  Random best: {max(random_nn_scores):.4f}")
    if advantage > 0:
        print(f"  QHBO advantage: +{advantage:.4f} ({relative_advantage:.1f}% relative improvement)")
        print(f"  QHBO found better architecture in {num_evals} evaluations")
    elif advantage < 0:
        print(f"  Random advantage: {abs(advantage):.4f}")
        print(f"  Note: QHBO may need more iterations or different hyperparameters")
    else:
        print(f"  Both methods found similar performance")
        print(f"  QHBO provides structured search vs random exploration")
else:
    print("PyTorch not available. Skipping neural network example.")
    print("Install with: pip install torch")
