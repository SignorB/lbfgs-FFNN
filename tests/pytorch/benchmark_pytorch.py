import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import time
import csv

# ==========================================
# CONFIGURAZIONE
# ==========================================
INPUT_SIZE = 784
NUM_CLASSES = 10
HIDDEN_LAYERS = [64] 
MAX_ITERS = 2000     
DATA_PATH = "../fashion-mnist/"
CSV_FILE = "benchmark_results.csv"

# ==========================================
# SETUP DEVICE
# ==========================================
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Running on GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("Running on CPU")

# ==========================================
# CARICAMENTO DATI (FULL BATCH)
# ==========================================
print("--- Loading Fashion-MNIST ---")
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.view(-1))
])

train_dataset = datasets.FashionMNIST(root=DATA_PATH, train=True, download=True, transform=transform)
test_dataset = datasets.FashionMNIST(root=DATA_PATH, train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=False)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)

X_train, y_train = next(iter(train_loader))
X_test, y_test = next(iter(test_loader))

# Spostamento dati su Device
X_train = X_train.to(device)
y_train = y_train.to(device)
X_test = X_test.to(device)
y_test = y_test.to(device)

# NOTA: Se L-BFGS crasha su GPU, scommenta queste righe per usare float64 (più stabile)
# X_train = X_train.double()
# X_test = X_test.double()

print(f"Data Loaded on Device. Shape: {X_train.shape}")

# ==========================================
# MODELLO
# ==========================================
class BenchmarkMLP(nn.Module):
    def __init__(self):
        super(BenchmarkMLP, self).__init__()
        layers = []
        in_dim = INPUT_SIZE
        
        for h_dim in HIDDEN_LAYERS:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.Tanh())
            in_dim = h_dim
        
        layers.append(nn.Linear(in_dim, NUM_CLASSES))
        
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='tanh')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.net(x)

# ==========================================
# BENCHMARK ENGINE
# ==========================================
def run_benchmark(opt_name, lr, momentum=0.0):
    print(f"\n[{opt_name}] Preparing Training (LR={lr}, Momentum={momentum})...")
    
    # Istanzia Modello (e converte a double se i dati lo sono)
    model = BenchmarkMLP().to(device)
    if X_train.dtype == torch.float64:
        model = model.double()
        
    criterion = nn.CrossEntropyLoss()
    
    # --- Selezione Optimizer ---
    if opt_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif opt_name == "GD":
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.0)
    elif opt_name == "SGD_Momentum":
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    elif opt_name == "LBFGS":
        # L-BFGS richiede max_iter nel costruttore per sapere quando fermarsi
        optimizer = optim.LBFGS(model.parameters(), 
                                lr=lr, 
                                max_iter=MAX_ITERS, 
                                history_size=10, 
                                line_search_fn='strong_wolfe')
    else:
        raise ValueError(f"Optimizer {opt_name} not supported")

    # Sincronizza Timer
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    start_time = time.time()
    
    model.train()
    
    # --- Training Loops Diversificati ---
    
    if opt_name == "LBFGS":
        # === LOGICA L-BFGS ===
        # L-BFGS richiede una funzione "closure" che valuta la loss
        def closure():
            optimizer.zero_grad()
            outputs = model(X_train)
            loss = criterion(outputs, y_train)
            loss.backward()
            return loss
        
        # .step() esegue internamente tutte le iterazioni fino a MAX_ITERS
        # o fino alla convergenza.
        try:
            optimizer.step(closure)
        except Exception as e:
            print(f"L-BFGS Error: {e}")

    else:
        # === LOGICA ADAM / SGD ===
        for i in range(MAX_ITERS):
            optimizer.zero_grad()
            outputs = model(X_train)
            loss = criterion(outputs, y_train)
            loss.backward()
            optimizer.step()
            
            # Logging manuale
            if i % (MAX_ITERS // 10) == 0:
                print(f"  Iter {i}/{MAX_ITERS} - Loss: {loss.item():.6f}")

    # Fine Timer
    if device.type == 'cuda':
        torch.cuda.synchronize()
        
    elapsed_time = time.time() - start_time
    
    # --- Valutazione Finale ---
    model.eval()
    with torch.no_grad():
        train_out = model(X_train)
        final_loss = criterion(train_out, y_train).item()
        _, train_pred = torch.max(train_out, 1)
        train_acc = (train_pred == y_train).float().mean().item()

        test_out = model(X_test)
        _, test_pred = torch.max(test_out, 1)
        test_acc = (test_pred == y_test).float().mean().item()

    print("-" * 40)
    print(f"RESULTS: {opt_name}")
    print("-" * 40)
    print(f"Time Taken:     {elapsed_time:.4f} seconds")
    print(f"Final Loss:     {final_loss:.6f}")
    print(f"Train Accuracy: {train_acc*100:.2f}%")
    print(f"Test Accuracy:  {test_acc*100:.2f}%")
    print("=" * 40)
    
    return [opt_name, lr, momentum, elapsed_time, final_loss, train_acc, test_acc]

# ==========================================
# ESECUZIONE TEST
# ==========================================
results = []

# 1. Adam (Baseline Veloce)
results.append(run_benchmark("Adam", lr=0.001))

# 2. SGD con Momentum (Baseline Classica)
results.append(run_benchmark("SGD_Momentum", lr=0.01, momentum=0.9))

# 3. L-BFGS (Target Competitor)
# Nota: L-BFGS di solito vuole lr=1.0
results.append(run_benchmark("LBFGS", lr=1.0, momentum=0.0))

# Salvataggio CSV
try:
    with open(CSV_FILE, mode='w', newline='') as f:
        writer = csv.writer(f)
        header = ["optimizer", "lr", "momentum", "time_sec", "final_loss", "train_acc", "test_acc"]
        writer.writerow(header)
        writer.writerows(results)
    print(f"\nRisultati salvati in {CSV_FILE}")
except IOError as e:
    print(f"Errore salvataggio CSV: {e}")