import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import time
import csv
import sys

# CONFIGURAZIONE
INPUT_SIZE = 784
ENCODER_LAYERS = [128, 64, 32]
DECODER_LAYERS = [64, 128, 784]
MAX_ITERS = 500
DATA_PATH = "../fashion-mnist/"
CSV_FILE = "autoencoder_results.csv"

if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Running on GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("Running on CPU")

print("--- Loading Fashion-MNIST (Autoencoder Mode) ---")
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.view(-1))
])

train_dataset = datasets.FashionMNIST(root=DATA_PATH, train=True, download=True, transform=transform)
test_dataset = datasets.FashionMNIST(root=DATA_PATH, train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=False)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)

X_train, _ = next(iter(train_loader))
X_test, _ = next(iter(test_loader))

X_train = X_train.double().to(device)
X_test = X_test.double().to(device)

print(f"Data Loaded. Shape: {X_train.shape} (Double Precision)")

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        
        enc_modules = []
        in_dim = INPUT_SIZE
        for h_dim in ENCODER_LAYERS:
            enc_modules.append(nn.Linear(in_dim, h_dim))
            enc_modules.append(nn.ReLU())
            in_dim = h_dim
        self.encoder = nn.Sequential(*enc_modules)
        
        dec_modules = []
        for h_dim in DECODER_LAYERS[:-1]:
            dec_modules.append(nn.Linear(in_dim, h_dim))
            dec_modules.append(nn.ReLU())
            in_dim = h_dim
        dec_modules.append(nn.Linear(in_dim, DECODER_LAYERS[-1]))
        self.decoder = nn.Sequential(*dec_modules)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# TRAINER
def run_benchmark(opt_name):
    print(f"\n[{opt_name}] Preparing Training...")
    
    model = Autoencoder().double().to(device)
    criterion = nn.MSELoss()
    
    if opt_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=0.001)
    elif opt_name == "LBFGS":
        optimizer = optim.LBFGS(model.parameters(), 
                                lr=1.0, 
                                max_iter=MAX_ITERS, 
                                history_size=10, 
                                line_search_fn='strong_wolfe')
    
    if device.type == 'cuda': torch.cuda.synchronize()
    start_time = time.time()
    
    print(f"[{opt_name}] Starting Loop...")
    model.train()
    
    if opt_name == "LBFGS":
        def closure():
            optimizer.zero_grad()
            outputs = model(X_train)
            loss = criterion(outputs, X_train)
            loss.backward()
            return loss
        
        try:
            optimizer.step(closure)
        except Exception as e:
            print(f"CRASH L-BFGS: {e}")
            return None
            
    else:
        for i in range(MAX_ITERS):
            optimizer.zero_grad()
            outputs = model(X_train)
            loss = criterion(outputs, X_train)
            loss.backward()
            optimizer.step()
            
            if i % 50 == 0:
                print(f"  Iter {i} Loss: {loss.item():.6f}")

    if device.type == 'cuda': torch.cuda.synchronize()
    elapsed = time.time() - start_time
    
    model.eval()
    with torch.no_grad():
        train_rec = model(X_train)
        train_mse = criterion(train_rec, X_train).item()
        
        test_rec = model(X_test)
        test_mse = criterion(test_rec, X_test).item()

    print("-" * 40)
    print(f"RESULTS: {opt_name}")
    print(f"Time: {elapsed:.4f}s | Train MSE: {train_mse:.6f} | Test MSE: {test_mse:.6f}")
    print("-" * 40)
    
    return [opt_name, elapsed, train_mse, test_mse]

# ESECUZIONE
results = []
results.append(run_benchmark("Adam"))
res_lbfgs = run_benchmark("LBFGS")
if res_lbfgs:
    results.append(res_lbfgs)

