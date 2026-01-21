import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import struct

TRAIN_IMG_PATH = "train-images.idx3-ubyte"
TRAIN_LBL_PATH = "train-labels.idx1-ubyte"
TEST_IMG_PATH = "t10k-images.idx3-ubyte"
TEST_LBL_PATH = "t10k-labels.idx1-ubyte"

TRAIN_SIZE = 5000
TEST_SIZE = 1000
INPUT_SIZE = 784
HIDDEN_SIZE = 128
OUTPUT_SIZE = 10
LEARNING_RATE = 0.00001
MOMENTUM = 0.9
MAX_ITERATIONS = 2000
DEVICE = torch.device("cuda")

def load_images(path, n):
    with open(path, 'rb') as f:
        magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
        data = np.frombuffer(f.read(n * rows * cols), dtype=np.uint8).copy()
        return torch.from_numpy(data).float().view(n, -1) / 255.0

def load_labels(path, n):
    with open(path, 'rb') as f:
        magic, num = struct.unpack('>II', f.read(8))
        data = np.frombuffer(f.read(n), dtype=np.uint8).copy()
        return torch.from_numpy(data).long()

print("Loading Training Data...")
train_x = load_images(TRAIN_IMG_PATH, TRAIN_SIZE).to(DEVICE)
train_y_indices = load_labels(TRAIN_LBL_PATH, TRAIN_SIZE).to(DEVICE)
train_y = F.one_hot(train_y_indices, num_classes=10).float()

print("Loading Test Data...")
test_x = load_images(TEST_IMG_PATH, TEST_SIZE).to(DEVICE)
test_y_indices = load_labels(TEST_LBL_PATH, TEST_SIZE).to(DEVICE)
test_y = F.one_hot(test_y_indices, num_classes=10).float()

model = nn.Sequential(
    nn.Linear(INPUT_SIZE, HIDDEN_SIZE),
    nn.Tanh(),
    nn.Linear(HIDDEN_SIZE, OUTPUT_SIZE)
).to(DEVICE)

def init_weights(m):
    if isinstance(m, nn.Linear):
        std = (1.0 / m.in_features) ** 0.5
        nn.init.normal_(m.weight, mean=0.0, std=std)
        nn.init.normal_(m.bias, mean=0.0, std=std)

model.apply(init_weights)

optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)

print("\nStarting Training (SGD)...")

for i in range(MAX_ITERATIONS):
    outputs = model(train_x)
    
    diff = outputs - train_y
    loss = 0.5 * (diff.pow(2).sum())
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print(f"MSE: {loss.item()}")

def evaluate(name, x, targets_onehot, targets_idx):
    with torch.no_grad():
        outputs = model(x)
        diff = outputs - targets_onehot
        mse = 0.5 * (diff.pow(2).sum()).item()
        
        _, predicted = torch.max(outputs, 1)
        correct = (predicted == targets_idx).sum().item()
        total = x.size(0)
        accuracy = 100.0 * correct / total
        
        print(f"=== {name} ===")
        print(f"Samples: {total}")
        print(f"Accuracy: {accuracy}% ({correct}/{total})")
        print(f"Total MSE: {mse}")
        print("====================")

print("\nTRAINING SET RESULTS:")
evaluate("Test Results", train_x, train_y, train_y_indices)

print("\nTEST SET RESULTS:")
evaluate("Test Results", test_x, test_y, test_y_indices)
