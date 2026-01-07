import numpy as np
import struct
import os

N_TRAIN = 150000
N_TEST = 10000
INPUT_DIM = 2048
OUTPUT_DIM = 10
TEACHER_HIDDEN = 512
INPUT_SCALE = 1.0
NOISE_STD_TRAIN = 0.05
NOISE_STD_TEST = 0.05
SEED = 231

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "data_bench")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def save_binary_matrix(filename, matrix):
    rows, cols = matrix.shape
    print(f"Salvataggio {filename}: {rows}x{cols} (float32)")
    
    with open(filename, 'wb') as f:
        f.write(struct.pack('ii', rows, cols))
        matrix.astype(np.float32).tofile(f)


def init_weights(rng, fan_in, fan_out):
    std = np.sqrt(1.0 / fan_in)
    return rng.normal(0.0, std, size=(fan_in, fan_out))


def teacher_forward(x, w1, b1, w2, b2):
    h = np.tanh(x @ w1 + b1)
    return h @ w2 + b2


def generate():
    rng = np.random.default_rng(SEED)
    print("Generazione dati sintetici (teacher-student)...")

    train_x = rng.normal(0.0, 1.0, size=(N_TRAIN, INPUT_DIM))
    test_x = rng.normal(0.0, 1.0, size=(N_TEST, INPUT_DIM))

    if INPUT_SCALE != 1.0:
        train_x *= INPUT_SCALE
        test_x *= INPUT_SCALE

    w1 = init_weights(rng, INPUT_DIM, TEACHER_HIDDEN)
    b1 = rng.normal(0.0, 0.01, size=(TEACHER_HIDDEN,))
    w2 = init_weights(rng, TEACHER_HIDDEN, OUTPUT_DIM)
    b2 = rng.normal(0.0, 0.01, size=(OUTPUT_DIM,))

    train_y = teacher_forward(train_x, w1, b1, w2, b2)
    test_y = teacher_forward(test_x, w1, b1, w2, b2)

    if NOISE_STD_TRAIN > 0.0:
        train_y += rng.normal(0.0, NOISE_STD_TRAIN, size=train_y.shape)
    if NOISE_STD_TEST > 0.0:
        test_y += rng.normal(0.0, NOISE_STD_TEST, size=test_y.shape)
    
    save_binary_matrix(os.path.join(OUTPUT_DIR, "train_x.bin"), train_x)
    save_binary_matrix(os.path.join(OUTPUT_DIR, "train_y.bin"), train_y)
    save_binary_matrix(os.path.join(OUTPUT_DIR, "test_x.bin"), test_x)
    save_binary_matrix(os.path.join(OUTPUT_DIR, "test_y.bin"), test_y)
    
    print("Fatto.")

if __name__ == "__main__":
    generate()
