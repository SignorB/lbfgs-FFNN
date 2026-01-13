import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.integrate import quad

target_times = [0.0, 0.5, 1.2, 1.4]
colors = ['tab:blue', 'tab:green', 'tab:red', 'tab:orange']
nu = 0.3 / np.pi 

def exact_solution(x, t):
    if t == 0:
        return np.sin(np.pi * x)
    
    # Formula per la trasformata di Cole-Hopf
    def f(eta):
        return np.exp(-np.cos(np.pi * (x - eta)) / (2 * np.pi * nu * t)) * np.exp(-eta**2 / (4 * nu * t))
    
    def g(eta):
        return eta * f(eta) 
    
    def f_user(eta):
        return np.exp(np.cos(np.pi * (x - eta)) / (2 * np.pi * nu * t)) * np.exp(-eta**2 / (4 * nu * t))
    
    def g_user(eta):
        return eta * f_user(eta)

    num, _ = quad(g_user, -np.inf, np.inf)
    den, _ = quad(f_user, -np.inf, np.inf)
    return num / den if den != 0 else 0

try:
    df = pd.read_csv('burgers_test_extrapolation.csv')
except FileNotFoundError:
    print("File CSV non trovato.")
    exit()

plt.figure(figsize=(12, 8))
sns.set_theme(style="whitegrid")
x_range = np.linspace(-1, 1, 300)

for i, t in enumerate(target_times):

    available_t = df['t'].unique()
    nearest_t = available_t[np.argmin(np.abs(available_t - t))]
    subset = df[df['t'] == nearest_t].sort_values('x')
    
    label_suffix = " (Train)" if t <= 1.0 else " (Extrap)"
    marker = 'o' if t <= 1.0 else 'x'


    plt.scatter(subset['x'], subset['u'], color=colors[i], marker=marker, s=30, alpha=0.8, 
                label=f'PINN t={t:.1f}{label_suffix}')
    

    u_exact = [exact_solution(xi, t) for xi in x_range]
    plt.plot(x_range, u_exact, color=colors[i], linestyle='-', linewidth=2, alpha=0.6)

plt.title("Burgers' Equation: PINN vs Exact (u0 = sin(pi*x))", fontsize=16)
plt.xlabel("x")
plt.ylabel("u(x,t)")
plt.legend()
plt.tight_layout()

output_filename = 'burgers_plot.png'
plt.savefig(output_filename, dpi=300, bbox_inches='tight')
print(f"Immagine salvata correttamente in: {output_filename}")
plt.close()