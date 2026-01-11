import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

target_times = [0.0, 0.5, 1.2, 1.4]
colors = ['tab:blue', 'tab:green', 'tab:red', 'tab:orange']
nu = 0.03 / np.pi 

# sampling point on x
nx = 100
x = np.linspace(-1, 1, nx)
dx = x[1] - x[0]
u = np.sin(np.pi * x) # initial condition


sigma = 0.5
dt_diff = dx**2 / (2 * nu)
dt_adv = dx / 1.0 
dt = sigma * min(dt_diff, dt_adv)

print(f"Simulating (nu={nu:.4f})...")

history_u = {} 
history_u[0.0] = u.copy()

t_max = max(target_times)
current_t = 0.0

while current_t < t_max:
    u_prev = u.copy()
    
    u_xx = (np.roll(u_prev, -1) - 2*u_prev + np.roll(u_prev, 1)) / dx**2

    u_x = (np.roll(u_prev, -1) - np.roll(u_prev, 1)) / (2*dx)
    

    u = u_prev + dt * (-u_prev * u_x + nu * u_xx)
    
    u[0] = 0.0
    u[-1] = 0.0
    
    current_t += dt
    
    for t_targ in target_times:
        if t_targ > 0 and abs(current_t - t_targ) < dt/2:
            history_u[t_targ] = u.copy()

print("Simulazione completata.")

try:
    df = pd.read_csv('burgers_test_extrapolation.csv')
except FileNotFoundError:
    print("ERRORE: File 'burgers_test_extrapolation.csv' non trovato.")
    exit()

plt.figure(figsize=(12, 8))
sns.set_theme(style="whitegrid")

for i, t in enumerate(target_times):
    col = colors[i]
    

    if t in history_u:
        plt.plot(x, history_u[t], color=col, linestyle='-', linewidth=2, 
                 label=f'Exact t={t}')
    
    
    available_t = df['t'].unique()
    nearest_t = available_t[np.argmin(np.abs(available_t - t))]

    subset = df[df['t'] == nearest_t].sort_values('x')
    
    label_suffix = " (Train)" if t <= 1.0 else " (Extrap)"
    marker = 'o' 
    
    plt.scatter(subset['x'], subset['u'], color=col, marker=marker, s=40, alpha=0.7, 
                edgecolor='black', linewidth=0.5,
                label=f'PINN t={nearest_t:.1f}{label_suffix}')

plt.title(f"Burgers' Equation: PINN vs Numerical Exact (nu={nu:.4f})", fontsize=16)
plt.xlabel("x")
plt.ylabel("u(x,t)")
plt.legend()
plt.tight_layout()
plt.show()
