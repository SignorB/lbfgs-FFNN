import matplotlib
matplotlib.use('Agg')
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

target_times = [0.0, 0.5, 1.0, 1.5]
colors = ['tab:blue', 'tab:green', 'tab:orange', 'tab:red']
nu = 0.01 / np.pi 

nx = 2000 
x = np.linspace(-1, 1, nx)
dx = x[1] - x[0]
u = -np.sin(np.pi * x) 
u = np.sin(np.pi * x)  

sigma = 0.5
dt_diff = dx**2 / (2 * nu)
dt_adv = dx / 1.0 # assumendo max|u| ~ 1
dt = sigma * min(dt_diff, dt_adv)

print(f"Simulating Reference Solution (nu={nu:.4f}, dx={dx:.5f})...")

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
        if t_targ > 0 and abs(current_t - t_targ) < dt:
            if t_targ not in history_u:
                history_u[t_targ] = u.copy()
                print(f"Snapshot t={t_targ} calcolato.")

print("Simulazione numerica completata.")

filename = 'burgers_test_extrapolation.csv'
try:
    df = pd.read_csv(filename)
    print(f"Dati caricati da {filename}")
except FileNotFoundError:
    print(f"ERRORE: File '{filename}' non trovato.")
    print("Esegui prima ./test_burgers_parallel per generare il CSV.")
    exit()


plt.figure(figsize=(12, 8))
sns.set_theme(style="whitegrid")

for i, t in enumerate(target_times):
    col = colors[i]

    if t in history_u:
        plt.plot(x, history_u[t], color=col, linestyle='-', linewidth=2, alpha=0.6,
                 label=f'Exact (FDM) t={t}')
    available_t = df['t'].unique()
    if len(available_t) > 0:
        nearest_t = available_t[np.argmin(np.abs(available_t - t))]
        if abs(nearest_t - t) < 0.05:
            subset = df[df['t'] == nearest_t].sort_values('x')
            label_suffix = " (Train)" if t <= 1.0 else " (EXTRAP)"
            marker_style = 'o' if t <= 1.0 else 'X' 
            plt.scatter(subset['x'], subset['u'], color=col, marker=marker_style, s=50, 
                        edgecolor='black', linewidth=0.5, zorder=10,
                        label=f'PINN t={nearest_t:.1f}{label_suffix}')

plt.title(f"Burgers' Equation: Enzyme PINN vs Reference (nu={nu:.4f})", fontsize=16)
plt.xlabel("x")
plt.ylabel("u(x,t)")
plt.ylim(-1.2, 1.2)
plt.legend()
plt.tight_layout()

output_img = "burgers_comparison.png"
plt.savefig(output_img, dpi=300)
print(f"Grafico salvato correttamente in: {output_img}")
plt.close()