import matplotlib
matplotlib.use('Agg')  # Backend non interattivo
import matplotlib.pyplot as plt
import pandas as pd
import glob
import os
import argparse

# Setup stile robusto
try:
    plt.style.use('seaborn-v0_8-whitegrid')
except:
    try:
        plt.style.use('seaborn-whitegrid')
    except:
        plt.style.use('ggplot')

def find_history_files():
    files = set()
    files.update(glob.glob("*_history.csv"))
    files.update(glob.glob("**/*_history.csv", recursive=True))
    return sorted(files)


def select_loss_column(df):
    if "TrainLoss" in df.columns:
        return "TrainLoss"
    if "Loss" in df.columns:
        return "Loss"
    return None


def plot_combined(title=None):
    # 1. Cerca i file
    files = find_history_files()
    if not files:
        print("Nessun file trovato.")
        return

    # Ordiniamo i file per nome così la legenda è pulita (es. GD prima, poi LBFGS)
    files.sort()
    
    print(f"Trovati {len(files)} file. Generazione grafico combinato...")

    # 2. Crea una figura con 3 subplots (1 riga, 3 colonne)
    fig, (ax_time, ax_iter, ax_grad) = plt.subplots(1, 3, figsize=(24, 8))
    fig_acc, (ax_acc_time, ax_acc_iter) = plt.subplots(1, 2, figsize=(18, 8))
    
    # Titolo generale opzionale
    if title:
        fig.suptitle(title, fontsize=16, weight='bold')

    has_data = False
    has_grad_data = False
    has_acc_data = False

    for f in files:
        try:
            # Nome pulito per la legenda
            name = f.replace("_history.csv", "")
            if name.startswith("FASHION_"):
                name = name[len("FASHION_"):]
            elif name.startswith("MNIST_"):
                name = name[len("MNIST_"):]
            
            # Caricamento dati
            df = pd.read_csv(f)

            loss_col = select_loss_column(df)
            if df.empty or loss_col is None:
                continue

            # Conversione esplicita a numpy (fix per pandas recenti)
            loss = df[loss_col].to_numpy()
            
            time_data = None
            # --- PLOT 1: TEMPO (Se disponibile) ---
            if 'Time(s)' in df.columns:
                time_data = df['Time(s)'].to_numpy()
                ax_time.plot(time_data, loss, label=name, linewidth=2, alpha=0.8)
                ax_time.set_xlabel('Time (s)', fontsize=12)
            elif 'TimeMs' in df.columns:
                time_data = df['TimeMs'].to_numpy() / 1000.0
                ax_time.plot(time_data, loss, label=name, linewidth=2, alpha=0.8)
                ax_time.set_xlabel('Time (s)', fontsize=12)
            
            # --- PLOT 2: ITERAZIONI (Se disponibile) ---
            if 'Iteration' in df.columns:
                iter_data = df['Iteration'].to_numpy()
                ax_iter.plot(iter_data, loss, label=name, linewidth=2, alpha=0.8)
            else:
                iter_data = df.index.to_numpy()
                ax_iter.plot(iter_data, loss, label=name, linewidth=2, alpha=0.8)

            has_data = True
            print(f"-> Aggiunto: {name}")

            # --- PLOT 3: GRADNORM vs ITERATIONS (Se disponibile) ---
            if 'GradNorm' in df.columns:
                grad_norm = df['GradNorm'].to_numpy()
                mask = ~pd.isna(grad_norm)
                if mask.any():
                    ax_grad.plot(iter_data[mask], grad_norm[mask],
                                 label=name, linewidth=2, alpha=0.8)
                    has_grad_data = True

            if 'TrainAcc' in df.columns:
                train_acc = df['TrainAcc'].to_numpy()
                mask = ~pd.isna(train_acc)
                if mask.any():
                    if time_data is not None:
                        ax_acc_time.plot(time_data[mask], train_acc[mask],
                                         label=f"{name} train", linewidth=2, alpha=0.8)
                    ax_acc_iter.plot(iter_data[mask], train_acc[mask],
                                     label=f"{name} train", linewidth=2, alpha=0.8)
                    has_acc_data = True

            if 'TestAcc' in df.columns:
                test_acc = df['TestAcc'].to_numpy()
                mask = ~pd.isna(test_acc)
                if mask.any():
                    if time_data is not None:
                        ax_acc_time.plot(time_data[mask], test_acc[mask],
                                         label=f"{name} test", linewidth=2, alpha=0.8)
                    ax_acc_iter.plot(iter_data[mask], test_acc[mask],
                                     label=f"{name} test", linewidth=2, alpha=0.8)
                    has_acc_data = True

        except Exception as e:
            print(f"Errore nel file {f}: {e}")

    if has_data:
        # --- Configurazione Pannello Sinistro (TEMPO) ---
        ax_time.set_title('Computational Efficiency (Loss vs Time)', fontsize=14)
        if not ax_time.get_xlabel():
            ax_time.set_xlabel('Time (s)', fontsize=12)
        ax_time.set_ylabel('Train Loss (Log Scale)', fontsize=12)
        ax_time.set_yscale('log')
        ax_time.grid(True, which="both", ls="-", alpha=0.4)
        ax_time.legend(fontsize=10, loc='upper right')

        # --- Configurazione Pannello Destro (ITERAZIONI) ---
        ax_iter.set_title('Numerical Convergence (Loss vs Iterations)', fontsize=14)
        ax_iter.set_xlabel('Iterations', fontsize=12)
        ax_iter.set_ylabel('Train Loss (Log Scale)', fontsize=12)
        ax_iter.set_yscale('log')
        ax_iter.grid(True, which="both", ls="-", alpha=0.4)
        ax_iter.legend(fontsize=10, loc='upper right') 
        
        # --- Configurazione Pannello Centrale (GRADNORM) ---
        ax_grad.set_title('GradNorm vs Iterations', fontsize=14)
        ax_grad.set_xlabel('Iterations', fontsize=12)
        ax_grad.set_ylabel('GradNorm', fontsize=12)
        ax_grad.set_yscale('log')
        ax_grad.grid(True, which="both", ls="-", alpha=0.4)
        ax_grad.legend(fontsize=10, loc='upper right')

        # Layout compatto
        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Lascia spazio per il titolo in alto
        
        output_filename = "benchmark_results.png"
        fig.savefig(output_filename, dpi=300)
        print(f"\n[SUCCESSO] Grafico salvato come: {output_filename}")
    else:
        print("Nessun dato valido trovato.")

    if has_acc_data:
        ax_acc_time.set_title('Accuracy (Train/Test) vs Time', fontsize=14)
        if not ax_acc_time.get_xlabel():
            ax_acc_time.set_xlabel('Time (s)', fontsize=12)
        ax_acc_time.set_ylabel('Accuracy (%)', fontsize=12)
        ax_acc_time.grid(True, which="both", ls="-", alpha=0.4)
        ax_acc_time.legend(fontsize=9, loc='lower right')

        ax_acc_iter.set_title('Accuracy (Train/Test) vs Iterations', fontsize=14)
        ax_acc_iter.set_xlabel('Iterations', fontsize=12)
        ax_acc_iter.set_ylabel('Accuracy (%)', fontsize=12)
        ax_acc_iter.grid(True, which="both", ls="-", alpha=0.4)
        ax_acc_iter.legend(fontsize=9, loc='lower right')

        fig_acc.tight_layout(rect=[0, 0.03, 1, 0.95])

        output_acc_filename = "accuracy_results.png"
        fig_acc.savefig(output_acc_filename, dpi=300)
        print(f"\n[SUCCESSO] Grafico salvato come: {output_acc_filename}")
    else:
        plt.close(fig_acc)

    plt.close('all')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot training history results.")
    parser.add_argument("--title", type=str, default=None, help="Optional figure title.")
    args = parser.parse_args()
    plot_combined(title=args.title)
