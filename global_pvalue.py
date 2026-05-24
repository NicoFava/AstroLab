import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os

def plotta_progressione_definitiva(file_log='penalizzazione_log.txt'):
    if not os.path.exists(file_log):
        print(f"Errore: {file_log} non trovato.")
        return

    data = []
    with open(file_log, 'r') as f:
        for line in f:
            if line.startswith("Batch_ID"): continue
            try:
                parts = line.strip().split('\t')
                if len(parts) >= 4:
                    data.append([float(parts[1]), float(parts[3])])
            except ValueError:
                continue
    
    df = pd.DataFrame(data, columns=["Cieli_Batch", "Superamenti_Batch"])
    df['Cieli_Cumulati'] = df['Cieli_Batch'].cumsum()
    df['Superamenti_Cumulativi'] = df['Superamenti_Batch'].cumsum()
    
    # Calcolo della proporzione (P-Value) e dell'errore statistico binomiale (1 sigma)
    df['P_Value_Globale'] = df['Superamenti_Cumulativi'] / df['Cieli_Cumulati']
    df['Errore_Sigma'] = np.sqrt((df['P_Value_Globale'] * (1 - df['P_Value_Globale'])) / df['Cieli_Cumulati'])

    # --- INIZIO STYLING MIGLIORATO ---
    plt.style.use('seaborn-v0_8-whitegrid') 
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = df['Cieli_Cumulati']
    y = df['P_Value_Globale']
    err = df['Errore_Sigma']

    # Banda di errore sfumata
    ax.fill_between(x, y - err, y + err, color='#3498db', alpha=0.3, label=r'Incertezza Statistica ($1\sigma$)')
    
    # Linea principale del p-value
    ax.plot(x, y, color='#2c3e50', linewidth=2, label='P-Value Globale Cumulativo')
    
    # Linea orizzontale di riferimento per il valore finale
    p_final = y.iloc[-1]
    ax.axhline(y=p_final, color='#e74c3c', linestyle='--', linewidth=1.5, label=f'Convergenza: {p_final:.2e}')
    
    # Formattazione Asse Y
    ax.set_yscale('log')
    ax.set_ylabel('P-Value Globale', fontsize=13, fontweight='bold', color='#333333')
    
    # Formattazione Asse X (Numeri in Milioni)
    ax.set_xlabel('Numero Totale di Simulazioni', fontsize=13, fontweight='bold', color='#333333')
    formatter = ticker.FuncFormatter(lambda val, pos: f'{val*1e-6:g}M' if val >= 1e6 else f'{val:g}')
    ax.xaxis.set_major_formatter(formatter)
    
    # Titolo e Griglia
    ax.set_title(f'Convergenza P-Value Globale su {int(x.iloc[-1]):,} Trial', fontsize=15, fontweight='bold', pad=15)
    ax.grid(True, which="both", ls=":", color='#bdc3c7', alpha=0.8)
    
    # Legenda personalizzata
    ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True, borderpad=1)
    
    plt.tight_layout()
    plt.savefig('progressione_definitiva_v2.png', dpi=300, bbox_inches='tight')
    print(f"Grafico aggiornato e migliorato. P-Value finale: {p_final:.3e}")

if __name__ == "__main__":
    plotta_progressione_definitiva()