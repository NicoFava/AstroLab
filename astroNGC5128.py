import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
import scipy.stats as stats

def carica_dati(filepath):
    """Carica e pulisce il dataset di Auger."""
    df = pd.read_csv(filepath, sep='\t+', engine='python')
    df.columns = [c.strip().replace('#', '') for c in df.columns]
    return df

def calcola_distanza_angolare(ra1, dec1, ra2, dec2):
    """Calcola la distanza angolare sferica tra due punti in gradi."""
    ra1_rad, dec1_rad = np.radians(ra1), np.radians(dec1)
    ra2_rad, dec2_rad = np.radians(ra2), np.radians(dec2)
    
    cos_theta = (np.sin(dec1_rad) * np.sin(dec2_rad) + 
                 np.cos(dec1_rad) * np.cos(dec2_rad) * np.cos(ra1_rad - ra2_rad))
    cos_theta = np.clip(cos_theta, -1.0, 1.0) 
    return np.degrees(np.arccos(cos_theta))

def test_statistico_centaurus_ottimizzato(df, expo_map, coord_src, max_raggio=40, n_simulazioni=1000000):
    """
    Esegue un Monte Carlo vettorializzato ad altissime prestazioni per una singola sorgente.
    Usa il pre-calcolo delle distanze sui pixel Healpix per elaborare milioni di cieli in batch.
    """
    N_totali = len(df)
    raggi = np.arange(1, max_raggio + 1)
    
    print(f"   -> Calcolo eventi reali osservati per raggi da 1° a {max_raggio}°...")
    dist_reale = calcola_distanza_angolare(df['RA'], df['Dec'], coord_src['RA'], coord_src['Dec'])
    dist_reale_sorted = np.sort(dist_reale)
    n_oss_array = np.searchsorted(dist_reale_sorted, raggi, side='right')

    print("   -> Pre-calcolo delle distanze Healpix in corso (Ottimizzazione MC)...")
    # 1. Calcolo probabilità dei pixel
    expo_pulita = np.where(expo_map > 0, expo_map, 0)
    prob_pixel = expo_pulita / np.sum(expo_pulita)
    n_pixel_totali = len(prob_pixel)
    
    # 2. Ottenimento coordinate per tutti i pixel della mappa
    nside = hp.npix2nside(n_pixel_totali)
    theta_pix, phi_pix = hp.pix2ang(nside, np.arange(n_pixel_totali))
    ra_pix = np.degrees(phi_pix)
    dec_pix = 90.0 - np.degrees(theta_pix)
    
    # 3. Pre-calcolo della distanza di ogni pixel da Centaurus A
    distanze_pixel = calcola_distanza_angolare(ra_pix, dec_pix, coord_src['RA'], coord_src['Dec'])
    
    conteggi_simulati_array = np.zeros((n_simulazioni, max_raggio))
    
    print(f"   -> Avvio simulazioni vettorializzate ({n_simulazioni} cieli totali)...")
    
    # Avvio del timer specifico per le simulazioni
    start_sim_time = time.perf_counter()
    
    # Elaboriamo i cieli a blocchi (batch) per massimizzare la velocità senza saturare la RAM
    batch_size = 10000 
    for i in range(0, n_simulazioni, batch_size):
        current_batch = min(batch_size, n_simulazioni - i)
        
        # Estrazione pesata di NxM indici pixel (forma: [batch_size, N_totali])
        pixel_estratti = np.random.choice(n_pixel_totali, size=(current_batch, N_totali), p=prob_pixel)
        
        # Mappiamo istantaneamente gli indici alle loro distanze dalla sorgente
        distanze_sim = distanze_pixel[pixel_estratti]
        
        # Contiamo quanti eventi cadono entro i raggi per l'intero batch
        for r_idx, r in enumerate(raggi):
            conteggi_simulati_array[i:i+current_batch, r_idx] = np.sum(distanze_sim <= r, axis=1)
            
        if (i + current_batch) % 100000 == 0:
            print(f"      ... Calcolati {i + current_batch}/{n_simulazioni} cieli finti")

    # Fine del timer delle simulazioni
    end_sim_time = time.perf_counter()
    tempo_simulazione = end_sim_time - start_sim_time
    print(f"   -> Simulazioni completate in {tempo_simulazione:.2f} secondi.")

    print("   -> Calcolo della significatività completato.")
    
    n_attesi = np.mean(conteggi_simulati_array, axis=0)
    simulazioni_superiori = np.sum(conteggi_simulati_array >= n_oss_array, axis=0)
    
    # Formula empirica corretta per il P-Value: (N_sup + 1) / (N_sim + 1)
    p_values = (simulazioni_superiori) / (n_simulazioni)
    
    sigmas = stats.norm.isf(p_values)
    sigmas[np.isinf(sigmas) | (sigmas < 0)] = 0.0
    
    idx_min = np.argmin(p_values)
    
    return {
        'Centaurus A': {
            'raggi': raggi,
            'osservati_array': n_oss_array,
            'attesi_array': n_attesi,
            'p_values_array': p_values,
            'sigmas_array': sigmas,
            'raggio_minimo': raggi[idx_min],
            'p_value_minimo': p_values[idx_min],
            'sigma_massimo': sigmas[idx_min],
            'tempo_simulazione': tempo_simulazione
        }
    }

def plot_pvalue_scan(risultati_scan, base_dir='plots'):
    """Genera e salva i grafici del P-Value in funzione del raggio."""
    print("\n---> Generazione dei grafici di scan del P-Value...")
    out_dir = os.path.join(base_dir, 'pvalue_scans')
    os.makedirs(out_dir, exist_ok=True)
    
    for nome, dati in risultati_scan.items():
        plt.figure(figsize=(10, 6))
        raggi = dati['raggi']
        p_values = dati['p_values_array']
        
        plt.plot(raggi, p_values, marker='o', linestyle='-', color='blue', alpha=0.8)
        plt.axhline(y=0.0013, color='red', linestyle='--', label='Soglia 3 Sigma')
        
        plt.yscale('log')
        plt.xlabel('Raggio Top Hat (°)', fontsize=12)
        plt.ylabel('P-Value Locale (Monte Carlo)', fontsize=12)
        plt.title(f'Andamento P-Value in funzione del raggio: {nome}', fontsize=14)
        
        r_min = dati['raggio_minimo']
        p_min = dati['p_value_minimo']
        plt.scatter([r_min], [p_min], color='red', s=100, zorder=5, label=f'Minimo a {r_min}°\n({dati["sigma_massimo"]:.2f} $\sigma$)')
        
        plt.legend(loc='upper left')
        plt.grid(True, which='both', linestyle='--', alpha=0.5)
        
        nome_file = f"scan_{nome.replace(' ', '_')}.png"
        plt.savefig(os.path.join(out_dir, nome_file))
        plt.close()

if __name__ == "__main__":
    
    # Avvio del timer totale del programma
    start_total_time = time.perf_counter()
    
    # Fokus esclusivo su Centaurus A
    sorgente_target = {"RA": 201.3, "Dec": -43.0}
    nome_sorgente = "Centaurus A"
    
    cartella_output = 'plots7'
    file_dati = 'auger.txt'
    file_esposizione = 'exposure.fits'
    
    # Adesso possiamo spingere davvero il numero di simulazioni
    NUM_SIMULAZIONI = 10000000
    
    os.makedirs(cartella_output, exist_ok=True)
    
    dataset = carica_dati(file_dati)
    print(f"Dati caricati: {len(dataset)} eventi trovati.")
    
    try:
        expo_map = hp.read_map(file_esposizione)
    except Exception as e:
        print(f"ERRORE CRITICO: Impossibile caricare {file_esposizione}.")
        exit()
        
    print(f"\n---> Inizio Analisi Statistica su {nome_sorgente} (Scan 1°-40°)...")
    risultati = test_statistico_centaurus_ottimizzato(
        df=dataset, 
        expo_map=expo_map, 
        coord_src=sorgente_target, 
        max_raggio=40,
        n_simulazioni=NUM_SIMULAZIONI
    )
    
    plot_pvalue_scan(risultati, base_dir=cartella_output)
    
    # Fine del timer totale
    end_total_time = time.perf_counter()
    tempo_totale = end_total_time - start_total_time
        
    percorso_report = os.path.join(cartella_output, f'report_{nome_sorgente.replace(" ", "_")}.txt')
    
    with open(percorso_report, 'w') as f_out:
        dati = risultati[nome_sorgente]
        f_out.write("=========================================================================\n")
        f_out.write(f"  REPORT STATISTICO: {nome_sorgente}\n")
        f_out.write(f"  Metodo: Monte Carlo Vettorializzato ({NUM_SIMULAZIONI} iterazioni)\n")
        f_out.write(f"  Tempo di calcolo MC:  {dati['tempo_simulazione']:.2f} secondi\n")
        f_out.write(f"  Tempo totale script:  {tempo_totale:.2f} secondi\n")
        f_out.write("=========================================================================\n")
        f_out.write(f"  Raggio Ottimale:      {dati['raggio_minimo']}°\n")
        f_out.write(f"  Significatività (Loc): {dati['sigma_massimo']:.3f} sigma\n")
        f_out.write(f"  P-Value (Loc):         {dati['p_value_minimo']:.3e}\n")
        f_out.write("=========================================================================\n")
        f_out.write("\nDettaglio per Raggio:\n")
        for r, oss, att, p, s in zip(dati['raggi'], dati['osservati_array'], dati['attesi_array'], dati['p_values_array'], dati['sigmas_array']):
            f_out.write(f" R={r:>2}° | Oss: {oss:>3} | Attesi: {att:>6.2f} | P-val: {p:.2e} | Sig: {s:.2f}\n")
            
    print("\n---------------------------------------------------")
    print("ANALISI COMPLETATA CON SUCCESSO!")
    print(f"Tempo di esecuzione MC: {risultati[nome_sorgente]['tempo_simulazione']:.2f} secondi")
    print(f"Tempo totale esecuzione: {tempo_totale:.2f} secondi ({tempo_totale/60:.2f} minuti)")
    print(f"Controlla il plot in: {cartella_output}/pvalue_scans/ e il report in {percorso_report}")