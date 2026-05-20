import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
import scipy.stats as stats
import time
from numba import njit, prange

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

def genera_cielo_isotropo_pesato(expo_map, n_eventi):
    """
    Genera un set di eventi (RA, Dec) distribuiti in modo isotropo, 
    estratti proporzionalmente al 'peso' (esposizione) di ciascun pixel.
    """
    expo_pulita = np.where(expo_map > 0, expo_map, 0)
    probabilita_pixel = expo_pulita / np.sum(expo_pulita)
    
    n_pixel_totali = len(probabilita_pixel)
    
    indici_estratti = np.random.choice(n_pixel_totali, size=n_eventi, p=probabilita_pixel)
    
    nside = hp.npix2nside(n_pixel_totali)
    theta, phi = hp.pix2ang(nside, indici_estratti)
    
    ra = np.degrees(phi)
    dec = 90.0 - np.degrees(theta)
    
    return ra, dec

def prepara_dati_numba(expo_map):
    """Precalcola la CDF dell'esposizione e le coordinate dei pixel in radianti."""
    expo_pulita = np.where(expo_map > 0, expo_map, 0)
    probabilita_pixel = expo_pulita / np.sum(expo_pulita)
    
    # Cumulative Distribution Function per l'estrazione veloce via Numba
    pixel_cdf = np.cumsum(probabilita_pixel)
    
    n_pixel_totali = len(probabilita_pixel)
    nside = hp.npix2nside(n_pixel_totali)
    theta, phi = hp.pix2ang(nside, np.arange(n_pixel_totali))
    
    # Convertiamo tutto in radianti per velocizzare Numba
    pixel_ra_rad = phi
    pixel_dec_rad = (np.pi / 2.0) - theta
    
    return pixel_cdf, pixel_ra_rad, pixel_dec_rad

@njit(parallel=True, fastmath=True)
def esegui_simulazioni_numba_multi(n_simulazioni, n_eventi, pixel_cdf, pixel_ra_rad, pixel_dec_rad, srcs_ra, srcs_dec, max_raggio):
    """Motore Monte Carlo ultra-veloce scritto in Numba."""
    n_src = len(srcs_ra)
    # Matrice per salvare il risultato: (sorgente, id_simulazione, raggio)
    conteggi_simulati = np.zeros((n_src, n_simulazioni, max_raggio), dtype=np.int32)
    
    srcs_ra_rad = np.radians(srcs_ra)
    srcs_dec_rad = np.radians(srcs_dec)
    
    # prange abilita il multithreading su tutti i core della CPU
    for i in prange(n_simulazioni):
        # Generiamo il cielo per questa iterazione simulata
        for j in range(n_eventi):
            # 1. Estrazione pixel random (Inverse Transform Sampling sulla CDF)
            u = np.random.rand()
            pix_idx = np.searchsorted(pixel_cdf, u)
            
            ra_rad = pixel_ra_rad[pix_idx]
            dec_rad = pixel_dec_rad[pix_idx]
            
            # 2. Controllo contro ogni sorgente
            for k in range(n_src):
                cos_theta = (np.sin(dec_rad) * np.sin(srcs_dec_rad[k]) + 
                             np.cos(dec_rad) * np.cos(srcs_dec_rad[k]) * np.cos(ra_rad - srcs_ra_rad[k]))
                
                if cos_theta < -1.0: cos_theta = -1.0
                elif cos_theta > 1.0: cos_theta = 1.0
                
                dist = np.degrees(np.arccos(cos_theta))
                
                # 3. Binning istantaneo nei raggi top-hat (1°...max_raggio)
                start_idx = int(np.ceil(dist)) - 1
                if start_idx < 0:
                    start_idx = 0
                
                # Incrementa i contatori solo per i raggi in cui l'evento cade
                for r_idx in range(start_idx, max_raggio):
                    conteggi_simulati[k, i, r_idx] += 1
                    
    return conteggi_simulati

def test_statistico_montecarlo_globale(df, expo_map, sorgenti_dict, max_raggio=40, n_simulazioni=10000):
    """
    Esegue l'analisi Monte Carlo parallelizzata su CPU tramite Numba,
    suddividendola in blocchi per mostrare la progressione a terminale.
    """
    N_totali = len(df)
    raggi = np.arange(1, max_raggio + 1) 
    risultati = {}
    
    nomi_sorgenti = list(sorgenti_dict.keys())
    srcs_ra = np.array([sorgenti_dict[n]['RA'] for n in nomi_sorgenti])
    srcs_dec = np.array([sorgenti_dict[n]['Dec'] for n in nomi_sorgenti])
    
    print(f"   -> Calcolo eventi reali osservati per raggi da 1° a {max_raggio}°...")
    for i, nome in enumerate(nomi_sorgenti):
        dist_reale = calcola_distanza_angolare(df['RA'], df['Dec'], srcs_ra[i], srcs_dec[i])
        dist_reale_sorted = np.sort(dist_reale) 
        n_oss_array = np.searchsorted(dist_reale_sorted, raggi, side='right')
        
        risultati[nome] = {
            'raggi': raggi,
            'osservati_array': n_oss_array
        }

    print("   -> Preparazione mappe CDF per il motore compilato Numba...")
    pixel_cdf, pixel_ra_rad, pixel_dec_rad = prepara_dati_numba(expo_map)

    # --- STRATEGIA DI PROGRESSIONE A BLOCCHI ---
    n_blocchi = 10  # Suddividiamo la barra di avanzamento in 10 step (es. 10%, 20%...)
    if n_simulazioni < n_blocchi:
        n_blocchi = 1
        
    sim_per_blocco = n_simulazioni // n_blocchi
    residuo = n_simulazioni % n_blocchi

    n_src = len(srcs_ra)
    # Allochiamo in anticipo l'intera matrice tridimensionale per contenere tutti i risultati
    conteggi_sim_all = np.zeros((n_src, n_simulazioni, max_raggio), dtype=np.int32)

    print(f"   -> Avvio {n_simulazioni} simulazioni Monte Carlo divise in {n_blocchi} blocchi...")
    start_sim_time = time.perf_counter()
    
    idx_corrente = 0
    for b in range(n_blocchi):
        # Gestiamo l'eventuale resto se n_simulazioni non è divisibile perfettamente per n_blocchi
        corrente_sim_blocco = sim_per_blocco + (1 if b < residuo else 0)
        end_idx = idx_corrente + corrente_sim_blocco
        
        # Chiamata al motore Numba per il blocco corrente
        conteggi_chunk = esegui_simulazioni_numba_multi(
            corrente_sim_blocco, N_totali, pixel_cdf, pixel_ra_rad, pixel_dec_rad, 
            srcs_ra, srcs_dec, max_raggio
        )
        
        # Iniettiamo i risultati del chunk nella matrice globale nella giusta posizione
        conteggi_sim_all[:, idx_corrente:end_idx, :] = conteggi_chunk
        idx_corrente = end_idx
        
        # Stampiamo il progresso a schermo
        percentuale = ((b + 1) / n_blocchi) * 100
        tempo_parziale = time.perf_counter() - start_sim_time
        print(f"      [{percentuale:>3.0f}%] Completati {idx_corrente:,} cieli su {n_simulazioni:,} ({tempo_parziale:.1f}s trascorsi)")

    end_sim_time = time.perf_counter()
    tempo_simulazione = end_sim_time - start_sim_time
    print(f"   -> Calcolo parallelo completato con successo in {tempo_simulazione:.2f} secondi.")
    
    # --- ELABORAZIONE STATISTICA FINALE (Invariata) ---
    for k, nome in enumerate(nomi_sorgenti):
        risultati[nome]['tempo_simulazione'] = tempo_simulazione
        n_oss = risultati[nome]['osservati_array']
        
        conteggi_sim = conteggi_sim_all[k]
        
        n_attesi = np.mean(conteggi_sim, axis=0)
        simulazioni_superiori = np.sum(conteggi_sim >= n_oss, axis=0)
        p_values = simulazioni_superiori / n_simulazioni
            
        sigmas = stats.norm.isf(p_values)
        sigmas[np.isinf(sigmas) | (sigmas < 0)] = 0.0
        
        idx_min = np.argmin(p_values)
        
        risultati[nome].update({
            'conteggi_simulati_array': conteggi_sim,
            'attesi_array': n_attesi,
            'p_values_array': p_values,
            'sigmas_array': sigmas,
            'raggio_minimo': raggi[idx_min],
            'p_value_minimo': p_values[idx_min],
            'sigma_massimo': sigmas[idx_min]
        })
        
    return risultati

def plot_pvalue_scan(risultati_scan, base_dir='plots'):
    """Genera e salva i grafici del P-Value in funzione del raggio per ogni sorgente."""
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
        plt.scatter([r_min], [p_min], color='red', s=100, zorder=5, label=f'Minimo a {r_min}°')
        
        plt.legend(loc='upper left')
        plt.grid(True, which='both', linestyle='--', alpha=0.5)
        
        nome_file = f"scan_{nome.replace(' ', '_').replace('*', '')}.png"
        plt.savefig(os.path.join(out_dir, nome_file))
        plt.close()
    
    print(f"Grafici di scan salvati nella cartella: {out_dir}")

def analisi_tophat_sorgente(df, nome_sorgente, ra_src, dec_src, max_raggio=40, base_dir='plots'):
    """Esegue l'analisi Top Hat per una singola sorgente e salva le mappe."""
    print(f"\n---> Generazione mappe Top Hat per: {nome_sorgente}")
    
    nome_cartella = nome_sorgente.replace(" ", "_").replace("*", "")
    out_dir = os.path.join(base_dir, 'tophat_maps', nome_cartella)
    os.makedirs(out_dir, exist_ok=True)
    
    distanza = calcola_distanza_angolare(df['RA'], df['Dec'], ra_src, dec_src)
    
    raggio = 0.0
    for i in range(max_raggio):
        raggio += 1.0
        df_selezionato = df[distanza <= raggio]
        
        plt.figure(figsize=(10, 6))
        plt.scatter(df['RA'], df['Dec'], color='gray', alpha=0.3, label='Tutti gli eventi', s=10)
        plt.scatter(df_selezionato['RA'], df_selezionato['Dec'], color='red', alpha=0.8, label=f'Eventi (R<={raggio}°)', s=20)
        plt.scatter(ra_src, dec_src, color='black', marker='*', s=200, label=nome_sorgente)

        plt.title(f'Top Hat (R={raggio}°) - {nome_sorgente}')
        plt.xlabel('Ascensione Retta (gradi)')
        plt.ylabel('Declinazione (gradi)')
        plt.xlim(0, 360)
        plt.ylim(-90, 90)
        plt.legend(loc='upper right')
        plt.grid(True, linestyle='--', alpha=0.6)
        
        plt.savefig(os.path.join(out_dir, f"map_R{int(raggio)}.png"))
        plt.close()

def mappa_calore_globale(df, dizionario_sorgenti, base_dir='plots'):
    """Crea la heatmap dell'energia e ci posiziona sopra TUTTE le sorgenti analizzate."""
    print("\n---> Generazione mappa celeste dell'energia globale in corso...")
    os.makedirs(base_dir, exist_ok=True)
    
    plt.figure(figsize=(12, 7))
    mappa_colori = plt.scatter(df['RA'], df['Dec'], c=df['E'], cmap='plasma', alpha=0.8, s=25)
    
    cbar = plt.colorbar(mappa_colori)
    cbar.set_label('Energia [EeV]', fontsize=12)

    colori_marker = ['black', 'red', 'cyan', 'magenta', 'yellow', 'lime', 'orange', 'green', 'blue']
    for idx, (nome, coordinate) in enumerate(dizionario_sorgenti.items()):
        colore = colori_marker[idx % len(colori_marker)]
        plt.scatter(coordinate['RA'], coordinate['Dec'], color=colore, marker='*', 
                    s=300, label=nome, edgecolors='black')

    plt.title("Mappa Celeste degli Eventi Auger (Intensità per Energia)", fontsize=14)
    plt.xlabel('Ascensione Retta (gradi)')
    plt.ylabel('Declinazione (gradi)')
    plt.xlim(0, 360)
    plt.ylim(-90, 90)
    
    plt.legend(loc='upper right', bbox_to_anchor=(1.35, 1.0)) 
    plt.grid(True, linestyle='--', alpha=0.5)
    
    heatmap_path = os.path.join(base_dir, 'sky_map_energy_heatmap.png')
    plt.savefig(heatmap_path, bbox_inches='tight') 
    plt.close()
    print(f"Mappa dell'energia globale salvata in: {heatmap_path}")

def mappa_eventi_casuali(expo_map, dizionario_sorgenti, n_eventi=2635, base_dir='plots'):
    """
    Genera UN cielo finto usando la mappa di esposizione e lo disegna su 
    una mappa sferica Mollweide, aggiungendo le sorgenti come riferimento.
    """
    print(f"\n---> Generazione mappa di check: Cielo finto ({n_eventi} eventi) su Esposizione...")
    os.makedirs(base_dir, exist_ok=True)
    
    ra, dec = genera_cielo_isotropo_pesato(expo_map, n_eventi)
    
    plt.figure(figsize=(12, 7))
    
    hp.mollview(expo_map, hold=True, title=f"Cielo Finto Monte Carlo ({n_eventi} eventi)", 
                cmap='viridis', unit='Esposizione Relativa', coord=['C'])
    hp.graticule()

    hp.projscatter(ra, dec, lonlat=True, coord='C', 
                   marker='o', facecolors='white', edgecolors='black', linewidths=0.5,
                   s=30, alpha=0.8, label='Eventi Simulati')

    colori_marker = ['black', 'red', 'cyan', 'magenta', 'yellow', 'lime', 'orange', 'green', 'blue']
    for idx, (nome, coordinate) in enumerate(dizionario_sorgenti.items()):
        colore = colori_marker[idx % len(colori_marker)]
        hp.projscatter(coordinate['RA'], coordinate['Dec'], lonlat=True, coord='C',
                       color=colore, marker='*', s=300, edgecolors='white', label=nome)

    plt.legend(loc='upper right', bbox_to_anchor=(1.25, 1.0))
    
    map_path = os.path.join(base_dir, 'check_montecarlo_cielo_finto.png')
    plt.savefig(map_path, bbox_inches='tight')
    plt.close()
    
    print(f"Mappa di check del cielo finto salvata in: {map_path}")

if __name__ == "__main__":
    
    # DIZIONARIO SPECIALIZZATO SOLO PER CENTAURUS A
    sorgenti_da_analizzare = {
        "Centaurus A": {"RA": 201.3, "Dec": -43.0}
    }
    
    cartella_output = 'plotsNGC5128'
    file_dati = 'auger.txt'
    file_esposizione = 'exposure.fits'
    
    raggio_di_ricerca = 15.0  
    NUM_SIMULAZIONI = 20000000
    
    os.makedirs(cartella_output, exist_ok=True)
    start_total_time = time.perf_counter()
    
    dataset = carica_dati(file_dati)
    print(f"Dati caricati: {len(dataset)} eventi trovati.")
    
    try:
        expo_map = hp.read_map(file_esposizione)
        nside = hp.get_nside(expo_map)
        print(f"Mappa Healpix caricata con successo (NSIDE={nside}).")
    except Exception as e:
        print(f"ERRORE CRITICO: Impossibile caricare {file_esposizione}. Assicurati che sia nella cartella.")
        exit()
        
    print(f"\n---> Inizio Analisi Statistica Globale (Scan 1°-40°)...")
    risultati_statistici = test_statistico_montecarlo_globale(
        df=dataset, 
        expo_map=expo_map, 
        sorgenti_dict=sorgenti_da_analizzare, 
        max_raggio=40,
        n_simulazioni=NUM_SIMULAZIONI
    )
    
    plot_pvalue_scan(risultati_statistici, base_dir=cartella_output)
        
    percorso_report = os.path.join(cartella_output, 'report_statistico_SCAN_LOCALE_DETTAGLIATO.txt')
    
    with open(percorso_report, 'w') as f_out:
        f_out.write("=========================================================================\n")
        f_out.write(f"  REPORT STATISTICO DETTAGLIATO PER OGNI RAGGIO\n")
        f_out.write(f"  Metodo: Monte Carlo ({NUM_SIMULAZIONI} cataloghi simulati)\n")
        f_out.write("=========================================================================\n\n")
    
        for nome, dati in risultati_statistici.items():
            f_out.write(f"SORGENTE: {nome}\n")
            f_out.write("-" * 80 + "\n")
            f_out.write(f"{'Raggio':<8} | {'Osservati':<10} | {'Attesi (Medio)':<15} | {'P-Value':<12} | {'Sigma'}\n")
            f_out.write("-" * 80 + "\n")
            
            for i, r in enumerate(dati['raggi']):
                oss = dati['osservati_array'][i]
                att = dati['attesi_array'][i]
                pval = dati['p_values_array'][i]
                sigma = dati['sigmas_array'][i]
                
                # Gestione del caso in cui zero simulazioni superano il dato reale
                if pval == 0.0:
                    pval_str = f"<{1/NUM_SIMULAZIONI:.1e}"
                    min_sigma = stats.norm.isf(1/NUM_SIMULAZIONI)
                    sigma_str = f">{min_sigma:.2f}"
                else:
                    pval_str = f"{pval:.3e}"
                    sigma_str = f"{sigma:.2f}"
                
                marker = "  <--- MASSIMO ECCESSO LOCALE" if r == dati['raggio_minimo'] else ""
                
                riga_report = f"{r:>3}°     | {oss:<10} | {att:<15.2f} | {pval_str:<12} | {sigma_str:<10} {marker}\n"
                f_out.write(riga_report)
                
            f_out.write("=========================================================================\n\n")
            
    print(f"I risultati statistici dettagliati grado per grado sono in: {percorso_report}")
                
    mappa_calore_globale(dataset, sorgenti_da_analizzare, cartella_output)
    
    mappa_eventi_casuali(
        expo_map=expo_map, 
        dizionario_sorgenti=sorgenti_da_analizzare, 
        n_eventi=len(dataset), 
        base_dir=cartella_output
    )
    
    end_total_time = time.perf_counter()
    tempo_totale = end_total_time - start_total_time
    tempo_simulazione = next(iter(risultati_statistici.values()))['tempo_simulazione']

    print("\n---------------------------------------------------")
    print("TUTTE LE ANALISI SONO STATE COMPLETATE CON SUCCESSO!")
    print(f"Tempo di simulazione Monte Carlo: {tempo_simulazione:.2f} secondi")
    print(f"Tempo totale esecuzione script: {tempo_totale:.2f} secondi")