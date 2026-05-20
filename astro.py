import os
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
        # plt.gca().invert_xaxis() 
        plt.legend(loc='upper right')
        plt.grid(True, linestyle='--', alpha=0.6)
        
        plt.savefig(os.path.join(out_dir, f"map_R{int(raggio)}.png"))
        plt.close()

def test_statistico_healpy(df, expo_map, nside, ra_src, dec_src, raggio_test=15.0):
    """
    Calcola la significatività statistica di un eccesso in una regione circolare 
    usando la mappa di esposizione reale di Auger via healpy.
    """
    N_totali = len(df)
    
    # 1. Eventi Osservati
    distanza = calcola_distanza_angolare(df['RA'], df['Dec'], ra_src, dec_src)
    N_osservati = len(df[distanza <= raggio_test])
    
    # 2. Eventi Attesi (tramite Healpy)
    theta_src = np.radians(90.0 - dec_src) # Healpy usa la colatitudine
    phi_src = np.radians(ra_src)
    vettore_sorgente = hp.ang2vec(theta_src, phi_src)
    
    pixel_nella_tophat = hp.query_disc(nside, vettore_sorgente, np.radians(raggio_test))
    
    expo_totale = np.sum(expo_map)
    expo_tophat = np.sum(expo_map[pixel_nella_tophat])
    frazione_esposizione = expo_tophat / expo_totale
    
    N_attesi = N_totali * frazione_esposizione
    
    # 3. Statistica
    if N_osservati > 0:
        p_value = stats.poisson.sf(N_osservati - 1, N_attesi)
    else:
        p_value = 1.0 # Se non ho eventi, il p-value è massimo
        
    sigma = stats.norm.isf(p_value)
    
    # Gestione di casi estremi (sigma infinito se p_value è vicinissimo a 0)
    if np.isinf(sigma) or sigma < 0: 
        sigma = 0.0
        
    return N_osservati, N_attesi, p_value, sigma

def mappa_calore_globale(df, dizionario_sorgenti, base_dir='plots'):
    """Crea la heatmap dell'energia e ci posiziona sopra TUTTE le sorgenti analizzate."""
    print("\n---> Generazione mappa celeste dell'energia in corso...")
    os.makedirs(base_dir, exist_ok=True)
    
    plt.figure(figsize=(12, 7))
    mappa_colori = plt.scatter(df['RA'], df['Dec'], c=df['E'], cmap='plasma', alpha=0.8, s=25)
    
    cbar = plt.colorbar(mappa_colori)
    cbar.set_label('Energia [EeV]', fontsize=12)

    colori_marker = ['black', 'lime', 'cyan', 'white']
    for idx, (nome, coordinate) in enumerate(dizionario_sorgenti.items()):
        colore = colori_marker[idx % len(colori_marker)]
        plt.scatter(coordinate['RA'], coordinate['Dec'], color=colore, marker='*', 
                    s=300, label=nome, edgecolors='black')

    plt.title("Mappa Celeste degli Eventi Auger (Intensità per Energia)", fontsize=14)
    plt.xlabel('Ascensione Retta (gradi)')
    plt.ylabel('Declinazione (gradi)')
    plt.xlim(0, 360)
    plt.ylim(-90, 90)
    # plt.gca().invert_xaxis() 
    
    plt.legend(loc='upper right', bbox_to_anchor=(1.35, 1.0)) # Sposta leggermente la legenda
    plt.grid(True, linestyle='--', alpha=0.5)
    
    heatmap_path = os.path.join(base_dir, 'sky_map_energy_heatmap.png')
    plt.savefig(heatmap_path, bbox_inches='tight') # Evita che la legenda venga tagliata
    plt.close()

if __name__ == "__main__":
    
    sorgenti_da_analizzare = {
        "Sagittarius A*": {"RA": 266.4168, "Dec": -29.0078},
        "Centaurus A (NGC 5128)":    {"RA": 201.3,    "Dec": -43.0},
        "Fornax A (NGC 1316)":       {"RA": 50.67,    "Dec": -37.2},
        "NGC 253 (Galassia dello Scultore)":        {"RA": 11.89,    "Dec": -25.29},
        "M83 (Galassia Girandola del Sud)":            {"RA": 253.47,   "Dec": -24.38},
        "M87 (Virgo A)":  {"RA": 187.7,    "Dec": 12.39},
        "Pulsar delle Vele (Vela SNR)":       {"RA": 128.4,    "Dec": -45.18},
        "Grande Nube di Magellano (LMC)":            {"RA": 80.89,    "Dec": -69.76},
        "Piccola Nube di Magellano (SMC)":            {"RA": 13.16,    "Dec": -72.8}
    }
    
    cartella_output = 'plots'
    file_dati = 'auger.txt'
    file_esposizione = 'exposure.fits'
    raggio_di_ricerca = 15.0 # Raggio fisso a cui calcolare la statistica (es. 15 gradi)
    
    os.makedirs(cartella_output, exist_ok=True)
    
    # 1. Caricamento Dati
    dataset = carica_dati(file_dati)
    print(f"Dati caricati: {len(dataset)} eventi trovati.")
    
    # 2. Caricamento Mappa di Esposizione Healpy
    try:
        expo_map = hp.read_map(file_esposizione)
        nside = hp.get_nside(expo_map)
        print(f"Mappa Healpix '{file_esposizione}' caricata con successo (NSIDE={nside}).")
    except Exception as e:
        print(f"ERRORE: Impossibile caricare {file_esposizione}. Assicurati che sia nella cartella.")
        exit()
        
    # Preparazione del file di testo per salvare i risultati statistici
    percorso_report = os.path.join(cartella_output, 'report_statistico.txt')
    
    with open(percorso_report, 'w') as f_out:
        f_out.write("=========================================================\n")
        f_out.write(f"  REPORT STATISTICO SORGENTI (Raggio Top Hat: {raggio_di_ricerca}°)\n")
        f_out.write("=========================================================\n")
        f_out.write(f"{'Sorgente':<20} | {'Osservati':<10} | {'Attesi (Fondo)':<15} | {'Sigma':<8} | {'P-Value'}\n")
        f_out.write("-" * 80 + "\n")
    
        # 3. Analisi per ogni sorgente
        for nome, coordinate in sorgenti_da_analizzare.items():
            
            # A. Genera i plot (fino a R=40)
            analisi_tophat_sorgente(
                df=dataset, 
                nome_sorgente=nome, 
                ra_src=coordinate['RA'], 
                dec_src=coordinate['Dec'], 
                max_raggio=40, 
                base_dir=cartella_output
            )
            
            # B. Calcolo Statistico a raggio fisso (es. 15 gradi)
            N_oss, N_att, p_val, sigma = test_statistico_healpy(
                df=dataset, 
                expo_map=expo_map, 
                nside=nside, 
                ra_src=coordinate['RA'], 
                dec_src=coordinate['Dec'], 
                raggio_test=raggio_di_ricerca
            )
            
            # Formattiamo la riga per il file di testo
            riga_report = f"{nome:<20} | {N_oss:<10} | {N_att:<15.2f} | {sigma:<8.2f} | {p_val:.2e}\n"
            f_out.write(riga_report)
            
            # Stampiamo un breve riassunto a schermo
            print(f"   -> Statistica a {raggio_di_ricerca}°: {N_oss} oss. vs {N_att:.1f} att. (Significatività: {sigma:.2f} sigma)")
            
        f_out.write("=========================================================\n")
        
    # 4. Mappa globale finale
    mappa_calore_globale(dataset, sorgenti_da_analizzare, cartella_output)
    
    print("\n---------------------------------------------------")
    print("TUTTE LE ANALISI SONO STATE COMPLETATE CON SUCCESSO!")
    print(f"Puoi trovare i risultati statistici in: {percorso_report}")