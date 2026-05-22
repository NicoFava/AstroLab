import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
import scipy.stats as stats
import time
from matplotlib.colors import LogNorm
from astropy.coordinates import SkyCoord
import astropy.units as u

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

def calcola_frazione_esposizione(expo_map, ra_src, dec_src, raggi):
    """
    Calcola analiticamente la frazione di esposizione totale contenuta 
    in vari raggi attorno alla sorgente usando Healpy.
    """
    nside = hp.get_nside(expo_map)
    expo_pulita = np.where(expo_map > 0, expo_map, 0)
    tot_expo = np.sum(expo_pulita)
    
    # CONVERSIONE: Da Coordinate Equatoriali (RA, Dec) a Galattiche (l, b)
    c = SkyCoord(ra=ra_src*u.degree, dec=dec_src*u.degree, frame='icrs')
    l_gal = c.galactic.l.degree
    b_gal = c.galactic.b.degree
    
    # Vettore 3D della sorgente per healpy (usando la colatitudine galattica)
    vec_src = hp.ang2vec(np.radians(90 - b_gal), np.radians(l_gal))
    
    frazioni = np.zeros(len(raggi))
    for i, r in enumerate(raggi):
        # Trova tutti i pixel dentro il raggio specificato
        pixel_in_raggio = hp.query_disc(nside, vec_src, np.radians(r))
        # Somma l'esposizione di questi pixel e dividi per il totale
        frazioni[i] = np.sum(expo_pulita[pixel_in_raggio]) / tot_expo
        
    return frazioni

def scan_2d_energia_raggio(df, frazioni_attese, ra_src, dec_src, raggi, energie):
    """
    Esegue lo scan bidimensionale. Per ogni energia e ogni raggio calcola 
    il p-value di Poisson rispetto all'isotropia pesata.
    """
    p_values_2d = np.zeros((len(energie), len(raggi)))
    osservati_2d = np.zeros((len(energie), len(raggi)))
    attesi_2d = np.zeros((len(energie), len(raggi)))
    
    distanze_all = calcola_distanza_angolare(df['RA'], df['Dec'], ra_src, dec_src)
    
    min_pval = 1.0
    best_E = 0
    best_R = 0
    best_oss = 0
    best_att = 0

    print("\n---> Avvio Scan 2D (Energia vs Raggio)...")
    
    for i, e_th in enumerate(energie):
        maschera_energia = df['E'] >= e_th
        n_eventi_totali_rimasti = np.sum(maschera_energia)
        distanze_tagliate = distanze_all[maschera_energia]
        
        for j, r in enumerate(raggi):
            oss = np.sum(distanze_tagliate <= r)
            att = n_eventi_totali_rimasti * frazioni_attese[j]
            
            if oss > 0:
                pval = stats.poisson.sf(oss - 1, att)
            else:
                pval = 1.0
                
            p_values_2d[i, j] = pval
            osservati_2d[i, j] = oss
            attesi_2d[i, j] = att
            
            if pval < min_pval:
                min_pval = pval
                best_E = e_th
                best_R = r
                best_oss = oss
                best_att = att

    print("=========================================================")
    print(" RISULTATI SCAN 2D: MASSIMO ECCESSO LOCALE TROVATO")
    print("=========================================================")
    print(f" Energia di Soglia (E_th) : >= {best_E} EeV")
    print(f" Raggio (Top-Hat)         : {best_R}°")
    print(f" Eventi Osservati         : {best_oss}")
    print(f" Eventi Attesi dal fondo  : {best_att:.2f}")
    print(f" P-Value (Poisson)        : {min_pval:.3e}")
    print(f" Significatività (Sigma)  : {stats.norm.isf(min_pval):.2f} σ")
    print("=========================================================\n")

    return p_values_2d, min_pval, best_E, best_R

def plot_heatmap_2d(p_values_2d, raggi, energie, best_E, best_R, out_dir):
    """Genera la mappa di calore 2D (Energia vs Raggio) in stile Auger."""
    os.makedirs(out_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 7))
    
    R_mesh, E_mesh = np.meshgrid(
        np.append(raggi - 0.5, raggi[-1] + 0.5), 
        np.append(energie - 0.5, energie[-1] + 0.5)
    )
    
    mesh = plt.pcolormesh(R_mesh, E_mesh, p_values_2d, 
                          norm=LogNorm(vmin=1e-7, vmax=1.0), 
                          cmap='hot_r', shading='flat')
    
    cbar = plt.colorbar(mesh)
    cbar.set_label('Local P-Value', fontsize=14)
    
    # Sintassi r"..." usata per evitare i SyntaxWarning
    plt.scatter(best_R, best_E, color='white', marker='P', s=200, edgecolors='black', 
                label=rf'Minimo Locale ({best_R}°, $\geq${best_E} EeV)')
    
    plt.title('Scan 2D P-Value: Regione di Centaurus A', fontsize=16)
    plt.xlabel(r'Raggio Top-Hat, $\Psi$ [gradi]', fontsize=14)
    plt.ylabel(r'Energia di Soglia, $E_{th}$ [EeV]', fontsize=14)
    
    plt.xlim(raggi[0]-0.5, raggi[-1]+0.5)
    plt.ylim(energie[0]-0.5, energie[-1]+0.5)
    plt.legend(loc='lower left')
    plt.grid(True, which='both', linestyle='--', alpha=0.3, color='black')
    
    save_path = os.path.join(out_dir, 'fig5_centaurus_2d_scan.png')
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"Heatmap 2D salvata in: {save_path}")

if __name__ == "__main__":
    
    RA_CEN_A = 201.3
    DEC_CEN_A = -43.0
    
    cartella_output = 'plotsNGC5128_Scan2D'
    file_dati = 'auger.txt'
    file_esposizione = 'exposure.fits'
    
    RAGGI_SCAN = np.arange(1.0, 31.0, 1.0)       
    ENERGIE_SCAN = np.arange(32.0, 81.0, 1.0)    
    
    os.makedirs(cartella_output, exist_ok=True)
    start_total_time = time.perf_counter()
    
    dataset = carica_dati(file_dati)
    print(f"Dati caricati: {len(dataset)} eventi trovati.")
    
    try:
        expo_map = hp.read_map(file_esposizione)
        nside = hp.get_nside(expo_map)
        print(f"Mappa Healpix caricata (NSIDE={nside}).")
    except Exception as e:
        print(f"ERRORE: Impossibile caricare {file_esposizione}.")
        exit()
        
    print(f"Calcolo esatto delle frazioni di esposizione attorno a Cen A...")
    frazioni_attese = calcola_frazione_esposizione(expo_map, RA_CEN_A, DEC_CEN_A, RAGGI_SCAN)
    
    p_values_2d, p_min, b_E, b_R = scan_2d_energia_raggio(
        df=dataset, 
        frazioni_attese=frazioni_attese, 
        ra_src=RA_CEN_A, dec_src=DEC_CEN_A, 
        raggi=RAGGI_SCAN, 
        energie=ENERGIE_SCAN
    )
    
    plot_heatmap_2d(p_values_2d, RAGGI_SCAN, ENERGIE_SCAN, b_E, b_R, cartella_output)
    
    end_total_time = time.perf_counter()
    print(f"\nOperazione completata in {end_total_time - start_total_time:.2f} secondi.")