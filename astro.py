import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the data, skipping the first row if it's a header with a hash
# Actually, let's use pandas to properly parse it
df = pd.read_csv('auger.txt', sep='\t+', engine='python')
print(df.head())
print(df.columns)

# If columns have spaces or hashtags, clean them up
df.columns = [c.strip().replace('#', '') for c in df.columns]

plt.figure(figsize=(8, 6))
plt.hist(df['E'], bins=50, color='blue', alpha=0.7)
plt.title('Distribuzione degli eventi in funzione dell\'energia')
plt.xlabel('Energia (EeV)')
plt.ylabel('Numero di eventi')
plt.grid(axis='y', alpha=0.75)
plt.savefig('energy_histogram.png')
print("Histogram saved")
plt.figure(figsize=(8, 6))
plt.scatter(df['E'], df['Th'], alpha=0.5)
plt.title('Distribuzione degli eventi in funzione dell\'energia e dell\'angolo zenitale')
plt.xlabel('Energia (EeV)')
plt.ylabel('Angolo zenitale (gradi)')
plt.grid()
plt.savefig('energy_theta_scatter.png')
print("Scatter plot saved")

plt.figure(figsize=(8, 6))
plt.scatter(df['Th'], df['Ph'], alpha=0.5)
plt.title('Distribuzione degli eventi in funzione dell\'angolo zenitale e dell\'angolo azimutale')
plt.xlabel('Angolo zenitale (gradi)')
plt.ylabel('Angolo azimutale (gradi)')
plt.grid()
plt.savefig('theta_phi_scatter.png')
print("Scatter plot saved")

# --- DEFINIZIONE DELLA SORGENTE E DELLA TOP HAT ---
# Esempio: Centaurus A
ra_sorgente = 201.3  # gradi
dec_sorgente = -43.0 # gradi
raggio_top_hat = 10.0 # gradi (la dimensione della tua regione)

# 1. Convertiamo tutto in radianti perché le funzioni numpy (sin, cos) usano i radianti
ra_rad = np.radians(df['RA'])
dec_rad = np.radians(df['Dec'])
ra_src_rad = np.radians(ra_sorgente)
dec_src_rad = np.radians(dec_sorgente)

# 2. Calcoliamo la distanza angolare sferica per ogni evento
cos_theta = (np.sin(dec_rad) * np.sin(dec_src_rad) + 
             np.cos(dec_rad) * np.cos(dec_src_rad) * np.cos(ra_rad - ra_src_rad))

# Evitiamo errori numerici arrotondando valori infinitesimamente fuori da [-1, 1]
cos_theta = np.clip(cos_theta, -1.0, 1.0) 

# Distanza angolare in gradi
distanza_angolare = np.degrees(np.arccos(cos_theta))

# 3. APPLICHIAMO IL FILTRO TOP HAT (Creiamo un nuovo DataFrame solo con gli eventi dentro)
df_sorgente = df[distanza_angolare <= raggio_top_hat]

print(f"Eventi totali registrati: {len(df)}")
print(f"Eventi nella regione Top Hat (R={raggio_top_hat}° attorno a RA={ra_sorgente}, Dec={dec_sorgente}): {len(df_sorgente)}")

# --- PLOT: Mappa del cielo con la Top Hat ---
plt.figure(figsize=(10, 6))

# Disegna tutti gli eventi in grigio chiaro (Background)
plt.scatter(df['RA'], df['Dec'], color='gray', alpha=0.3, label='Tutti gli eventi', s=10)

# Disegna gli eventi nella Top Hat in rosso
plt.scatter(df_sorgente['RA'], df_sorgente['Dec'], color='red', alpha=0.8, label='Eventi Top Hat', s=20)

# Segna il centro della sorgente
plt.scatter(ra_sorgente, dec_sorgente, color='black', marker='*', s=200, label='Sorgente (Es. Cen A)')

plt.title('Mappa del cielo in Coordinate Equatoriali (RA, Dec)')
plt.xlabel('Ascensione Retta (gradi)')
plt.ylabel('Declinazione (gradi)')
plt.xlim(0, 360)
plt.ylim(-90, 90)
# plt.gca().invert_xaxis() 
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig('sky_map_tophat.png')
print("Sky map con regione Top Hat salvata.")