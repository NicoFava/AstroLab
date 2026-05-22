# AstroLab: Cosmic Ray Statistical Analysis

> 🇮🇹 **[Italiano](#italiano)** | 🇬🇧 **[English](#english)**

---

<a name="italiano"></a>
# 🇮🇹 AstroLab: Analisi Statistica di Raggi Cosmici

## Panoramica

**AstroLab** è una suite di strumenti Python per l'analisi statistica avanzata di dati di raggi cosmici ad altissima energia provenienti dal **Pierre Auger Observatory**. Il progetto implementa sofisticate tecniche Monte Carlo per identificare correlazioni anomale tra gli eventi di raggi cosmici osservati e sorgenti astrofisiche extragalattiche, con focus particolare su **Centaurus A (NGC 5128)**.

### Obiettivi Scientifici

- **Anisotropia cosmiche**: Identificare clustering non-isotropi di raggi cosmici attorno a sorgenti extragalattiche
- **Test statistici robusti**: Implementazione di test Monte Carlo pesati sulla mappa di esposizione strumentale
- **Analisi multi-scala**: Scan simultanei su 40 raggi angolari (da 1° a 40°)
- **Penalizzazione statistica**: Correzione per test multipli mantenendo potenza statistica

## Struttura del Progetto

```
AstroLab/
├── astro.py                      # Analisi base globale - test Monte Carlo
├── astroNGC5128.py              # Analisi ottimizzata con Numba (NGC 5128)
├── astroNGC5128CutE.py          # Analisi con filtro energetico
├── astroNGC5128sEDscan.py       # Scan bidimensionale energia-raggio
├── global_pvalue.py              # Calcolo p-value globale penalizzato
├── penalizzazione.py             # Visualizzazione convergenza p-value
├── auger.txt                     # Dataset events (RA, Dec, Energia)
├── exposure.fits                 # Mappa FITS esposizione strumentale
├── penalizzazione_log.txt        # Log simulazioni sequenziali
├── OldPlots/                     # Archivio risultati precedenti
├── plotsNGC5128/                 # Output analisi NGC 5128
├── plotsNGC5128_CutE/            # Output analisi NGC 5128 con filtro
└── plotsNGC5128_Scan2D/          # Output scan bidimensionali
```

## Descrizione dei Moduli

### 1. **astro.py** - Analisi Globale Monte Carlo
Implementa il test statistico base per l'identificazione di anisotropie:

**Funzioni principali:**
- `carica_dati()`: Caricamento e pulizia dataset Auger (RA, Dec, Energia)
- `calcola_distanza_angolare()`: Distanza sferica tra coppie di coordinate celesti
- `genera_cielo_isotropo_pesato()`: Generazione eventi Monte Carlo distribuiti secondo la mappa di esposizione
- `test_statistico_montecarlo_globale()`: Test simultaneo su tutti i raggi

**Algoritmo:**
1. Carica gli eventi osservati dal dataset
2. Genera N cieli isotropi simulati (default 10,000), rispettando il pattern di esposizione dello strumento
3. Per ogni sorgente e ogni raggio (1°-40°), conta gli eventi osservati
4. Confronta con la distribuzione simulata per calcolare p-value
5. Non applica penalizzazioni locali

### 2. **astroNGC5128.py** - Analisi Ottimizzata (Numba JIT)
Versione accelerata tramite compilazione JIT (Just-In-Time) con Numba per prestazioni produzione:

**Ottimizzazioni:**
- `@njit(parallel=True)`: Parallelizzazione multi-core dei loop Monte Carlo
- Inverse Transform Sampling sulla CDF dell'esposizione
- Coordinate in radianti per minimizzare operazioni floating-point
- Pre-calcolo della CDF per estrazione veloce

**Funzioni:**
- `prepara_dati_numba()`: Pre-calcolo strutture dati ottimizzate
- `esegui_simulazioni_numba_multi()`: Motore Monte Carlo Numba (parallelo)

**Vantaggi:**
- 10-100× più veloce di `astro.py` a parità di statistiche
- Permette 100,000+ simulazioni in tempo ragionevole

### 3. **astroNGC5128CutE.py** - Analisi con Filtro Energetico
Extends `astroNGC5128.py` con filtri energetici per selezionare sottocampioni di eventi:

**Feature:**
- Filtraggio su intervalli energetici (es. E > 60 EeV)
- Analisi della correlazione evento-energia con posizione sorgente
- Valutazione della significanza in funzione dell'energia

### 4. **astroNGC5128sEDscan.py** - Scan Bidimensionale Energia-Raggio
Implementa uno scan 2D simultaneo su energia e raggio angolare:

**Algoritmo:**
- Per ogni bin energetico (E_1, E_2):
  - Per ogni raggio da 1° a 40°:
    - Conta gli eventi osservati
    - Calcola la frazione di esposizione attesa
    - Valuta il p-value di Poisson

**Output:**
- Mappa 2D p-value(energia, raggio)
- Identificazione di picchi di significanza anomala
- Visualizzazione con scala logaritmica

### 5. **global_pvalue.py** - Penalizzazione Statistica Globale
Calcola il p-value globale corretto per il test multiplo:

**Metodologia:**
1. Esegue M batch sequenziali di simulazioni Monte Carlo
2. Per ogni batch: calcola quante simulazioni superano il max p-value osservato
3. Somma cumulativamente i superamenti
4. p-value_globale = superamenti_totali / simulazioni_totali

**Gestisce:**
- Correzione per look-elsewhere effect
- Convergenza statistica con >100,000 simulazioni
- Salvare dello stato in log per ripresa incrementale

### 6. **penalizzazione.py** - Visualizzazione Convergenza
Plotta il trend del p-value globale durante le simulazioni sequenziali:

**Output:**
- Grafico con asse Y in scala logaritmica
- Linea di convergenza mostrando stima finale
- Validazione della significanza statistica raggiunta

## Dataset e Configurazione

### Input Data

**auger.txt**: Tab-separated, formato:
```
#  RA         Dec        Energia(EeV)
   12.345     -45.678    78.5
   ...
```

**exposure.fits**: Mappa HEALPix FITS dell'esposizione strumentale in coordinate Galattiche

### Sorgenti Astronomiche Analizzate

Default: `Centaurus A (NGC 5128)`
- **RA**: 201.36° | **Dec**: -43.02°
- **Classe**: AGN (Active Galactic Nuclei) - Galassia ellittica gigante
- **Distanza**: ~3.7 Mpc
- **Significanza**: Potenziale sorgente di raggi cosmici ultra-ad alta energia

Altre sorgenti disponibili: M87 (Virgo A), Fornax A, NGC 253, M83, Vela SNR, Sagittarius A

## Tecnologie e Dipendenze

```python
numpy           # Calcoli numerici vettorizzati
pandas          # Manipolazione dati tabellari
matplotlib      # Visualizzazione
healpy          # Operazioni su mappe sferiche HEALPix
scipy.stats     # Distribuzioni statistiche (Poisson, normale)
astropy         # Coordinate astronomiche + trasformazioni
numba           # Compilazione JIT per Python -> C performance
```

## Utilizzo

### Setup Ambiente

```bash
# Attivare virtual environment (assumendo lab_env)
source /home/fava/lab/lab_env/bin/activate

# Installare dipendenze (se necessario)
pip install numpy pandas matplotlib healpy scipy astropy numba
```

### Esecuzione Analisi

**Analisi base globale** (lenta, per piccole simulazioni):
```bash
python astro.py
```

**Analisi NGC 5128 ottimizzata** (produzione):
```bash
python astroNGC5128.py
```

**Analisi con filtro energetico**:
```bash
python astroNGC5128CutE.py
```

**Scan bidimensionale**:
```bash
python astroNGC5128sEDscan.py
```

**Calcolo p-value globale penalizzato**:
```bash
python global_pvalue.py
```

**Visualizzazione convergenza**:
```bash
python penalizzazione.py
```

## Output e Risultati

### Directory Risultati

- **plotsNGC5128/**: Contiene report statistici e curve p-value per NGC 5128
- **plotsNGC5128_CutE/**: Risultati con filtri energetici
- **plotsNGC5128_Scan2D/**: Mappe 2D energia-raggio
- **OldPlots/**: Storico analisi precedenti (Centaurus A, M87, Fornax A, LMC, SMC, etc.)

### File di Output Tipici

```
report_statistico_SCAN_LOCALE.txt    # Resoconto p-value per raggi
pvalue_scans/                        # Grafici curve p-value
tophat_maps/                         # Mappe di densità eventi
progressione_definitiva_v2.png       # Convergenza p-value globale
```

## Metodologia Statistica

### Test di Anisotropia

**Ipotesi nulla (H₀)**: Gli eventi di raggi cosmici sono distribuiti isotropicamente secondo la mappa di esposizione.

**Ipotesi alternativa (H₁)**: Esiste clustering preferenziale verso specifiche sorgenti astronomiche.

### Approccio Monte Carlo Pesato su Esposizione

1. **Costruzione CDF esposizione**: 
   - Integra la mappa di esposizione in HEALPix
   - Normalizza a distribuzione di probabilità

2. **Generazione cieli simulati**:
   - Inverse Transform Sampling sulla CDF
   - Mantiene statistiche di esposizione osservate
   - Genera N_eventos ~ O(10,000-100,000)

3. **Conteggi e p-value**:
   - Per ogni raggio: n_oss = eventi osservati entro raggio
   - Per ogni simulazione: n_sim = eventi simulati entro raggio
   - p-value = P(N_sim ≥ n_oss | H₀)

4. **Penalizzazione globale**:
   - Correzione for "look-elsewhere effect"
   - p-value_globale con 100,000+ simulazioni

## Note Scientifiche

### Assunzioni Critiche

1. **Isotropia baseline**: In assenza di sorgenti, i raggi cosmici dovrebbero distribuirsi uniformemente ponderati dall'esposizione
2. **Stabilità esposizione**: Assumiamo che la mappa di esposizione sia stabile nel periodo di osservazione
3. **Indipendenza eventi**: Non considerati correlazioni temporali tra eventi successivi

### Limitazioni

- **Sensibilità angolare**: Risoluzione limitata da angular resolution dell'osservatorio
- **Statistica bassa**: Piccolo numero di eventi a energie estreme (E > 100 EeV)
- **Sistematiche**: Possibili bias non modellati nella mappa di esposizione
- **Look-elsewhere effect**: Test multipli su 40 raggi riducono significanza locale

## Risultati Precedenti

Nel database `OldPlots/` sono conservate analisi su multiple sorgenti e campioni:
- Centaurus A: Report completo, mappe tophat per 9 sorgenti diverse
- Scan locale dettagliato (multipli raggi simultanei)
- Statistiche Monte Carlo con variazioni di metodologia

## Autore e Licenza

**Autore**: Nicolò Favagrossa  
**Licenza**: MIT (vedi LICENSE)  
**Versione**: 2026  
**Repository**: Private research project

## Contatti e Sviluppi Futuri

### Possibili Estensioni

1. **Correlazioni temporali**: Analisi di clustering temporale degli eventi
2. **Modelli spettrali**: Fit della distribuzione energetica attorno alle sorgenti
3. **Machine Learning**: Classificazione di eventi anomali
4. **Analisi multi-messenger**: Correlazione con osservazioni gravitazionali/neutrini
5. **Scan anisotropia globale**: Ricerca di anisotropie non associate a sorgenti specifiche

### Performance Attuale

- **astro.py**: 10,000 simulazioni → ~5-10 min (single core)
- **astroNGC5128.py**: 100,000 simulazioni → ~2-3 min (8 cores, con Numba)
- **Penalizzazione**: 1,000,000 simulazioni → ~30-40 min (distributed)

---

**Ultimo aggiornamento**: Maggio 2026  
**Status**: Production - Analysis pipeline active

---

<a name="english"></a>
# 🇬🇧 AstroLab: Cosmic Ray Statistical Analysis

## Overview

**AstroLab** is a Python toolkit for advanced statistical analysis of ultra-high-energy cosmic ray data from the **Pierre Auger Observatory**. The project implements sophisticated Monte Carlo techniques to identify anomalous correlations between observed cosmic ray events and extragalactic astrophysical sources, with particular focus on **Centaurus A (NGC 5128)**.

### Scientific Goals

- **Cosmic anisotropies**: Detect non-isotropic clustering of cosmic rays around extragalactic sources
- **Robust statistical tests**: Implementation of Monte Carlo tests weighted by instrumental exposure map
- **Multi-scale analysis**: Simultaneous scans over 40 angular radii (from 1° to 40°)
- **Statistical penalization**: Multiple testing correction while maintaining statistical power

## Project Structure

```
AstroLab/
├── astro.py                      # Base global analysis - Monte Carlo tests
├── astroNGC5128.py              # Numba-optimized analysis (NGC 5128)
├── astroNGC5128CutE.py          # Analysis with energy filtering
├── astroNGC5128sEDscan.py       # 2D energy-radius scan
├── global_pvalue.py              # Penalized global p-value computation
├── penalizzazione.py             # P-value convergence visualization
├── auger.txt                     # Event dataset (RA, Dec, Energy)
├── exposure.fits                 # HEALPix FITS exposure map (instrumental)
├── penalizzazione_log.txt        # Sequential simulation log
├── OldPlots/                     # Archive of previous results
├── plotsNGC5128/                 # NGC 5128 analysis output
├── plotsNGC5128_CutE/            # NGC 5128 filtered analysis output
└── plotsNGC5128_Scan2D/          # 2D scan output
```

## Module Descriptions

### 1. **astro.py** - Global Monte Carlo Analysis
Implements the baseline statistical test for anisotropy detection:

**Main functions:**
- `carica_dati()`: Load and clean Auger dataset (RA, Dec, Energy)
- `calcola_distanza_angolare()`: Spherical angular distance between sky coordinates
- `genera_cielo_isotropo_pesato()`: Generate Monte Carlo events distributed according to exposure map
- `test_statistico_montecarlo_globale()`: Simultaneous test across all radii

**Algorithm:**
1. Load observed events from dataset
2. Generate N isotropic simulated skies (default 10,000), respecting instrumental exposure pattern
3. For each source and each radius (1°-40°), count observed events
4. Compare with simulated distribution to compute p-values
5. No local penalizations applied

### 2. **astroNGC5128.py** - Optimized Analysis (Numba JIT)
Accelerated version via Numba Just-In-Time compilation for production performance:

**Optimizations:**
- `@njit(parallel=True)`: Multi-core parallelization of Monte Carlo loops
- Inverse Transform Sampling on exposure CDF
- Radian coordinates to minimize floating-point operations
- Pre-computed CDF for fast extraction

**Functions:**
- `prepara_dati_numba()`: Pre-compute optimized data structures
- `esegui_simulazioni_numba_multi()`: Numba Monte Carlo engine (parallel)

**Benefits:**
- 10-100× faster than `astro.py` for equivalent statistics
- Enables 100,000+ simulations in reasonable time

### 3. **astroNGC5128CutE.py** - Energy-Filtered Analysis
Extends `astroNGC5128.py` with energy filters to select event subsamples:

**Features:**
- Energy interval filtering (e.g., E > 60 EeV)
- Analysis of event-energy correlation with source position
- Significance evaluation as function of energy

### 4. **astroNGC5128sEDscan.py** - 2D Energy-Radius Scan
Implements simultaneous 2D scan over energy and angular radius:

**Algorithm:**
- For each energy bin (E_1, E_2):
  - For each radius from 1° to 40°:
    - Count observed events
    - Calculate expected exposure fraction
    - Evaluate Poisson p-value

**Output:**
- 2D p-value map: p-value(energy, radius)
- Identification of anomalous significance peaks
- Log-scale visualization

### 5. **global_pvalue.py** - Global Statistical Penalization
Computes globally corrected p-value for multiple testing:

**Methodology:**
1. Execute M sequential batches of Monte Carlo simulations
2. Per batch: calculate how many simulations exceed max observed p-value
3. Cumulatively sum exceedances
4. global_p-value = total_exceedances / total_simulations

**Handles:**
- Correction for look-elsewhere effect
- Statistical convergence with >100,000 simulations
- Incremental state saving to log for resume capability

### 6. **penalizzazione.py** - Convergence Visualization
Plots global p-value trend during sequential simulations:

**Output:**
- Plot with logarithmic Y-axis scale
- Convergence line showing final estimate
- Validation of achieved statistical significance

## Dataset and Configuration

### Input Data

**auger.txt**: Tab-separated format:
```
#  RA         Dec        Energy(EeV)
   12.345     -45.678    78.5
   ...
```

**exposure.fits**: HEALPix FITS map of instrumental exposure in Galactic coordinates

### Analyzed Astronomical Sources

Default: `Centaurus A (NGC 5128)`
- **RA**: 201.36° | **Dec**: -43.02°
- **Type**: AGN (Active Galactic Nuclei) - Giant elliptical galaxy
- **Distance**: ~3.7 Mpc
- **Significance**: Potential source of ultra-high-energy cosmic rays

Other available sources: M87 (Virgo A), Fornax A, NGC 253, M83, Vela SNR, Sagittarius A

## Technologies and Dependencies

```python
numpy           # Vectorized numerical computations
pandas          # Tabular data manipulation
matplotlib      # Visualization
healpy          # Spherical map operations (HEALPix)
scipy.stats     # Statistical distributions (Poisson, normal)
astropy         # Astronomical coordinates + transformations
numba           # JIT compilation Python -> C performance
```

## Usage

### Environment Setup

```bash
# Activate virtual environment (assuming lab_env)
source /home/fava/lab/lab_env/bin/activate

# Install dependencies (if needed)
pip install numpy pandas matplotlib healpy scipy astropy numba
```

### Running Analyses

**Base global analysis** (slow, for small simulations):
```bash
python astro.py
```

**NGC 5128 optimized analysis** (production):
```bash
python astroNGC5128.py
```

**Energy-filtered analysis**:
```bash
python astroNGC5128CutE.py
```

**2D scan**:
```bash
python astroNGC5128sEDscan.py
```

**Penalized global p-value**:
```bash
python global_pvalue.py
```

**Convergence visualization**:
```bash
python penalizzazione.py
```

## Output and Results

### Results Directories

- **plotsNGC5128/**: Statistical reports and p-value curves for NGC 5128
- **plotsNGC5128_CutE/**: Results with energy filters
- **plotsNGC5128_Scan2D/**: 2D energy-radius maps
- **OldPlots/**: Historical analyses (Centaurus A, M87, Fornax A, LMC, SMC, etc.)

### Typical Output Files

```
report_statistico_SCAN_LOCALE.txt    # P-value report across radii
pvalue_scans/                        # P-value curve plots
tophat_maps/                         # Event density maps
progressione_definitiva_v2.png       # Global p-value convergence
```

## Statistical Methodology

### Anisotropy Test

**Null hypothesis (H₀)**: Cosmic ray events are isotropically distributed according to the exposure map.

**Alternative hypothesis (H₁)**: Preferential clustering toward specific astronomical sources exists.

### Exposure-Weighted Monte Carlo Approach

1. **Exposure CDF construction**: 
   - Integrate HEALPix exposure map
   - Normalize to probability distribution

2. **Simulated sky generation**:
   - Inverse Transform Sampling on CDF
   - Preserve observed exposure statistics
   - Generate N_events ~ O(10,000-100,000)

3. **Counting and p-values**:
   - Per radius: n_obs = observed events within radius
   - Per simulation: n_sim = simulated events within radius
   - p-value = P(N_sim ≥ n_obs | H₀)

4. **Global penalization**:
   - Correction for "look-elsewhere effect"
   - global_p-value with 100,000+ simulations

## Scientific Notes

### Critical Assumptions

1. **Baseline isotropy**: Without sources, cosmic rays should distribute uniformly weighted by exposure
2. **Exposure stability**: Assume exposure map is stable over observation period
3. **Event independence**: Temporal correlations between successive events not considered

### Limitations

- **Angular sensitivity**: Resolution limited by observatory angular resolution
- **Low statistics**: Small number of events at extreme energies (E > 100 EeV)
- **Systematics**: Possible unmodeled biases in exposure map
- **Look-elsewhere effect**: Multiple tests over 40 radii reduce local significance

## Previous Results

Archive in `OldPlots/` contains analyses on multiple sources and samples:
- Centaurus A: Complete report, tophat maps for 9 different sources
- Detailed local scan (multiple simultaneous radii)
- Monte Carlo statistics with methodology variations

## Author and License

**Author**: Nicolò Favagrossa  
**License**: MIT (see LICENSE)  
**Version**: 2026  
**Repository**: Private research project

## Contacts and Future Developments

### Possible Extensions

1. **Temporal correlations**: Clustering analysis of event temporal patterns
2. **Spectral models**: Energy distribution fitting around sources
3. **Machine Learning**: Classification of anomalous events
4. **Multi-messenger analysis**: Correlation with gravitational wave/neutrino observations
5. **Global anisotropy scan**: Search for anisotropies unassociated with specific sources

### Current Performance

- **astro.py**: 10,000 simulations → ~5-10 min (single core)
- **astroNGC5128.py**: 100,000 simulations → ~2-3 min (8 cores, with Numba)
- **Penalization**: 1,000,000 simulations → ~30-40 min (distributed)

---

**Last updated**: May 2026  
**Status**: Production - Analysis pipeline active
