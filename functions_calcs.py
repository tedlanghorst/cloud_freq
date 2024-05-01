import pandas as pd
import numpy as np
from scipy.signal import welch, coherence, csd
from tqdm import tqdm

def calc_masked_norm_diff(sites_in: pd.DataFrame,
                          q: pd.Series,
                          mask: pd.Series,
                          quant: float,
                          col_name: str) -> pd.DataFrame:
    
    grouped = q.groupby(['id'])
    all_values = grouped.quantile(quant)
    
    grouped = q[~mask].groupby(['id'])
    no_cloud_values = grouped.quantile(quant)
    
    norm_diff = ((no_cloud_values - all_values) / all_values).rename(col_name)
    sites_out = sites_in.join(norm_diff)
    
    return sites_out


def resampled_norm_diff(sites_in: pd.DataFrame,
                       df_in: pd.DataFrame,
                       windows: list,
                       quants: list) -> pd.DataFrame:
    
    df_in['Q_mask'] = df_in['Q']
    df_in.loc[df_in['cloud_binary'],'Q_mask'] = np.nan
    df = df_in.reset_index(level='id')[['id','Q','Q_mask']]
    grouped = df.groupby('id')

    columns = pd.MultiIndex.from_product([windows, quants], names=['Window', 'Quantile'])
    resampled_df = pd.DataFrame(index=grouped.groups.keys(), columns=columns)

    for window in tqdm(windows, desc="Rolling means"):
        resampled = grouped[['Q','Q_mask']].resample(window).mean()
        regrouped = resampled.reset_index(level='id').groupby('id')
        resampled_quants = regrouped.quantile(quants)
        for quant in quants:
            tmp = resampled_quants.xs(quant, level=1)
            ND = (tmp['Q_mask']-tmp['Q'])/tmp['Q']

            for idx in ND.index:
                resampled_df.loc[idx, (window, quant)] = ND.loc[idx]
            
    sites_out = sites_in.copy()
    sites_out.columns = pd.MultiIndex.from_product([['sites'], sites_out.columns])
    
    out = sites_out.join(resampled_df)
    return out

def calc_quantiles(df_in: pd.DataFrame):
    # Rank each Q value
    df_in['Q_rank'] = df_in.groupby('id')['Q'].rank()
    
    # Calculate the minimum and maximum rank for each 'id'
    min_rank = df_in.groupby('id')['Q_rank'].transform('min')
    max_rank = df_in.groupby('id')['Q_rank'].transform('max')
    
    # Normalize the 'Q_rank' to the range [0, 1]
    Q_norm = (df_in['Q_rank'] - min_rank) / (max_rank - min_rank)
    
    return Q_norm

def calc_phase_amp(a: pd.Series,
                   b: pd.Series):
    a = a - a.mean()
    b = b - b.mean()
    amp1 = (a.max() - b.min()) / 2
    amp2 = (a.max() - b.min()) / 2
    amp = np.min([amp1,amp2])
    corr = np.correlate(a, b, mode='full')

    # Find the index of the maximum correlation value
    max_corr_index = np.argmax(corr)
    
    # Calculate the phase offset
    n = len(a)
    phase_offset = np.abs(max_corr_index - (n - 1))
    # Wrap around for cross-year offsets
    if phase_offset > (n//2):
        phase_offset = phase_offset - (n//2)
        
    return phase_offset/n, amp

def calc_spectral_props(df_in: pd.DataFrame):
    phase = []
    power = []
    coh = []
    idx = []   
    for i,g in tqdm(df_in.groupby('id'),total=len(df_in.index.get_level_values('id').unique())):
        g = g.droplevel('id')

        #number of valid days per year. Excludes missing data or ice flagged days.
        nperseg = 365.25*len(g)/(g.index.max()-g.index.min()).days

        if (len(g) < nperseg) | (nperseg<180):
            continue
        
        # Extract discharge (Q) and cloud cover (cloudMask) columns
        discharge = g['Q'].values
        cloud_cover = g['cloudMask'].values

        # Spectral analysis
        fs = 1  # Sampling frequency (1 sample per day)
        f_coherence, Cxy = coherence(discharge, cloud_cover, fs=fs, nperseg=nperseg)

        # Calculate cross spectral density which contains phase information
        f, Pxy = csd(discharge, cloud_cover, fs=fs, nperseg=nperseg)

        # Calculate the phase spectrum in degrees
        phase_spectrum = np.angle(Pxy)

        phase.append(phase_spectrum[1])
        power.append(np.linalg.norm(Pxy))
        coh.append(Cxy[1])
        idx.append(i)
    
    df_out = pd.DataFrame({'CSD_phase':phase,'CSD_magnitude':power,'coherence':coh},index=idx)
    return df_out