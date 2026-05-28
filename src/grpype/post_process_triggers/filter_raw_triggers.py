import argparse
from datetime import timedelta  
from time import time

import numpy as np
from scipy.stats import norm
import pandas as pd

from gdt.missions.fermi.time import Time

from grpype.detection.utils import exp2_tail, sf
from grpype.detection.global_params import detectors, DATAPATH


def calibrate_dist(vals, fitting_func, bdurs):
    timescales = np.concatenate([np.around(0.002*1.35**np.arange(8), 3), np.around(0.04*1.35**np.arange(-2, 19), 3)])
    coeffs = {timescale: np.load(DATAPATH / f'dist_coeffs/coefficients{timescale:.3f}.npy') for timescale in timescales}

    x = np.linspace(2, 35, int(1e4))

    mask = np.isnan(bdurs)
    bdurs = bdurs[~mask]
    vals = vals[~mask]

    new_vals = np.zeros_like(vals)

    unique_bdurs = np.unique(bdurs)

    log_sf_x = np.log(norm.sf(x))

    for bdur in unique_bdurs:
        coeff = coeffs[bdur]
        matching_indices = (bdurs == bdur)

        sf_vals = np.log(sf(fitting_func, coeff, vals[matching_indices]))
        idx = np.abs(log_sf_x[:, None] - sf_vals).argmin(axis=0)
        new_vals[matching_indices] = x[idx]

    new_vals[vals >= 19] = vals[vals >= 19]
    
    return np.around(new_vals, 3)

def calibrate_dist_per_template_naive(df, fitting_func):
    timescales = np.concatenate([np.around(0.002*1.35**np.arange(8), 3), np.around(0.04*1.35**np.arange(-2, 19), 3)])
    coeffs = {timescale: np.load(DATAPATH / f'dist_coeffs/coefficients{timescale:.3f}.npy') for timescale in timescales}
    highsnr_coeffs = {timescale: np.load(DATAPATH / f'dist_coeffs/highsnr_coefficients{timescale:.3f}.npy') for timescale in timescales}

    calibrated = np.zeros_like(df.snr)

    for idx, row in df.iterrows():
        bdur = row.timescale
        temp_num = row.template_num

        if row.snr > 15:
            highsnr_coeff = highsnr_coeffs[bdur]
            calibrated[idx] = global_high_snr_fit_func_approx(row.snr, *highsnr_coeff) # not integrating here is an approximation which is fine for high SNR.
            # calibrated[idx] = row.snr
            continue
        elif row.snr < 6:
            calibrated[idx] = row.snr
            continue

        coeff = coeffs[bdur][temp_num]
        calibrated[idx] = norm.isf(sf(fitting_func, coeff, row.snr))


    return np.around(calibrated, 3)

def global_high_snr_fit_func_approx(x, a, b, c):
    return np.sqrt(-2*(np.log(a) - b * x - c * x**2))

def load_sharptimes(kind='.', abspath=None):
    sharp_path = DATAPATH / f'results/{kind}/sharptimes.txt' if abspath is None else abspath
    with open(sharp_path, 'r') as f:
        lines = f.readlines()

    fixed_lines = []
    for line in lines:
        if len(line.split()) == 2:
            temp = [float(st) for st in line.split('\n')[0].split()]
            fixed_lines.append(temp)
    
    return np.array(fixed_lines)

def add_loc_stat(trigdf, locdf):
    locdf['trigtime'] = pd.to_datetime(locdf['date'])
    if 'gcen_stat' not in trigdf.columns:
        trigdf['gcen_stat'] = 0.
    
    merged_df = pd.merge(
        trigdf, 
        locdf[['index', 'sun_stat', 'earth_stat', 'gcen_stat', 'trigtime']], 
        left_index=True, right_on='index', 
        how='left',
        suffixes=('_trig', '_loc')
    )

    merged_df = merged_df[(merged_df['trigtime_trig'] - merged_df['trigtime_loc']) <= pd.Timedelta(seconds=1)]
    merged_df = merged_df.dropna(subset=['sun_stat_loc', 'earth_stat_loc', 'gcen_stat_loc'])

    merged_df.rename(columns={
        'sun_stat_trig': 'sun_stat',   # No need to rename this as it’s already fine
        'earth_stat_trig': 'earth_stat',
        'gcen_stat_loc': 'gcen_stat',
        'trigtime_trig': 'trigtime'
    }, inplace=True)

    result_df = merged_df.reset_index(drop=True)
    return result_df

def filter_dates(df):
    df = df.sort_values(by='trigtime')
    df['rounded_timestamp'] = df['trigtime'].dt.floor('min')
    df = df.loc[df.groupby('rounded_timestamp')['snr'].idxmax()]
    df = df.sort_values(by='trigtime').reset_index(drop=True)

    inds = np.where(np.abs(df.trigtime.diff()) <= timedelta(seconds=80))[0]

    for i, ind in enumerate(inds):
        if df.iloc[inds[i]].snr > df.iloc[inds[i]-1].snr:
            df.iloc[inds[i]-1] = df.iloc[inds[i]]
        else:
            df.iloc[inds[i]] = df.iloc[inds[i]-1]
    
    df['rounded_timestamp'] = df['trigtime'].dt.floor('S')
    df = df.loc[df.groupby('rounded_timestamp')['snr'].idxmax()]

    return df

def associate_sharptimes(df, kind='.'):
    sharps = load_sharptimes(kind)
    # sharps = np.loadtxt(DATAPATH / f'results/{kind}/sharptimes.txt')
    sharps = sharps[np.argsort(sharps[:, 0])]

    # Create lower and upper bounds for each sharp range with padding
    sharp_starts = sharps[:, 0] - 30
    sharp_ends = sharps[:, 1] + 30

    # Vectorized condition to check if each trigtime_met falls within any sharp time range
    mask = np.zeros(len(df), dtype=bool)
    for start, end in zip(sharp_starts, sharp_ends):
        mask |= (df.trigmet >= start) & (df.trigmet <= end)

    # Drop rows where trigtime_met falls within a sharp time range
    df = df[~mask]

    return df

def associate_locations(df, trigdf):
    """
    Tag triggers that correspond to a solar flare or TGF from the timeslides/simulation dataframe.
    Also tags triggers occuring at the same time as the zero-lag.
    """
    for idx, date in zip(df.index, df['trigtime']):  # The timeslides/simulation dataframe
        matching = trigdf[np.abs(trigdf.trigtime - df.iloc[idx].trigtime) < pd.Timedelta(seconds=20)]
        if len(matching) > 0:
            # earth_stat = np.max(matching['earth_stat'])
            # sun_stat = np.max(matching['sun_stat'])
            # df.loc[idx, 'earth_stat'] = earth_stat
            # df.loc[idx, 'sun_stat'] = sun_stat

            snr = np.max(matching['snr'])
            if df.loc[idx, 'snr'] < 1.1*snr:
                df.loc[idx, 'rm_zerolag'] = True
            else:
                df.loc[idx, 'rm_zerolag'] = True
                trigdf.loc[matching.index, 'rm_zerolag'] = True
    
    return df, trigdf

def add_is_catalog(df, cat):
    # Create a timedelta comparison window of 60 seconds
    time_diff = pd.Timedelta(seconds=120)
    
    df['is_catalog'] = False
    df['catalog_type'] = None
    
    # Loop through the catalog and mark True for matching entries in one operation
    for cat_time, cat_type in zip(cat['date'], cat['trigger_type']):
        # Create a mask for times within 60 seconds of the catalog time
        mask = np.abs(df['trigtime'] - cat_time) < time_diff
        df.loc[mask, 'is_catalog'] = True
        df.loc[mask, 'catalog_type'] = cat_type

    return df

def add_t90(df, burstcat, assoc_time=60):
    df['catalog_t90'] = None
    time_diff = pd.Timedelta(seconds=assoc_time)
    
    for cat_time, cat_t90 in zip(burstcat['date'], burstcat['t90']):
        # Create a mask for times within 60 seconds of the catalog time
        mask = np.abs(df['trigtime'] - cat_time) < time_diff
        df.loc[mask, 'catalog_t90'] = cat_t90
    
    return df

def add_locsource(df, trigcat, assoc_time=60):
    df['cat_loc_source'] = None
    df['cat_ra'] = None
    df['cat_dec'] = None
    time_diff = pd.Timedelta(seconds=assoc_time)
    
    for cat_time, locsource, catra, catdec, in zip(trigcat['date'], trigcat['localization_source'], trigcat['ra'], trigcat['dec']):
        # Create a mask for times within 60 seconds of the catalog time
        mask = np.abs(df['trigtime'] - cat_time) < time_diff
        df.loc[mask, 'cat_loc_source'] = locsource
        df.loc[mask, 'cat_ra'] = catra
        df.loc[mask, 'cat_dec'] = catdec
    
    return df

def load_triggers(trigcat, burstcat, limit, kind='.', lower_limit=0):
    ptrigs = pd.read_csv(DATAPATH / f'results/{kind}/triggers.csv')
    ptrigs = ptrigs[ptrigs.timescale <= limit].reset_index(drop=True)
    ptrigs = ptrigs[ptrigs.timescale >= lower_limit].reset_index(drop=True)
    ptrigs['trigtime'] = pd.to_datetime(ptrigs['trigtime'])
    ptrigs = ptrigs[~np.isinf(ptrigs.snr)]

    ptrigs['orig_snr'] = ptrigs.snr
    ptrigs['snr'] = calibrate_dist_per_template_naive(ptrigs, exp2_tail)
    
    ptrigs['rm_zerolag'] = False
    ptrigs = add_is_catalog(ptrigs, trigcat)
    ptrigs = add_t90(ptrigs, burstcat)
    ptrigs = add_locsource(ptrigs, trigcat)

    i = 0
    while ptrigs.trigtime.diff().dt.total_seconds().min() < 60:
        ptrigs = filter_dates(ptrigs)
        if i == 20:
            print('EEROR?')
            break
        i += 1
    ptrigs['trigmet'] = ptrigs['trigtime'].apply(
        lambda x: int(Time(x.to_pydatetime(), scale='utc').fermi)
    )

    tstrigs = pd.read_csv(DATAPATH / f'results_timeslides/{kind}/triggers.csv')
    tstrigs = tstrigs[tstrigs.timescale <= limit].reset_index(drop=True)
    tstrigs = tstrigs[tstrigs.timescale >= lower_limit].reset_index(drop=True)
    tstrigs['trigtime'] = pd.to_datetime(tstrigs['trigtime'])
    tstrigs = tstrigs[~np.isinf(tstrigs.snr)]
    tstrigs['orig_snr'] = tstrigs.snr
    tstrigs['snr'] = calibrate_dist_per_template_naive(tstrigs, exp2_tail)
    tstrigs['rm_zerolag'] = False
    tstrigs, ptrigs = associate_locations(tstrigs, ptrigs)
    i = 0
    while tstrigs.trigtime.diff().dt.total_seconds().min() < 60:
        tstrigs = filter_dates(tstrigs)
        if i == 20:
            print('EEROR?')
            break
        i += 1
    tstrigs['trigmet'] = tstrigs['trigtime'].apply(
        lambda x: int(Time(x.to_pydatetime(), scale='utc').fermi)
    )

    ptrigs = associate_sharptimes(ptrigs, kind)
    tstrigs = associate_sharptimes(tstrigs, kind)

    ptrigs = ptrigs.sort_values(by='trigtime').reset_index(drop=True)
    tstrigs = tstrigs.sort_values(by='trigtime').reset_index(drop=True)

    ptrigs['earth_stat'] = ptrigs['earth_stat'].apply(lambda x: float(x))
    ptrigs['sun_stat'] = ptrigs['sun_stat'].apply(lambda x: float(x))
    ptrigs['gcen_stat'] = ptrigs['gcen_stat'].apply(lambda x: float(x))

    tstrigs['earth_stat'] = tstrigs['earth_stat'].apply(lambda x: float(x))
    tstrigs['sun_stat'] = tstrigs['sun_stat'].apply(lambda x: float(x))
    tstrigs['gcen_stat'] = tstrigs['gcen_stat'].apply(lambda x: float(x))

    ptrigs.loc[np.isinf(ptrigs.earth_stat), 'earth_stat'] = 999
    ptrigs.loc[np.isinf(ptrigs.sun_stat), 'sun_stat'] = 999
    ptrigs.loc[np.isinf(ptrigs.gcen_stat), 'gcen_stat'] = 999

    tstrigs.loc[np.isinf(tstrigs.earth_stat), 'earth_stat'] = 999
    tstrigs.loc[np.isinf(tstrigs.sun_stat), 'sun_stat'] = 999
    tstrigs.loc[np.isinf(tstrigs.gcen_stat), 'gcen_stat'] = 999

    return ptrigs, tstrigs

def load_catalogs():
    trigcat = pd.read_csv(DATAPATH / 'catalogs/trigcat.csv')
    trigcat['date'] = pd.to_datetime(trigcat['trigger_time'])

    burstcat = pd.read_csv(DATAPATH / 'catalogs/burstcat.csv')
    burstcat['date'] = pd.to_datetime(burstcat['trigger_time'])

    return trigcat, burstcat

def get_mfs(df):
    # mf_numers = df.iloc[:, -14*2:-14].to_numpy()
    # mf_vars = df.iloc[:, -14:].to_numpy()

    mf_numers = np.zeros((len(df), 14))
    mf_vars = np.zeros((len(df), 14))
    mf_zvars0 = np.zeros((len(df), 14))
    mf_zvars = np.ones((len(df), 14))
    for i, det in enumerate(detectors):
        mf_numers[:, i] = df[f'{det}_numer'].to_numpy()
        mf_vars[:, i] = df[f'{det}_var'].to_numpy()
        mf_zvars0[:, i] = df[f'{det}_test_zvar0'].to_numpy()
        mf_zvars[:, i] = df[f'{det}_test_zvar'].to_numpy()

    return mf_numers, mf_vars, mf_zvars0, mf_zvars

def get_singledet_mask(df, thresh=5):
    mf_numers, mf_vars, mf_zvars0, mf_zvars = get_mfs(df)

    test = (mf_numers.sum(axis=1)[:, None] - mf_numers - mf_zvars0) / np.sqrt(mf_vars.sum(axis=1)[:, None] - mf_vars) / np.sqrt(np.clip(mf_zvars - mf_zvars0**2, 0, np.inf) + 1e-5)
    # test = (mf_numers.sum(axis=1)[:, None] - mf_numers) / np.sqrt(mf_vars.sum(axis=1)[:, None] - mf_vars) / np.sqrt(mf_zvars + 1e-5)
    singledets = (test < thresh).any(axis=1)

    singledets |= (mf_vars.sum(axis=1) == 0)
    # print(df.iloc[np.where(mf_vars == 0)[0][0]])
    
    return singledets, test


def save_filtered_triggers(year, kind='all', save=True):
    trigcat, burstcat = load_catalogs()
    t0 = time()
    ptrigs, tstrigs = load_triggers(trigcat, burstcat, round(0.04*1.35**17, 3), f'{year}/{kind}')  # Note that it filters duplicates and resets the index to match the remove inds below
    with open(DATAPATH / f'results/{year}/filters_log.txt', 'a') as f:
        f.write(f'Loading took {time() - t0:.1f} seconds\n')
        f.write('Number of triggers after time maximization: \n')
        f.write(f'zerolag: {len(ptrigs)} \t timeslides: {len(tstrigs)}\n\n')

    gth = 5
    adj = 0.05

    ptrigs['adj_timing_stat'] = ptrigs.timing_stat - adj*ptrigs.orig_snr
    tstrigs['adj_timing_stat'] = tstrigs.timing_stat - adj*tstrigs.orig_snr

    thresh = 4

    singledet_pmask, ptest = get_singledet_mask(ptrigs, thresh)
    singledet_tsmask, tstest = get_singledet_mask(tstrigs, thresh)

    cleaned_ptrigs = ptrigs.iloc[~singledet_pmask]
    cleaned_tstrigs = tstrigs.iloc[~singledet_tsmask]
    with open(DATAPATH / f'results/{year}/filters_log.txt', 'a') as f:
        f.write('Number of triggers after single detector veto: \n')
        f.write(f'zerolag: {len(cleaned_ptrigs)} \t timeslides: {len(cleaned_tstrigs)}\n\n')

    gth = 5
    max_timing_timscale = 0.3

    cleaned_ptrigs = cleaned_ptrigs[~((cleaned_ptrigs.adj_timing_stat > gth) & (cleaned_ptrigs.timescale <= max_timing_timscale))]
    cleaned_tstrigs = cleaned_tstrigs[~((cleaned_tstrigs.adj_timing_stat > gth) & (cleaned_tstrigs.timescale <= max_timing_timscale))]
    with open(DATAPATH / f'results/{year}/filters_log.txt', 'a') as f:
        f.write('Number of triggers after timing stat filter: \n')
        f.write(f'zerolag: {len(cleaned_ptrigs)} \t timeslides: {len(cleaned_tstrigs)}\n\n')

    cleaned_ptrigs = cleaned_ptrigs[np.abs(cleaned_ptrigs.occultation_stat) < 4]
    cleaned_tstrigs = cleaned_tstrigs[np.abs(cleaned_tstrigs.occultation_stat) < 4]
    with open(DATAPATH / f'results/{year}/filters_log.txt', 'a') as f:
        f.write('Number of triggers after occultation filter: \n')
        f.write(f'zerolag: {len(cleaned_ptrigs)} \t timeslides: {len(cleaned_tstrigs)}\n\n')

    cleaned_ptrigs = cleaned_ptrigs[cleaned_ptrigs.shower_stat < 4]
    cleaned_tstrigs = cleaned_tstrigs[cleaned_tstrigs.shower_stat < 4]
    with open(DATAPATH / f'results/{year}/filters_log.txt', 'a') as f:
        f.write('Number of triggers after shower filter: \n')
        f.write(f'zerolag: {len(cleaned_ptrigs)} \t timeslides: {len(cleaned_tstrigs)}\n\n')

    cleaned_ptrigs = cleaned_ptrigs[cleaned_ptrigs.rm_zerolag == False]
    cleaned_tstrigs = cleaned_tstrigs[cleaned_tstrigs.rm_zerolag == False]
    with open(DATAPATH / f'results/{year}/filters_log.txt', 'a') as f:
        f.write('Number of triggers after timeslides association filter: \n')
        f.write(f'zerolag: {len(cleaned_ptrigs)} \t timeslides: {len(cleaned_tstrigs)}\n\n')

    sns = cleaned_ptrigs.iloc[:, 28:42].sum(axis=1) / np.sqrt(cleaned_ptrigs.iloc[:, 42:56].sum(axis=1))
    sns_ts = cleaned_tstrigs.iloc[:, 28:42].sum(axis=1) / np.sqrt(cleaned_tstrigs.iloc[:, 42:56].sum(axis=1))

    cleaned_ptrigs = cleaned_ptrigs[sns > 4]
    cleaned_tstrigs = cleaned_tstrigs[sns_ts > 4]
    with open(DATAPATH / f'results/{year}/filters_log.txt', 'a') as f:
        f.write('Number of triggers after drift glitch filter: \n')
        f.write(f'zerolag: {len(cleaned_ptrigs)} \t timeslides: {len(cleaned_tstrigs)}\n\n')

    # Add swift BAT trigger info:
    # -----------------------------
    swift_grbs = pd.read_csv(DATAPATH / 'catalogs/swift_grbs.csv')
    swift_grbs['trigtime_swift'] = pd.to_datetime(swift_grbs['trigtime_swift'])
    swift_grbs = swift_grbs.dropna(subset=['trigtime_swift']).sort_values(by='trigtime_swift').reset_index(drop=True)

    cleaned_ptrigs = cleaned_ptrigs.sort_values(by='trigtime').reset_index(drop=True)
    cleaned_ptrigs = pd.merge_asof(
        cleaned_ptrigs,swift_grbs,
        left_on='trigtime',
        right_on='trigtime_swift',
        direction='nearest',
        tolerance=pd.Timedelta(seconds=30)
        )
    
    filtered_sorted = cleaned_ptrigs.sort_values(by='trigtime').reset_index(drop=True)
    ts_filtered_sorted = cleaned_tstrigs.sort_values(by='trigtime').reset_index(drop=True)

    if save:
        # Here we do not have the kind on purpose, the final trigger list is in the parent folder
        filtered_sorted.to_csv(DATAPATH / f'results/{year}/filtered_triggers.csv', index=False)
        ts_filtered_sorted.to_csv(DATAPATH / f'results_timeslides/{year}/filtered_triggers.csv', index=False)
    
    return filtered_sorted, ts_filtered_sorted

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('year', type=str, help='Year of the data to process (e.g., "2014")')
    parser.add_argument('--kind', type=str, default='all', help='Kind of triggers to process (default: "all")')
    args = parser.parse_args()

    ptrigs, tstrigs = save_filtered_triggers(args.year, kind=args.kind, save=True)

if __name__ == '__main__':
    main()
