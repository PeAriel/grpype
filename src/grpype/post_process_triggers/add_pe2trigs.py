import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from astropy.coordinates import SkyCoord, BarycentricMeanEcliptic
from astropy.time import Time
import astropy.units as u

from grpype.detection.global_params import DATAPATH


def replace_nans_with_defaults(df):
    pe_nans_msk = df.ra_med.isna()
    df.loc[pe_nans_msk, ['ra_med', 'ra_max']] = df.loc[pe_nans_msk, 'ra']
    df.loc[pe_nans_msk, ['dec_med', 'dec_max']] = df.loc[pe_nans_msk, 'dec']
    df.loc[pe_nans_msk, ['alpha_med', 'alpha_max']] = df.loc[pe_nans_msk, 'alpha']
    df.loc[pe_nans_msk, ['beta_med', 'beta_max']] = df.loc[pe_nans_msk, 'beta']
    df.loc[pe_nans_msk, ['epeak_med', 'epeak_max']] = df.loc[pe_nans_msk, 'epeak']
    return df

def add_coords_transforms(df, mode='med'):
    c = SkyCoord(ra=df[f'ra_{mode}'].to_numpy()*u.deg, dec=df[f'dec_{mode}'].to_numpy()*u.deg, frame="icrs")
    t = Time(df["trigtime"].to_numpy(), scale="utc")
    ecl = c.transform_to(BarycentricMeanEcliptic(equinox=t))
    
    df['b'] = c.galactic.b.degree
    df['l'] = c.galactic.l.degree
    df[f"b_{mode}"] = c.galactic.b.degree
    df[f"l_{mode}"] = c.galactic.l.degree
    df[f"ecl_lon_{mode}"] = ecl.lon.deg
    df[f"ecl_lat_{mode}"] = ecl.lat.deg
    df[f"dist_to_ecliptic_{mode}"] = abs(ecl.lat.deg)

    return df

def save_pe_triggers(year, save=True, input_root=None, output_root=None):
    base_input = Path(input_root) if input_root is not None else DATAPATH
    base_output = Path(output_root) if output_root is not None else DATAPATH

    ptrigs = pd.read_csv(base_input / f'results/{year}/filtered_triggers.csv')
    tstrigs = pd.read_csv(base_input / f'results_timeslides/{year}/filtered_triggers.csv')

    ptrigs['trigtime'] = pd.to_datetime(ptrigs['trigtime'])
    tstrigs['trigtime'] = pd.to_datetime(tstrigs['trigtime'])

    pe = pd.read_csv(base_input / f'results/{year}/filtered_triggers_pe.csv').sort_values(by='trigger_index')
    ptrigs = ptrigs.join(pe.set_index("trigger_index"), how="left")

    tspe = pd.read_csv(base_input / f'results_timeslides/{year}/filtered_triggers_pe.csv').sort_values(by='trigger_index')
    tstrigs = tstrigs.join(tspe.set_index("trigger_index"), how="left")

    ptrigs = replace_nans_with_defaults(ptrigs)
    tstrigs = replace_nans_with_defaults(tstrigs)

    ptrigs = add_coords_transforms(ptrigs, mode='med')
    ptrigs = add_coords_transforms(ptrigs, mode='max')
    tstrigs = add_coords_transforms(tstrigs, mode='med')
    tstrigs = add_coords_transforms(tstrigs, mode='max')

    if save:
        ptrigs.to_csv(base_output / f'results/{year}/filtered_triggers_with_pe.csv', index=False)
        tstrigs.to_csv(base_output / f'results_timeslides/{year}/filtered_triggers_with_pe.csv', index=False)

    return ptrigs, tstrigs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('year', type=str, help='Year of the data to process (e.g., "2014")')
    parser.add_argument("--input-root", type=str, default=None)
    parser.add_argument("--output-root", type=str, default=None)
    args = parser.parse_args()

    ptrigs, tstrigs = save_pe_triggers(args.year, save=True, input_root=args.input_root, output_root=args.output_root)

if __name__ == '__main__':
    main()