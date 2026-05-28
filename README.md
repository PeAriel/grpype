# grpype

**grpype** is a matched-filter search pipeline for detecting short gamma-ray
bursts (sGRBs) in [Fermi Gamma-ray Burst Monitor (GBM)](https://fermi.gsfc.nasa.gov/ssc/data/access/gbm/)
time-tagged event (TTE) data. It implements the method described in
[Perera, Zackay & Venumadhav (2025)](https://arxiv.org/abs/2507.05739),
which achieved a ~50% increase in the number of sGRB detections compared to
the standard GBM catalog.

The pipeline covers the full workflow from raw photon data to a vetted trigger
catalog:

- **Template bank generation** — builds banks of spectral, sky-direction, and
  time-profile templates, as well as instrumental glitch templates used for
  cleaning.
- **Matched-filter detection** — convolves GBM TTE data against the template
  banks with SVD-based background drift correction, automatic glitch rejection,
  and calibrated detection thresholds.
- **Trigger post-processing** — filters, clusters, and classifies raw triggers;
  concatenates results across time slices and timescales.
- **Parameter estimation** — localizes candidates on the sky (HEALPix maps via
  MCMC) and fits their spectral parameters.
- **Multi-mission follow-up** — cross-matches triggers with the Swift/BAT
  catalog and refines localizations using BAT data.

## Requirements

- Python 3.9+
- `astro-gdt` and `astro-gdt-fermi`
- Optional: GBM response generator for template creation

### Validated dependency stack

The following versions are pinned in `pyproject.toml` and are the combination this
project is tested against:


| Package    | Version |
| ---------- | ------- |
| NumPy      | 1.23.5  |
| pandas     | 2.1.0   |
| matplotlib | 3.9.4   |


Install as usual:

```bash
pip install -e .
```

If you want the resolver to treat these three packages as hard pins while
satisfying the rest of the graph, use the bundled constraints file:

```bash
pip install -e . -c constraints.txt
```

(`constraints.txt` duplicates the same versions as `pyproject.toml` for
`pip -c`.) Create a fresh venv if an old environment already pulled incompatible
versions; avoid loosening the three pins above without re-testing the pipeline.

## External tools (optional)

GBM response generator (used in template creation):

- Reference guide: [https://fermi.gsfc.nasa.gov/ssc/data/analysis/gbm/INSTALL.html](https://fermi.gsfc.nasa.gov/ssc/data/analysis/gbm/INSTALL.html)
- macOS notes: `README` previously embedded a long guide; keep any local notes
in your own docs if you need custom build steps.

## Configuration

Edit `config/config.yaml` to set your data path and detector settings. The
pipeline resolves a relative `data_path` against the config directory.

Key fields:

- `data_path`: location for GBM data and pipeline outputs
- `detectors`, `ndetectors`, `echans`, `DATALEN`, `EPS`
- `search_bank_folder`, `integration_sky_folder`, `integration_spec_folder`

## Usage (library)

```python
from datetime import datetime

from grpype.pipeline_executers import resolve_config, run_pipeline

config = resolve_config()
print(f"Using data path: {config.data_path}")

start = datetime(2017, 8, 17, 12, 41)
run_pipeline(start, delta=1, tte_npar=10)
```

## Usage (CLI)

```
python -m pipeline_executers.run_pipeline 2017 8 17 12 1 --tte-npar 10 --save True
```

If you add the console scripts from `pyproject.toml`, you can also run:

```
gbm-run-pipeline 2017 8 17 12 1 --tte-npar 10 --save True
```

## Examples

See `examples/basic_usage.py` for a minimal entry point.

## Citation

If you use this pipeline in your research, please cite:

```bibtex
@article{perera2025new,
  title={A New Search Pipeline for Short Gamma-Ray Bursts in Fermi/GBM Data—A 50\% Increase in the Number of Detections},
  author={Perera, Ariel and Zackay, Barak and Venumadhav, Tejaswi},
  journal={The Astrophysical Journal Supplement Series},
  volume={281},
  number={1},
  pages={4},
  year={2025},
  publisher={The American Astronomical Society}
}
```

## Acknowledgements

This pipeline uses data products from the
[Fermi Gamma-ray Burst Monitor (GBM)](https://fermi.gsfc.nasa.gov/ssc/data/access/gbm/),
including the official GBM burst catalog and trigger catalogs.

It also makes use of data from the
[Swift Burst Alert Telescope (BAT)](https://swift.gsfc.nasa.gov/about_swift/bat_desc.html),
including the Swift/BAT GRB catalog.

Some data reduction and analysis rely on  
[HEASoft](https://heasarc.gsfc.nasa.gov/docs/software/heasoft/),
a suite of tools provided by NASA's HEASARC.

## License

This project is licensed under the [MIT License](LICENSE).

## Contact

For questions or inquiries about the pipeline, reach out to
Ariel Perera — [ariel.perera@gmail.com](mailto:ariel.perera@gmail.com).
