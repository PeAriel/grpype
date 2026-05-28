from datetime import datetime

from grpype.detection.global_params import (
    INTEGRATION_SKY_FOLDER,
    INTEGRATION_SPEC_FOLDER,
    SEARCH_BANK_FOLDER,
)
from grpype.templates.search_template_bank_gen import gen_rand_band_search_bank
from grpype.templates.sky_template_bank_gen import generate_integration_sky_templates
from grpype.templates.spec_template_bank_gen import generate_integration_spec_templates
from grpype.templates.template_utils import random_placement
from grpype.templates.glitch_template_gen import generate_1d_glitch_templates


def generate_all_templates():
    print('Generating the single detector glitch templates')
    generate_1d_glitch_templates(3)
    
    ref_date = datetime(2021, 9, 27, 0)
    
    method = 'random'
    responses_path = None
    bank_path = SEARCH_BANK_FOLDER
    nsearch_templates = 1000
    print(f'Generating search bank with {nsearch_templates} templates')
    gen_rand_band_search_bank(nsearch_templates, ref_date, method, responses_path, bank_path)

    print('Doing random placement on the search bank')
    print('Warning! The random placement is done with no detection limit amplitudes! This may degrade the performance of the search bank.')
    print('Placing templates with binning 0.01 seconds')
    random_placement(ref_date, 0.01, bank_path, hasamps=False)
    print('Placing templates with binning 0.001 seconds')
    random_placement(ref_date, 0.001, bank_path, hasamps=False)
    print('Done with the search bank')
    
    nangs_sky = 2500
    sky_bank_path = INTEGRATION_SKY_FOLDER
    print(f'Generating integration sky bank with {nangs_sky} angles')
    generate_integration_sky_templates(nangs_sky, ref_date, responses_path, sky_bank_path)

    nangs_spec = 1000
    spec_bank_path = INTEGRATION_SPEC_FOLDER
    print(f'Generating integration spec bank with {nangs_spec} angles')
    generate_integration_spec_templates(nangs_spec, ref_date, responses_path, spec_bank_path)


if __name__ == "__main__":    
    generate_all_templates()
    print('All templates generated successfully')
