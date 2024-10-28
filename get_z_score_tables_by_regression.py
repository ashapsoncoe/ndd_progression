import os
import sys

working_dir = os.path.dirname(__file__)
sys.path.insert(0, working_dir)
os.chdir(working_dir)

import pandas as pd
from copy import deepcopy
import numpy as np
from itertools import product
from statsmodels.regression.linear_model import OLS
from statsmodels.genmod.families import links
from statsmodels.tools.tools import add_constant
from statsmodels.stats.diagnostic import het_breuschpagan, normal_ad
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.stats import norm, probplot, cramervonmises, kstest
from sklearn.preprocessing import PolynomialFeatures
import pickle
import statsmodels.api as sm
from scipy.stats import skew
from scipy.ndimage import gaussian_filter


regression_model_type = 'gaussian' # options: 'gaussian', 'negative binomial','poisson', 'gamma' 
experiment_name = 'age_TIV_centre_grouped_regions_onlyage2_sep_sex'
input_data_file = 'UKBW_EBM_78867r674582_250923v2.csv'
age_at_diagnosis = 'ML_C42C240Xf41270f20002_DementiaAge'
neurological_condition = 'R_NeuroExcludeSRFinalforDemAnalysis'
patient_withdrawn = 'R_ExclusionWithdrawals'
patient_lost_to_fup = 'R_LTFU_f191'
death_date = 'date_of_death_f40000_0_0'
has_alz = 'ML_C42C240Xf41270_Alzheimers'
has_dementia = 'ML_C42C240Xf41270f20002_Dementia'
age_at_scan = 'R_Age_f21003_2'
total_intracranial_volume = 'R_TIV_f25009f25003_2'
sex = 'R_Sex_f31'
centre = 'R_Centre_f54_2' # i.e. MRI scanner
prs_ad_std = 'prs_for_standard_alzheimers_disease_ad_f26206_0_0'                                                     
prs_ad_enh = 'prs_for_enhanced_alzheimers_disease_ad_f26207_0_0'                                                   
prs_pd_std = 'prs_for_standard_parkinsons_disease_pd_f26260_0_0'              
prs_pd_enh = 'prs_for_enhanced_parkinsons_disease_pd_f26261_0_0'

polynomial_features_degree = 2
log_transform_covariates = False
log_transform_dependent_variables = False
essential_plots_only = True
grouped_brain_regions = True
use_pre_diagnosis_scans_only = False


variables_for_separate_models = [
    sex,
]

continous_explanatory_variables = [
    age_at_scan,
    total_intracranial_volume,
    ]

categorical_explanatory_variables = [
    centre,
    ]

features_to_drop = [
    'R_Sex_f31_dv0^2',
    'R_Centre_f54_2_dv0^2',
    'R_Centre_f54_2_dv0 R_Centre_f54_2_dv1', 
    'R_Centre_f54_2_dv1^2',
    'R_TIV_f25009f25003_2 R_Sex_f31_dv0', 
    'R_TIV_f25009f25003_2 R_Centre_f54_2_dv0', 
    'R_TIV_f25009f25003_2 R_Centre_f54_2_dv1', 
    'R_Sex_f31_dv0 R_Centre_f54_2_dv0', 
    'R_Sex_f31_dv0 R_Centre_f54_2_dv1', 
    'R_Age_f21003_2 R_Sex_f31_dv0', 
    'R_Age_f21003_2 R_Centre_f54_2_dv0', 
    'R_Age_f21003_2 R_Centre_f54_2_dv1', 
    'R_Age_f21003_2 R_TIV_f25009f25003_2',
    'R_TIV_f25009f25003_2^2',

]


groupings = {
    'parietal_lobe': [
        'volume_of_grey_matter_in_angular_gyrus_left_f25822_2_0',
        'volume_of_grey_matter_in_angular_gyrus_right_f25823_2_0',
        'volume_of_grey_matter_in_parietal_operculum_cortex_left_f25866_2_0', 
        'volume_of_grey_matter_in_parietal_operculum_cortex_right_f25867_2_0', 
        'volume_of_grey_matter_in_cingulate_gyrus_posterior_division_left_f25840_2_0',
        'volume_of_grey_matter_in_cingulate_gyrus_posterior_division_right_f25841_2_0',
        'volume_of_grey_matter_in_postcentral_gyrus_left_f25814_2_0', 
        'volume_of_grey_matter_in_postcentral_gyrus_right_f25815_2_0', 
        'volume_of_grey_matter_in_precuneous_cortex_left_f25842_2_0', 
        'volume_of_grey_matter_in_precuneous_cortex_right_f25843_2_0', 
        'volume_of_grey_matter_in_superior_parietal_lobule_left_f25816_2_0', 
        'volume_of_grey_matter_in_superior_parietal_lobule_right_f25817_2_0', 
        'volume_of_grey_matter_in_supramarginal_gyrus_anterior_division_left_f25818_2_0', 
        'volume_of_grey_matter_in_supramarginal_gyrus_anterior_division_right_f25819_2_0', 
        'volume_of_grey_matter_in_supramarginal_gyrus_posterior_division_left_f25820_2_0', 
        'volume_of_grey_matter_in_supramarginal_gyrus_posterior_division_right_f25821_2_0', 
    ],

    'occipital_lobe': [
        'volume_of_grey_matter_in_cuneal_cortex_left_f25844_2_0',
        'volume_of_grey_matter_in_cuneal_cortex_right_f25845_2_0',
        'volume_of_grey_matter_in_intracalcarine_cortex_left_f25828_2_0', 
        'volume_of_grey_matter_in_intracalcarine_cortex_right_f25829_2_0', 
        'volume_of_grey_matter_in_lateral_occipital_cortex_inferior_division_left_f25826_2_0', 
        'volume_of_grey_matter_in_lateral_occipital_cortex_inferior_division_right_f25827_2_0', 
        'volume_of_grey_matter_in_lateral_occipital_cortex_superior_division_left_f25824_2_0', 
        'volume_of_grey_matter_in_lateral_occipital_cortex_superior_division_right_f25825_2_0', 
        'volume_of_grey_matter_in_lingual_gyrus_left_f25852_2_0', 
        'volume_of_grey_matter_in_lingual_gyrus_right_f25853_2_0', 
        'volume_of_grey_matter_in_occipital_fusiform_gyrus_left_f25860_2_0', 
        'volume_of_grey_matter_in_occipital_fusiform_gyrus_right_f25861_2_0', 
        'volume_of_grey_matter_in_occipital_pole_left_f25876_2_0', 
        'volume_of_grey_matter_in_occipital_pole_right_f25877_2_0', 
        'volume_of_grey_matter_in_supracalcarine_cortex_left_f25874_2_0', 
        'volume_of_grey_matter_in_supracalcarine_cortex_right_f25875_2_0', 
        'volume_of_grey_matter_in_temporal_occipital_fusiform_cortex_left_f25858_2_0', 
        'volume_of_grey_matter_in_temporal_occipital_fusiform_cortex_right_f25859_2_0', 
    ],

    'frontal_lobe': [
        'volume_of_grey_matter_in_frontal_medial_cortex_left_f25830_2_0',
        'volume_of_grey_matter_in_frontal_medial_cortex_right_f25831_2_0',
        'volume_of_grey_matter_in_frontal_operculum_cortex_left_f25862_2_0',
        'volume_of_grey_matter_in_frontal_operculum_cortex_right_f25863_2_0',
        'volume_of_grey_matter_in_frontal_orbital_cortex_left_f25846_2_0', 
        'volume_of_grey_matter_in_frontal_orbital_cortex_right_f25847_2_0', 
        'volume_of_grey_matter_in_frontal_pole_left_f25782_2_0', 
        'volume_of_grey_matter_in_frontal_pole_right_f25783_2_0', 
        'volume_of_grey_matter_in_inferior_frontal_gyrus_pars_opercularis_left_f25792_2_0', 
        'volume_of_grey_matter_in_inferior_frontal_gyrus_pars_opercularis_right_f25793_2_0', 
        'volume_of_grey_matter_in_inferior_frontal_gyrus_pars_triangularis_left_f25790_2_0', 
        'volume_of_grey_matter_in_inferior_frontal_gyrus_pars_triangularis_right_f25791_2_0', 
        'volume_of_grey_matter_in_juxtapositional_lobule_cortex_formerly_supplementary_motor_cortex_left_f25832_2_0', 
        'volume_of_grey_matter_in_juxtapositional_lobule_cortex_formerly_supplementary_motor_cortex_right_f25833_2_0', 
        'volume_of_grey_matter_in_middle_frontal_gyrus_left_f25788_2_0', 
        'volume_of_grey_matter_in_middle_frontal_gyrus_right_f25789_2_0', 
        'volume_of_grey_matter_in_paracingulate_gyrus_left_f25836_2_0', 
        'volume_of_grey_matter_in_paracingulate_gyrus_right_f25837_2_0', 
        'volume_of_grey_matter_in_cingulate_gyrus_anterior_division_left_f25838_2_0',
        'volume_of_grey_matter_in_cingulate_gyrus_anterior_division_right_f25839_2_0',
        'volume_of_grey_matter_in_precentral_gyrus_left_f25794_2_0', 
        'volume_of_grey_matter_in_precentral_gyrus_right_f25795_2_0', 
        'volume_of_grey_matter_in_subcallosal_cortex_left_f25834_2_0', 
        'volume_of_grey_matter_in_subcallosal_cortex_right_f25835_2_0', 
        'volume_of_grey_matter_in_superior_frontal_gyrus_left_f25786_2_0', 
        'volume_of_grey_matter_in_superior_frontal_gyrus_right_f25787_2_0', 
        'volume_of_grey_matter_in_central_opercular_cortex_left_f25864_2_0', 
        'volume_of_grey_matter_in_central_opercular_cortex_right_f25865_2_0', 
    ],

    'temporal_lobe': [
        'volume_of_grey_matter_in_heschls_gyrus_includes_h1_and_h2_left_f25870_2_0', 
        'volume_of_grey_matter_in_heschls_gyrus_includes_h1_and_h2_right_f25871_2_0', 
        'volume_of_grey_matter_in_inferior_temporal_gyrus_anterior_division_left_f25808_2_0', 
        'volume_of_grey_matter_in_inferior_temporal_gyrus_anterior_division_right_f25809_2_0', 
        'volume_of_grey_matter_in_inferior_temporal_gyrus_posterior_division_left_f25810_2_0', 
        'volume_of_grey_matter_in_inferior_temporal_gyrus_posterior_division_right_f25811_2_0', 
        'volume_of_grey_matter_in_inferior_temporal_gyrus_temporooccipital_part_left_f25812_2_0', 
        'volume_of_grey_matter_in_inferior_temporal_gyrus_temporooccipital_part_right_f25813_2_0', 
        'volume_of_grey_matter_in_middle_temporal_gyrus_anterior_division_left_f25802_2_0', 
        'volume_of_grey_matter_in_middle_temporal_gyrus_anterior_division_right_f25803_2_0', 
        'volume_of_grey_matter_in_middle_temporal_gyrus_posterior_division_left_f25804_2_0', 
        'volume_of_grey_matter_in_middle_temporal_gyrus_posterior_division_right_f25805_2_0', 
        'volume_of_grey_matter_in_middle_temporal_gyrus_temporooccipital_part_left_f25806_2_0', 
        'volume_of_grey_matter_in_middle_temporal_gyrus_temporooccipital_part_right_f25807_2_0', 
        'volume_of_grey_matter_in_parahippocampal_gyrus_anterior_division_left_f25848_2_0', 
        'volume_of_grey_matter_in_parahippocampal_gyrus_anterior_division_right_f25849_2_0', 
        'volume_of_grey_matter_in_parahippocampal_gyrus_posterior_division_left_f25850_2_0', 
        'volume_of_grey_matter_in_parahippocampal_gyrus_posterior_division_right_f25851_2_0', 
        'volume_of_grey_matter_in_planum_polare_left_f25868_2_0', 
        'volume_of_grey_matter_in_planum_polare_right_f25869_2_0', 
        'volume_of_grey_matter_in_planum_temporale_left_f25872_2_0', 
        'volume_of_grey_matter_in_planum_temporale_right_f25873_2_0', 
        'volume_of_grey_matter_in_superior_temporal_gyrus_anterior_division_left_f25798_2_0', 
        'volume_of_grey_matter_in_superior_temporal_gyrus_anterior_division_right_f25799_2_0', 
        'volume_of_grey_matter_in_superior_temporal_gyrus_posterior_division_left_f25800_2_0', 
        'volume_of_grey_matter_in_superior_temporal_gyrus_posterior_division_right_f25801_2_0', 
        'volume_of_grey_matter_in_temporal_fusiform_cortex_anterior_division_left_f25854_2_0', 
        'volume_of_grey_matter_in_temporal_fusiform_cortex_anterior_division_right_f25855_2_0', 
        'volume_of_grey_matter_in_temporal_fusiform_cortex_posterior_division_left_f25856_2_0', 
        'volume_of_grey_matter_in_temporal_fusiform_cortex_posterior_division_right_f25857_2_0', 
        'volume_of_grey_matter_in_temporal_pole_left_f25796_2_0', 
        'volume_of_grey_matter_in_temporal_pole_right_f25797_2_0', 
        
    ],
    'insular_cortex': [
        'volume_of_grey_matter_in_insular_cortex_left_f25784_2_0', 
        'volume_of_grey_matter_in_insular_cortex_right_f25785_2_0', 
    ],

    'hippocampus': [
        'volume_of_hippocampaltail_left_hemisphere_f26620_2_0', 
        'volume_of_hippocampaltail_right_hemisphere_f26642_2_0', 
        'volume_of_wholehippocampalbody_left_hemisphere_f26639_2_0', 
        'volume_of_wholehippocampalbody_right_hemisphere_f26661_2_0', 
        'volume_of_wholehippocampalhead_left_hemisphere_f26640_2_0', 
        'volume_of_wholehippocampalhead_right_hemisphere_f26662_2_0', 
    ],

    'striatum': [
        'volume_of_accumbens_left_f25023_2_0',
        'volume_of_accumbens_right_f25024_2_0',
        'volume_of_caudate_left_f25013_2_0',
        'volume_of_caudate_right_f25014_2_0',
        'volume_of_putamen_left_f25015_2_0',
        'volume_of_putamen_right_f25016_2_0',
    ],

    'globus pallidus': [
        'volume_of_pallidum_left_f25017_2_0',
        'volume_of_pallidum_right_f25018_2_0',
    ],

    'substantia nigra': [
    ],

    'subthalamic nucleus': [
    ],

    'brainstem': [
        'volume_of_grey_matter_in_brainstem_f25892_2_0'
    ],


    'cerebellum':
    [
        'volume_of_grey_matter_in_crus_i_cerebellum_left_f25900_2_0',
        'volume_of_grey_matter_in_crus_i_cerebellum_right_f25902_2_0',
        'volume_of_grey_matter_in_crus_i_cerebellum_vermis_f25901_2_0',
        'volume_of_grey_matter_in_crus_ii_cerebellum_left_f25903_2_0',
        'volume_of_grey_matter_in_crus_ii_cerebellum_right_f25905_2_0',
        'volume_of_grey_matter_in_crus_ii_cerebellum_vermis_f25904_2_0',
        'volume_of_grey_matter_in_iiv_cerebellum_left_f25893_2_0',
        'volume_of_grey_matter_in_iiv_cerebellum_right_f25894_2_0',
        'volume_of_grey_matter_in_ix_cerebellum_left_f25915_2_0',
        'volume_of_grey_matter_in_ix_cerebellum_right_f25917_2_0',
        'volume_of_grey_matter_in_ix_cerebellum_vermis_f25916_2_0',
        'volume_of_grey_matter_in_v_cerebellum_left_f25895_2_0',
        'volume_of_grey_matter_in_v_cerebellum_right_f25896_2_0',
        'volume_of_grey_matter_in_vi_cerebellum_left_f25897_2_0',
        'volume_of_grey_matter_in_vi_cerebellum_right_f25899_2_0',
        'volume_of_grey_matter_in_vi_cerebellum_vermis_f25898_2_0',
        'volume_of_grey_matter_in_viiia_cerebellum_left_f25909_2_0',
        'volume_of_grey_matter_in_viiia_cerebellum_right_f25911_2_0',
        'volume_of_grey_matter_in_viiia_cerebellum_vermis_f25910_2_0',
        'volume_of_grey_matter_in_viiib_cerebellum_left_f25912_2_0',
        'volume_of_grey_matter_in_viiib_cerebellum_right_f25914_2_0',
        'volume_of_grey_matter_in_viiib_cerebellum_vermis_f25913_2_0',
        'volume_of_grey_matter_in_viib_cerebellum_left_f25906_2_0',
        'volume_of_grey_matter_in_viib_cerebellum_right_f25908_2_0',
        'volume_of_grey_matter_in_viib_cerebellum_vermis_f25907_2_0',
        'volume_of_grey_matter_in_x_cerebellum_left_f25918_2_0',
        'volume_of_grey_matter_in_x_cerebellum_vermis_f25919_2_0',
        'volume_of_grey_matter_in_x_cerebellum_right_f25920_2_0',
    ],

    'thalamus':
    [
        'volume_of_thalamus_left_f25011_2_0',
        'volume_of_thalamus_right_f25012_2_0',
    ],

    'amygdala':
    [    
        'volume_of_amygdala_left_f25021_2_0',
        'volume_of_amygdala_right_f25022_2_0',
    ]
    
}



brain_regions_to_use = [
    'volume_of_grey_matter_in_angular_gyrus_left_f25822_2_0',
    'volume_of_grey_matter_in_angular_gyrus_right_f25823_2_0',
    'volume_of_grey_matter_in_central_opercular_cortex_left_f25864_2_0', 
    'volume_of_grey_matter_in_central_opercular_cortex_right_f25865_2_0', 
    'volume_of_grey_matter_in_cingulate_gyrus_anterior_division_left_f25838_2_0',
    'volume_of_grey_matter_in_cingulate_gyrus_anterior_division_right_f25839_2_0',
    'volume_of_grey_matter_in_cingulate_gyrus_posterior_division_left_f25840_2_0',
    'volume_of_grey_matter_in_cingulate_gyrus_posterior_division_right_f25841_2_0',
    'volume_of_grey_matter_in_cuneal_cortex_left_f25844_2_0',
    'volume_of_grey_matter_in_cuneal_cortex_right_f25845_2_0',
    'volume_of_grey_matter_in_frontal_medial_cortex_left_f25830_2_0',
    'volume_of_grey_matter_in_frontal_medial_cortex_right_f25831_2_0',
    'volume_of_grey_matter_in_frontal_operculum_cortex_left_f25862_2_0',
    'volume_of_grey_matter_in_frontal_operculum_cortex_right_f25863_2_0',
    'volume_of_grey_matter_in_frontal_orbital_cortex_left_f25846_2_0', 
    'volume_of_grey_matter_in_frontal_orbital_cortex_right_f25847_2_0', 
    'volume_of_grey_matter_in_frontal_pole_left_f25782_2_0', 
    'volume_of_grey_matter_in_frontal_pole_right_f25783_2_0', 
    'volume_of_grey_matter_in_heschls_gyrus_includes_h1_and_h2_left_f25870_2_0', 
    'volume_of_grey_matter_in_heschls_gyrus_includes_h1_and_h2_right_f25871_2_0', 
    'volume_of_grey_matter_in_inferior_frontal_gyrus_pars_opercularis_left_f25792_2_0', 
    'volume_of_grey_matter_in_inferior_frontal_gyrus_pars_opercularis_right_f25793_2_0', 
    'volume_of_grey_matter_in_inferior_frontal_gyrus_pars_triangularis_left_f25790_2_0', 
    'volume_of_grey_matter_in_inferior_frontal_gyrus_pars_triangularis_right_f25791_2_0', 
    'volume_of_grey_matter_in_inferior_temporal_gyrus_anterior_division_left_f25808_2_0', 
    'volume_of_grey_matter_in_inferior_temporal_gyrus_anterior_division_right_f25809_2_0', 
    'volume_of_grey_matter_in_inferior_temporal_gyrus_posterior_division_left_f25810_2_0', 
    'volume_of_grey_matter_in_inferior_temporal_gyrus_posterior_division_right_f25811_2_0', 
    'volume_of_grey_matter_in_inferior_temporal_gyrus_temporooccipital_part_left_f25812_2_0', 
    'volume_of_grey_matter_in_inferior_temporal_gyrus_temporooccipital_part_right_f25813_2_0', 
    'volume_of_grey_matter_in_insular_cortex_left_f25784_2_0', 
    'volume_of_grey_matter_in_insular_cortex_right_f25785_2_0', 
    'volume_of_grey_matter_in_intracalcarine_cortex_left_f25828_2_0', 
    'volume_of_grey_matter_in_intracalcarine_cortex_right_f25829_2_0', 
    'volume_of_grey_matter_in_juxtapositional_lobule_cortex_formerly_supplementary_motor_cortex_left_f25832_2_0', 
    'volume_of_grey_matter_in_juxtapositional_lobule_cortex_formerly_supplementary_motor_cortex_right_f25833_2_0', 
    'volume_of_grey_matter_in_lateral_occipital_cortex_inferior_division_left_f25826_2_0', 
    'volume_of_grey_matter_in_lateral_occipital_cortex_inferior_division_right_f25827_2_0', 
    'volume_of_grey_matter_in_lateral_occipital_cortex_superior_division_left_f25824_2_0', 
    'volume_of_grey_matter_in_lateral_occipital_cortex_superior_division_right_f25825_2_0', 
    'volume_of_grey_matter_in_lingual_gyrus_left_f25852_2_0', 
    'volume_of_grey_matter_in_lingual_gyrus_right_f25853_2_0', 
    'volume_of_grey_matter_in_middle_frontal_gyrus_left_f25788_2_0', 
    'volume_of_grey_matter_in_middle_frontal_gyrus_right_f25789_2_0', 
    'volume_of_grey_matter_in_middle_temporal_gyrus_anterior_division_left_f25802_2_0', 
    'volume_of_grey_matter_in_middle_temporal_gyrus_anterior_division_right_f25803_2_0', 
    'volume_of_grey_matter_in_middle_temporal_gyrus_posterior_division_left_f25804_2_0', 
    'volume_of_grey_matter_in_middle_temporal_gyrus_posterior_division_right_f25805_2_0', 
    'volume_of_grey_matter_in_middle_temporal_gyrus_temporooccipital_part_left_f25806_2_0', 
    'volume_of_grey_matter_in_middle_temporal_gyrus_temporooccipital_part_right_f25807_2_0', 
    'volume_of_grey_matter_in_occipital_fusiform_gyrus_left_f25860_2_0', 
    'volume_of_grey_matter_in_occipital_fusiform_gyrus_right_f25861_2_0', 
    'volume_of_grey_matter_in_occipital_pole_left_f25876_2_0', 
    'volume_of_grey_matter_in_occipital_pole_right_f25877_2_0', 
    'volume_of_grey_matter_in_paracingulate_gyrus_left_f25836_2_0', 
    'volume_of_grey_matter_in_paracingulate_gyrus_right_f25837_2_0', 
    'volume_of_grey_matter_in_parahippocampal_gyrus_anterior_division_left_f25848_2_0', 
    'volume_of_grey_matter_in_parahippocampal_gyrus_anterior_division_right_f25849_2_0', 
    'volume_of_grey_matter_in_parahippocampal_gyrus_posterior_division_left_f25850_2_0', 
    'volume_of_grey_matter_in_parahippocampal_gyrus_posterior_division_right_f25851_2_0', 
    'volume_of_grey_matter_in_parietal_operculum_cortex_left_f25866_2_0', 
    'volume_of_grey_matter_in_parietal_operculum_cortex_right_f25867_2_0', 
    'volume_of_grey_matter_in_planum_polare_left_f25868_2_0', 
    'volume_of_grey_matter_in_planum_polare_right_f25869_2_0', 
    'volume_of_grey_matter_in_planum_temporale_left_f25872_2_0', 
    'volume_of_grey_matter_in_planum_temporale_right_f25873_2_0', 
    'volume_of_grey_matter_in_postcentral_gyrus_left_f25814_2_0', 
    'volume_of_grey_matter_in_postcentral_gyrus_right_f25815_2_0', 
    'volume_of_grey_matter_in_precentral_gyrus_left_f25794_2_0', 
    'volume_of_grey_matter_in_precentral_gyrus_right_f25795_2_0', 
    'volume_of_grey_matter_in_precuneous_cortex_left_f25842_2_0', 
    'volume_of_grey_matter_in_precuneous_cortex_right_f25843_2_0', 
    'volume_of_grey_matter_in_subcallosal_cortex_left_f25834_2_0', 
    'volume_of_grey_matter_in_subcallosal_cortex_right_f25835_2_0', 
    'volume_of_grey_matter_in_superior_frontal_gyrus_left_f25786_2_0', 
    'volume_of_grey_matter_in_superior_frontal_gyrus_right_f25787_2_0', 
    'volume_of_grey_matter_in_superior_parietal_lobule_left_f25816_2_0', 
    'volume_of_grey_matter_in_superior_parietal_lobule_right_f25817_2_0', 
    'volume_of_grey_matter_in_superior_temporal_gyrus_anterior_division_left_f25798_2_0', 
    'volume_of_grey_matter_in_superior_temporal_gyrus_anterior_division_right_f25799_2_0', 
    'volume_of_grey_matter_in_superior_temporal_gyrus_posterior_division_left_f25800_2_0', 
    'volume_of_grey_matter_in_superior_temporal_gyrus_posterior_division_right_f25801_2_0', 
    'volume_of_grey_matter_in_supracalcarine_cortex_left_f25874_2_0', 
    'volume_of_grey_matter_in_supracalcarine_cortex_right_f25875_2_0', 
    'volume_of_grey_matter_in_supramarginal_gyrus_anterior_division_left_f25818_2_0', 
    'volume_of_grey_matter_in_supramarginal_gyrus_anterior_division_right_f25819_2_0', 
    'volume_of_grey_matter_in_supramarginal_gyrus_posterior_division_left_f25820_2_0', 
    'volume_of_grey_matter_in_supramarginal_gyrus_posterior_division_right_f25821_2_0', 
    'volume_of_grey_matter_in_temporal_fusiform_cortex_anterior_division_left_f25854_2_0', 
    'volume_of_grey_matter_in_temporal_fusiform_cortex_anterior_division_right_f25855_2_0', 
    'volume_of_grey_matter_in_temporal_fusiform_cortex_posterior_division_left_f25856_2_0', 
    'volume_of_grey_matter_in_temporal_fusiform_cortex_posterior_division_right_f25857_2_0', 
    'volume_of_grey_matter_in_temporal_occipital_fusiform_cortex_left_f25858_2_0', 
    'volume_of_grey_matter_in_temporal_occipital_fusiform_cortex_right_f25859_2_0', 
    'volume_of_grey_matter_in_temporal_pole_left_f25796_2_0', 
    'volume_of_grey_matter_in_temporal_pole_right_f25797_2_0', 
    'volume_of_gcmldgbody_left_hemisphere_f26633_2_0',
    'volume_of_gcmldgbody_right_hemisphere_f26655_2_0',
    'volume_of_gcmldghead_left_hemisphere_f26631_2_0',
    'volume_of_gcmldghead_right_hemisphere_f26653_2_0',
    'volume_of_ca1body_left_hemisphere_f26622_2_0', 
    'volume_of_ca1body_right_hemisphere_f26644_2_0', 
    'volume_of_ca1head_left_hemisphere_f26626_2_0', 
    'volume_of_ca1head_right_hemisphere_f26648_2_0', 
    'volume_of_ca3body_left_hemisphere_f26632_2_0', 
    'volume_of_ca3body_right_hemisphere_f26654_2_0', 
    'volume_of_ca3head_left_hemisphere_f26637_2_0', 
    'volume_of_ca3head_right_hemisphere_f26659_2_0', 
    'volume_of_ca4body_left_hemisphere_f26635_2_0', 
    'volume_of_ca4body_right_hemisphere_f26657_2_0', 
    'volume_of_ca4head_left_hemisphere_f26634_2_0', 
    'volume_of_ca4head_right_hemisphere_f26656_2_0', 
    'volume_of_hippocampaltail_left_hemisphere_f26620_2_0', 
    'volume_of_hippocampaltail_right_hemisphere_f26642_2_0', 
    'volume_of_molecularlayerhpbody_left_hemisphere_f26630_2_0', 
    'volume_of_molecularlayerhpbody_right_hemisphere_f26652_2_0', 
    'volume_of_molecularlayerhphead_left_hemisphere_f26629_2_0', 
    'volume_of_molecularlayerhphead_right_hemisphere_f26651_2_0', 
    'volume_of_parasubiculum_left_hemisphere_f26628_2_0', 
    'volume_of_parasubiculum_right_hemisphere_f26650_2_0', 
    'volume_of_presubiculumbody_left_hemisphere_f26627_2_0', 
    'volume_of_presubiculumbody_right_hemisphere_f26649_2_0', 
    'volume_of_presubiculumhead_left_hemisphere_f26625_2_0', 
    'volume_of_presubiculumhead_right_hemisphere_f26647_2_0', 
    'volume_of_subiculumbody_left_hemisphere_f26621_2_0', 
    'volume_of_subiculumbody_right_hemisphere_f26643_2_0', 
    'volume_of_subiculumhead_left_hemisphere_f26623_2_0', 
    'volume_of_subiculumhead_right_hemisphere_f26645_2_0', 
    'volume_of_accumbens_left_f25023_2_0',
    'volume_of_accumbens_right_f25024_2_0',
    'volume_of_amygdala_left_f25021_2_0',
    'volume_of_amygdala_right_f25022_2_0',
    'volume_of_caudate_left_f25013_2_0',
    'volume_of_caudate_right_f25014_2_0',
    'volume_of_pallidum_left_f25017_2_0',
    'volume_of_pallidum_right_f25018_2_0',
    'volume_of_putamen_left_f25015_2_0',
    'volume_of_putamen_right_f25016_2_0',
    'volume_of_thalamus_left_f25011_2_0',
    'volume_of_thalamus_right_f25012_2_0',
]





def add_dummy_variables(df_to_use, categorical_explanatory_variables):

    dummy_lookup = {}
    dv_col_names = []

    for col in categorical_explanatory_variables:

        all_vals = set(df_to_use[col])

        num_dvs = len(all_vals)-1

        dv_combos = [x for x in product([0,1], repeat=num_dvs) if sum(x)<=1] 

        dummy_lookup[col] = {val: dv_combo for dv_combo, val in zip(dv_combos, all_vals)}

        for pos in range(num_dvs):

            df_to_use[f'{col}_dv{pos}'] = np.array([dummy_lookup[col][df_to_use.at[i, col]][pos] for i in df_to_use.index])

            dv_col_names.append(f'{col}_dv{pos}')

    return dummy_lookup, dv_col_names

def plot_and_save_scatter(xs, ys, xlab, ylab, save_path, fname, draw_xy_line=False, draw_horizonal_line=False, s=0.1, title=None):

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    plt.clf()

    if draw_xy_line:
        max_overall_xy = max([max(xs), max(ys)])
        tmp = plt.plot([0, max_overall_xy], [0, max_overall_xy], ls="--", c=".3")

    if draw_horizonal_line:
        max_x = max(xs)
        tmp = plt.plot([0, max_x], [0, 0], ls="--", c=".3")

    tmp = plt.scatter(xs, ys, s=s)
    tmp = plt.xlim([min(xs), max(xs)])
    tmp = plt.ylim([min(ys), max(ys)])
    tmp = plt.xlabel(xlab)
    tmp = plt.ylabel(ylab)

    if title != None:
        tmp = plt.title(title)

    tmp = plt.savefig(f'{save_path}\\{fname}')

def chunk_values(n_inc, vals1, vals2=[]):

    max_val = max(vals1)
    min_val = min(vals1)
    window = (max_val-min_val)/n_inc
    chosen_range = np.arange(min_val, max_val, window)

    if len(vals2) == len(vals1):
        chunked_values = [[(x,y) for x,y in zip(vals1, vals2) if start+window > x >= start] for start in chosen_range]
    else:
        chunked_values = [[x for x in vals1 if start+window > x >= start] for start in chosen_range]
        
    return chunked_values, min_val, max_val, window, chosen_range

def plot_dep_vs_ind(save_dir, brain_regions_to_use, all_covariates, controls_df):

    tmp = plt.figure(figsize=(20, 10))

    if not os.path.exists(f'{save_dir}\\dependent_vs_independent_variables'):
        os.mkdir(f'{save_dir}\\dependent_vs_independent_variables')

    for brain_region in brain_regions_to_use:

        # fitted_model = fitted_models[combo_type][brain_region]['model']
        # covar_means = np.mean(fitted_model.model.exog, axis=0)

        for con_v in all_covariates:

            #pos = fitted_model.model.exog_names.index(con_v)
            save_path_dvip = f"{save_dir}\\dependent_vs_independent_variables\\{con_v}"
            f_name = f"{con_v}_vs_{brain_region}_temp.png"
            plot_and_save_scatter(controls_df[con_v], controls_df[brain_region], con_v, brain_region, save_path_dvip, f_name)

def test_interactions_statistically(brain_regions_to_use, controls_df, all_covariates, save_dir, return_new_terms = False):

    int_df = pd.DataFrame(columns=[], index=brain_regions_to_use)

    for brain_region in brain_regions_to_use:

        Y = controls_df[brain_region]

        for covar1 in all_covariates:

            for covar2 in all_covariates[all_covariates.index(covar1)+1:]:

                col_name = f'{covar1}_x_{covar2}'

                if col_name not in int_df.columns:
                    int_df[col_name] = None

                controls_df[col_name] = controls_df[covar1] * controls_df[covar2]

                X_without_combo = add_constant(controls_df[[covar1, covar2]])
                X_with_combo = add_constant(controls_df[[covar1, covar2, col_name]])
                fitted_without_combo = OLS(Y, X_without_combo).fit() #, cov_type='HC3').fit()
                fitted_with_combo = OLS(Y, X_with_combo).fit() #, cov_type='HC3').fit()

                ts, p_val = fitted_with_combo.compare_lr_test(fitted_without_combo)[:2]

                if ts == -0.0:
                    p_val = 1.0

                int_df.loc[brain_region, col_name] = p_val

    if not os.path.exists(f'{save_dir}\\interaction_LR_test_p_val_tables.csv'):
        int_df.to_csv(f'{save_dir}\\interaction_LR_test_p_val_tables.csv')

    interaction_terms_to_keep = [x for x in int_df.columns if np.mean(int_df[x]) < 0.05]

    if return_new_terms:
        return interaction_terms_to_keep

def plot_interactions_func(controls_df, brain_regions_to_use, save_dir, categorical_explanatory_variables, continous_explanatory_variables, cit_chunk = 5, con_var_bucket = 20):

    categorical_explanatory_variables = list(set(categorical_explanatory_variables))

    tmp = plt.figure(figsize=(20, 10))

    if not os.path.exists(f'{save_dir}\\interaction_plots'):
        os.mkdir(f'{save_dir}\\interaction_plots')

    for brain_region in brain_regions_to_use:

        for candidate_interaction_term in categorical_explanatory_variables+continous_explanatory_variables:

            if not os.path.exists(f'{save_dir}\\interaction_plots\\{candidate_interaction_term}'):
                os.mkdir(f'{save_dir}\\interaction_plots\\{candidate_interaction_term}')

            for covar in continous_explanatory_variables:

                plt.clf()

                if not os.path.exists(f'{save_dir}\\interaction_plots\\{candidate_interaction_term}\\{covar}'):
                    os.mkdir(f'{save_dir}\\interaction_plots\\{candidate_interaction_term}\\{covar}')

                if candidate_interaction_term == covar: continue

                if candidate_interaction_term in categorical_explanatory_variables:
                    levels = set(controls_df[candidate_interaction_term])
                else:
                    window, levels = chunk_values(cit_chunk, controls_df[candidate_interaction_term])[3:]

                legend_labels = []

                for val in levels:

                    if candidate_interaction_term in categorical_explanatory_variables:
                        val_df = controls_df[controls_df[candidate_interaction_term] == val]
                    else:
                        val_df = controls_df[controls_df[candidate_interaction_term] >= val]
                        val_df = val_df[val_df[candidate_interaction_term] < val+window]
                        
                    if len(val_df) == 0: continue

                    buckets = chunk_values(con_var_bucket, val_df[covar], vals2 = val_df[brain_region])[0]

                    x_mean = [np.mean([a[0] for a in b]) for b in buckets if b != []]
                    y_mean = [np.mean([a[1] for a in b]) for b in buckets if b != []]

                    pl = plt.plot(x_mean, y_mean)

                    if candidate_interaction_term in categorical_explanatory_variables:
                        legend_labels.append(val)
                    else:
                        legend_labels.append(f'{round(val, 2)} to {round(val+window, 2)}')
                    
                lg = plt.legend(legend_labels, loc='upper left')
                xlab = plt.xlabel(covar)
                ylab = plt.ylabel(brain_region)
                tl = plt.title(f'Interaction of {candidate_interaction_term} with {covar} , {brain_region} relationship')
                sv = plt.savefig(f'{save_dir}\\interaction_plots\\{candidate_interaction_term}\\{covar}\\{brain_region}.png')

def plot_multicolinearity(controls_df, all_covariates, save_dir):

    X = controls_df[all_covariates]
    tmp = plt.figure(figsize=(20, 10))
    tmp = sns.heatmap(X.corr(),vmin=-1,annot= True)
    tmp = plt.savefig(f'{save_dir}\\independent_variables_correlation_heatmap.png')

def measure_vif(controls_df, all_covariates):    

    # 1= Not correlated,  1-5 = Moderately correlated, >5 = Highly correlated

    X = controls_df[all_covariates]
    vif_per_cov = [variance_inflation_factor(controls_df[all_covariates].values, i) for i in range(X.shape[1])]
    vif_per_cov = {a: vif for a, vif in zip(all_covariates, vif_per_cov)}

    return vif_per_cov

def plot_mvr_each_y(brain_regions_to_use, save_dir, controls_df, num_chunks=100, var_calc_min_chunk = 100, var_smoothing_sigma=0):

    save_path_mvr = f"{save_dir}\\mean_variance_plots"

    if not os.path.exists(save_path_mvr):
        os.mkdir(save_path_mvr)

    for brain_region in brain_regions_to_use:

        Y = controls_df[brain_region]

        chunked_values = chunk_values(num_chunks, Y)[0]
        mean_vals = [np.mean(a) for a in chunked_values if len(a)>var_calc_min_chunk]
        variances = [np.var(a) for a in chunked_values if len(a)>var_calc_min_chunk]

        counts = [len(a) for a in chunked_values if len(a)>var_calc_min_chunk]

        plot_mvr(mean_vals, variances, counts, save_path_mvr, brain_region, var_smoothing_sigma=var_smoothing_sigma)

def plot_mvr(mean_vals, variances, counts, save_path_mvr, brain_region, f_name=None, var_smoothing_sigma=0):

    counts = gaussian_filter(counts, var_smoothing_sigma)
    variances = gaussian_filter(variances, var_smoothing_sigma)

    fig,ax = plt.subplots(figsize=(20, 10))
    ax2=ax.twinx()

    ax.plot(mean_vals, counts, color="red", marker="o")
    ax.set_xlabel("Mean", fontsize = 14)
    ax.set_ylabel("Bin Count", color="red", fontsize=14)

    ax2.plot(mean_vals, variances, color="blue", marker="o")
    ax2.set_ylabel("Variance", color="blue", fontsize=14)
    
    if f_name == None:
        save_name = brain_region
    else:
        save_name = f_name

    fig.savefig(f"{save_path_mvr}\\{save_name}.png")

    plt.close(fig)

def plot_actual_vs_pred(save_dir, brain_regions_to_use, controls_df, fitted_model_dict, all_covariates):

    save_path_ap = f'{save_dir}\\actual_vs_predicted_Y'

    for brain_region in brain_regions_to_use:

        Y = controls_df[brain_region]

        poly_degree = fitted_model_dict[brain_region]['degree']
        fitted_model = fitted_model_dict[brain_region]['model']

        polynomial_features = PolynomialFeatures(degree=poly_degree)
        Xt = polynomial_features.fit_transform(controls_df[all_covariates])
        feature_names = polynomial_features.get_feature_names_out(controls_df[all_covariates].columns)
        X_with_constant = pd.DataFrame(Xt, columns = feature_names)

        X_with_constant = X_with_constant[[x for x in feature_names if x not in features_to_drop]]
        pred_Y = fitted_model.get_prediction(X_with_constant).predicted_mean
        f_name = f"{brain_region}_actual_vs_predicted.png"
        plot_and_save_scatter(Y, pred_Y, f'Actual {brain_region}', f'Predicted {brain_region}', save_path_ap, f_name, draw_xy_line=True)

def fit_and_save_models_for_all_regions(brain_regions_to_use, controls_df, regression_model_type):

    log_link_inst = links.Log()

    glm_model_lookup = {
        'gamma': sm.families.Gamma(link=log_link_inst),
        'binomial': sm.families.Binomial(),
        'gaussian': sm.families.Gaussian(),
        'negative binomial': sm.families.NegativeBinomial(),
        'poisson': sm.families.Poisson(),
    }

    fitted_model_dict = {}
    model_type = glm_model_lookup[regression_model_type]

    for brain_region in brain_regions_to_use:
        
        Y = controls_df[brain_region].reset_index(drop=True)

        polynomial_features = PolynomialFeatures(polynomial_features_degree)
        Xt = polynomial_features.fit_transform(controls_df[all_covariates])
        feature_names = polynomial_features.get_feature_names_out(controls_df[all_covariates].columns)

        X = pd.DataFrame(Xt, columns = feature_names)
        X = X[[x for x in feature_names if x not in features_to_drop]]

        fitted_model = sm.GLM(Y, X, family=model_type).fit(cov_type='hc3') 
        fitted_model_dict[brain_region] = {'model': fitted_model, 'degree': polynomial_features_degree}
            
    return fitted_model_dict

def plot_residuals_vs_fitted_y_and_covariates(brain_regions_to_use, fitted_model_dict, save_dir, controls_df, all_covariates, combo_type): #, chunks = 100, var_smoothing_sigma=0):

    cat_var = combo_type.split('_')[-1]

    save_path_resid = f'{save_dir}\\residual_plots'

    if not os.path.exists(save_path_resid):
        os.mkdir(save_path_resid)

    for brain_region in brain_regions_to_use:
        
        poly_degree = fitted_model_dict[brain_region]['degree']
        fitted_model = fitted_model_dict[brain_region]['model']

        polynomial_features = PolynomialFeatures(degree=poly_degree)
        Xt = polynomial_features.fit_transform(controls_df[all_covariates])
        feature_names = polynomial_features.get_feature_names_out(controls_df[all_covariates].columns)
        X_with_constant = pd.DataFrame(Xt, columns = feature_names)

        X_with_constant = X_with_constant[[x for x in feature_names if x not in features_to_drop]]

        pred_Y = fitted_model.get_prediction(X_with_constant).predicted_mean

        resid_types = {
            'response': fitted_model.resid_response,
        }

        for var in continous_explanatory_variables + ['Expected value of subregion volume (cubic millimetres)']:
    
            if var == 'Expected value of subregion volume (cubic millimetres)':
                to_plot = pred_Y
            else:
                to_plot = controls_df[var]

            for resid_type in resid_types:

                # Plot residual scatter plot 
                f_name = f"{brain_region}_{var}_vs_{resid_type}_residuals.png"
                title = cat_var + ' ' + brain_region.replace('_', ' ')
                plot_and_save_scatter(to_plot, resid_types[resid_type], var, 'Residuals', save_path_resid, f_name, draw_horizonal_line=True, title=title)

                # Plot average residual for each unique value of to_plot:
                coo = np.array([[x] for x in to_plot])
                values = np.array(fitted_model.resid_response)
                sortidx = np.lexsort(coo.T)
                sorted_coo =  coo[sortidx]
                unqID_mask = np.append(True,np.any(np.diff(sorted_coo,axis=0),axis=1))
                id1 = unqID_mask.cumsum()-1
                unique_vals = [x[0] for x in sorted_coo[unqID_mask]]
                resid_means = np.bincount(id1,values[sortidx])/np.bincount(id1)
                n = np.bincount(id1)

                f_name = f"{brain_region}_{var}_vs_mean_of_residual.png"
                plot_and_save_scatter(unique_vals, resid_means, var, f'Mean {resid_type} residual', save_path_resid, f_name, draw_horizonal_line=True, s=[x/10 for x in n])

def perform_normality_tests_and_plots(brain_regions_to_use, fitted_model_dict, save_dir):

    cvm_criterions = {}
    ks_res_pvals = {}
    ad_pvals = {}
    skews = {}


    for brain_region in brain_regions_to_use:

        fitted_model = fitted_model_dict[brain_region]['model']

        # Generate and record normality test results:
        mu, std = norm.fit(fitted_model.resid_response)
        cvm_statistic = cramervonmises(fitted_model.resid_response, 'norm', args = (mu, std)).statistic
        cvm_criterions[brain_region] = cvm_statistic / len(fitted_model.resid_response)
        ks_res_pvals[brain_region] = kstest(fitted_model.resid_response, 'norm', args = (mu, std)).pvalue
        ad_pvals[brain_region] = normal_ad(fitted_model.resid_response)[1]
        skews[brain_region] = skew(fitted_model.resid_response)

        if not essential_plots_only:

            # Plot vs best fitting normal distribution:
            if not os.path.exists(f'{save_dir}\\normality_plots'):
                os.mkdir(f'{save_dir}\\normality_plots')

            plt.clf()
            tmp = plt.hist(fitted_model.resid_response, bins=100, density=True, alpha=0.6, color='g')
            xmin, xmax = plt.xlim()
            x = np.linspace(xmin, xmax, 100)
            p = norm.pdf(x, mu, std)
            tmp = plt.plot(x, p, 'k', linewidth=2)
            tmp = plt.title(f"Fit results: mu = {mu},  std = {std}. A-D test P-Value: {ad_pvals[brain_region]}")
            tmp = plt.xlabel('Residual')
            tmp = plt.ylabel('Probability density')
            tmp = plt.savefig(f'{save_dir}\\normality_plots\\{brain_region}_residuals_distribution.png')

            # Plot QQ plot:
            plt.clf() 
            probplot(fitted_model.resid_response, dist="norm", plot= plt)
            tmp = plt.title("MODEL Residuals Q-Q Plot")
            tmp = plt.legend(['Actual','Theoretical'])
            tmp = plt.savefig(f'{save_dir}\\normality_plots\\{brain_region}_QQ_plot.png')

    return cvm_criterions, ks_res_pvals, ad_pvals, skews

def hetsked_test_all_regions(brain_regions_to_use, fitted_model_dict):

    test_res = {}

    for brain_region in brain_regions_to_use:

        fitted_model = fitted_model_dict[brain_region]['model']

        pval = het_breuschpagan(fitted_model.resid_response, fitted_model.model.exog)[1]
        test_res[brain_region] = pval

    return test_res

def make_save_dir(save_dir, experiment_name, regression_model_type, polynomial_features_degree):

    if '\\\\?\\' not in save_dir:
        save_dir = '\\\\?\\' + save_dir

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    experiment_name = f'GLM_{regression_model_type}_max_deg_{polynomial_features_degree}_{experiment_name}'

    save_dir = f'{save_dir}\\{experiment_name}'

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    return save_dir

def get_separate_dfs_for_each_combo(variables_for_separate_models, controls_df):

    df_subsets = {}

    if variables_for_separate_models == []:

        df_subsets['all_data'] = controls_df

    else:

        sep_var_lookup = [[f'{x}__{a}' for a in set(controls_df[x])] for x in variables_for_separate_models]

        for combo in product(*sep_var_lookup):

            combo_df = deepcopy(controls_df)

            key_name = ','.join(combo)

            for var in combo:

                cat, val = var.split('__')

                if combo_df[cat].dtype.name != 'string':
                    original_type = deepcopy(combo_df[cat].dtype.name)
                    combo_df[cat] = combo_df[cat].astype("string")
                    combo_df = combo_df[combo_df[cat]==val]
                    combo_df[cat] = combo_df[cat].astype(original_type)
                else:
                    combo_df = combo_df[combo_df[cat]==val]
            
            if len(combo_df) > 0 :
                df_subsets[key_name] = combo_df

    return df_subsets

def calc_mean_and_sd_directly(continous_explanatory_variables, brain_regions_to_use, control_df_subsets):

    chosen_variable = continous_explanatory_variables[0]

    mean_and_sd_final = {x: {} for x in control_df_subsets}

    for combo_type in control_df_subsets:

        subset_df = control_df_subsets[combo_type]

        mean_and_sd_final[combo_type][chosen_variable] = {}

        for cv in set(subset_df[chosen_variable]):

            temp_df = subset_df[subset_df[chosen_variable]==cv]

            mean_and_sd_final[combo_type][chosen_variable][cv] = {}

            for brain_region in brain_regions_to_use:

                mean_and_sd_final[combo_type][chosen_variable][cv][brain_region] = {
                    'mean': float(np.mean(temp_df[brain_region])),
                    'std': float(np.std(temp_df[brain_region])),
                }

    return mean_and_sd_final

def get_std_per_observation(std_calc_min_chunk, num_chunks, observed_vals):

    chunked_values = chunk_values(num_chunks, observed_vals)[0]
    chunks_to_use = [x for x in chunked_values if len(x)>=std_calc_min_chunk]
    ave_std_of_chunks = np.mean([np.std(chunk) for chunk in chunks_to_use])
    overall_std = np.std(observed_vals)

    std_lookup = {}

    for chunk in chunks_to_use:
        std_lookup.update({i: (np.std(chunk)/ave_std_of_chunks)*overall_std for i in chunk})

    a = np.array(list(std_lookup.keys()))

    for y in observed_vals:

        if y not in std_lookup:
            
            closest_val = a[np.abs(a-y).argmin()]

            std_lookup[y] = std_lookup[closest_val]

    return std_lookup

def get_z_scores(input_df, brain_regions_to_use, fitted_models, all_covariates):

    z_scores_all_individuals = {x: {} for x in brain_regions_to_use}

    df_subsets = get_separate_dfs_for_each_combo(variables_for_separate_models, input_df)

    for combo_type in df_subsets:

        for brain_region in brain_regions_to_use:

            deg = fitted_models[combo_type][brain_region]['degree']
            polynomial_features = PolynomialFeatures(degree=deg)
            Xt = polynomial_features.fit_transform(df_subsets[combo_type][all_covariates])
            feature_names = polynomial_features.get_feature_names_out(df_subsets[combo_type][all_covariates].columns)
            X_with_constant = pd.DataFrame(Xt, columns = feature_names)
            X_with_constant = X_with_constant[[x for x in feature_names if x not in features_to_drop]]


            pred_vals = fitted_models[combo_type][brain_region]['model'].get_prediction(X_with_constant).predicted_mean
            observed_vals = df_subsets[combo_type][brain_region]
            common_control_sd = np.std(fitted_models[combo_type][brain_region]['model'].resid_response)

            #common_control_sd = np.std(control_df_subsets[combo_type][brain_region])
            #print(common_control_sd1, common_control_sd, 'a')
            #std_lookup = get_std_per_observation(50, 100, observed_vals)
            #pred_vals = [np.mean(df_subsets[combo_type][brain_region]) for x in df_subsets[combo_type][brain_region]]

            z_vals = [(observed_val-pred_val)/common_control_sd for observed_val, pred_val in zip(observed_vals, pred_vals)]

            #print(combo_type, brain_region, np.mean(z_vals), np.std(z_vals), 'using common')

            z_scores_all_individuals[brain_region].update({k:v for k,v in zip(df_subsets[combo_type]['EID'], z_vals)})

    z_score_df = pd.DataFrame(z_scores_all_individuals)

    return z_score_df

def fit_linear_models_to_control_data(regression_model_type, control_df_subsets, controls_df, brain_regions_to_use, save_dir, categorical_explanatory_variables, variables_for_separate_models, continous_explanatory_variables, all_covariates):

    if not essential_plots_only:
        # Plot interactions of each covariate with relationship between each other covariate and Y:
        print('Plotting interactions between variables')
        plot_interactions_func(controls_df, brain_regions_to_use, save_dir, categorical_explanatory_variables+variables_for_separate_models, continous_explanatory_variables)

    aic_scores = {x: {} for x in control_df_subsets}
    fitted_models = {x: {} for x in control_df_subsets}
    vif_scores = {x: {} for x in control_df_subsets}
    hs_pvals = {x: {} for x in control_df_subsets}
    cvm_criterions = {x: {} for x in control_df_subsets}
    ks_res_pvals = {x: {} for x in control_df_subsets}
    ad_pvals = {x: {} for x in control_df_subsets}
    skew_vals = {x: {} for x in control_df_subsets}

    for combo_type in control_df_subsets:

        subset_df = control_df_subsets[combo_type]

        if combo_type == 'all_data':
            subset_save_dir = save_dir
        else:
            subset_save_dir = f'{save_dir}\\{combo_type}'

        if not os.path.exists(subset_save_dir):
            os.mkdir(subset_save_dir)

        ctype = combo_type.replace('__', ':')

        if not essential_plots_only:
            # Plot the mean-variance relationship for each dependent variable:
            print(f'Plotting the mean-variance relationship for each dependent variable for data subset {ctype}')
            plot_mvr_each_y(brain_regions_to_use, subset_save_dir, subset_df, num_chunks=200, var_smoothing_sigma=3)

            # Plot each independent variable against the dependent variable:
            print(f'Plotting each independent variable against the dependent variable for data subset {ctype}')
            plot_dep_vs_ind(subset_save_dir, brain_regions_to_use, all_covariates, subset_df)

            if len(all_covariates) > 1:
                print(f'Measuring multicolinearity for data subset {ctype}')

                # Measure multi-colinearity - although it does not matter for our purposes:
                plot_multicolinearity(subset_df, all_covariates, subset_save_dir)

        if len(all_covariates) > 1:

            # For each X, calculate VIF and save in dataframe: 
            vif_scores[combo_type] = measure_vif(subset_df, all_covariates)

            # Test for interaction statistically:
            test_interactions_statistically(brain_regions_to_use, subset_df, all_covariates, subset_save_dir)

        print(f'Fitting models for each brain region for data subset {ctype}')

        # Then, using the selected model create fitted models for each brain region, and save them:
        fitted_model_dict = fit_and_save_models_for_all_regions(brain_regions_to_use, subset_df, regression_model_type)
        fitted_models[combo_type] = fitted_model_dict

        # Save coeffients:
        coef_df = pd.DataFrame({br: dict(fitted_model_dict[br]['model'].params) for br in brain_regions_to_use})
        coef_df.to_csv(f'{subset_save_dir}\\all_coeff_df.csv')

        # Save coefficient p-values:
        pval_df = pd.DataFrame({br: dict(fitted_model_dict[br]['model'].pvalues) for br in brain_regions_to_use})
        pval_df.to_csv(f'{subset_save_dir}\\all_coeff_pval_df.csv')

        print(f'Plotting predicted vs actual values of dependent variable and residual plots for data subset {ctype}')

        # Plot predicted vs actual values of dependent variable
        plot_actual_vs_pred(subset_save_dir, brain_regions_to_use, subset_df, fitted_model_dict, all_covariates)

        # Plot different types of residuals vs the dependent variable:
        plot_residuals_vs_fitted_y_and_covariates(brain_regions_to_use, fitted_model_dict, subset_save_dir, subset_df, all_covariates, combo_type)

        # Test for heteroskedasticity:
        hs_pvals[combo_type] = hetsked_test_all_regions(brain_regions_to_use, fitted_model_dict)

        print(f'Testing for heteroskedasticity, and normality, for data subset {ctype}')

        # Assess normality of residuals:
        n_res = perform_normality_tests_and_plots(brain_regions_to_use, fitted_model_dict, subset_save_dir)
        cvm_criterions[combo_type], ks_res_pvals[combo_type], ad_pvals[combo_type], skew_vals[combo_type] = n_res

        # Record the model's AIC scores:
        aic_scores[combo_type] = {brain_region: fitted_model_dict[brain_region]['model'].aic for brain_region in brain_regions_to_use}

    # Save the AIC scores in a single table:
    pd.DataFrame(aic_scores).to_csv(f'{save_dir}\\aic_scores.csv')

    # Save the VIF scores in a single table:
    pd.DataFrame(vif_scores).to_csv(f'{save_dir}\\vif_scores.csv')

    # Save the HS test p values in a single table:
    pd.DataFrame(hs_pvals).to_csv(f'{save_dir}\\HS_test_p_values.csv')

    # Save the KS test p values in a single table:
    pd.DataFrame(ks_res_pvals).to_csv(f'{save_dir}\\KS_test_p_values.csv')

    # Save the CVM criterion results in a single table:
    pd.DataFrame(cvm_criterions).to_csv(f'{save_dir}\\CVM_criterion_results.csv')

    # Save the AD test p values in a single table:
    pd.DataFrame(ad_pvals).to_csv(f'{save_dir}\\AD_test_p_values.csv')

    # Save the skew  values in a single table:
    pd.DataFrame(skew_vals).to_csv(f'{save_dir}\\skew_values.csv')

    # Save all the models:
    with open(f'{save_dir}\\fitted_models.pkl', 'wb') as fp:
        pickle.dump(fitted_models, fp)

    return fitted_models



if __name__ == '__main__':

    pd.options.mode.chained_assignment = None
    plt.rcParams.update({'figure.max_open_warning': 0})

    save_dir = make_save_dir(working_dir, experiment_name, regression_model_type, polynomial_features_degree)

    # Load data and add additional fields:
    print('Reading input data')
    df = pd.read_csv(f'{working_dir}\\{input_data_file}', low_memory=False)

    # Exclude data from MRI centre 11028, due to very low numbers of scans from this scanner:
    df = df[df['R_Centre_f54_2'] != 11028] 

    if log_transform_covariates:
        for cov in continous_explanatory_variables:
            df = df[df[cov]>1] 
            df[cov] = np.log(df[cov])

    if grouped_brain_regions == True:
        brain_regions_to_use = [a for b in groupings.values() for a in b]


    # Get data where MRI volume is available for all regions
    for f in brain_regions_to_use:
        df = df[df[f].notnull()]
        
        if log_transform_dependent_variables:
            df = df[df[f]>1] 
            df[f] = np.log(df[f])


    if grouped_brain_regions == True:

        groupings = {k:groupings[k] for k in groupings if len(groupings[k])>0}
        
        midline_regions = [x for x in groupings if len([y for y in groupings[x] if 'left' in y or 'right' in y])==0]

        brain_regions_to_use = []

        for side in ('left', 'right'):

            for grouped_region in groupings:

                if grouped_region in midline_regions:
                    region_name = grouped_region
                    regions = groupings[grouped_region]

                else:
                    region_name = f'{grouped_region}_{side}'
                    regions = [x for x in groupings[grouped_region] if side in x]
                
                df[region_name] = df[regions].sum(axis=1)
            
                if region_name not in brain_regions_to_use:
                    brain_regions_to_use.append(region_name)


    # Replace categorical values with dummy values:
    print('Replacing categorical values with dummy values')
    dummy_lookup, dv_col_names = add_dummy_variables(df, categorical_explanatory_variables)

    all_covariates = continous_explanatory_variables+dv_col_names

    # Get dementia cases only:
    dementia_df = df[df[has_dementia]==1]

    if use_pre_diagnosis_scans_only == True:
        dementia_df = dementia_df[dementia_df[age_at_diagnosis] > dementia_df[age_at_scan]]
    
    # Get dist of difference between age at scan and age at dx
    dementia_df['age_diff'] = dementia_df[age_at_scan] - dementia_df[age_at_diagnosis]
    print(f'median age diff: {np.median(dementia_df['age_diff'])}')
    print(f'min age diff: {min(dementia_df['age_diff'])}')
    print(f'max age diff: {max(dementia_df['age_diff'])}')

    # Get controls only:
    controls_df = df[df['R_ML_NeuroExcludeSRFinal'] == 0]

    # Get seperate subsets of the data for separate combinations of specified variables:
    control_df_subsets = get_separate_dfs_for_each_combo(variables_for_separate_models, controls_df)

    fitted_models = fit_linear_models_to_control_data(
            regression_model_type, control_df_subsets, controls_df, 
            brain_regions_to_use, save_dir, categorical_explanatory_variables, 
            variables_for_separate_models, continous_explanatory_variables, 
            all_covariates
            )

    # Then calculate z values to feed into pysustain:
    cases_z_score_df = get_z_scores(dementia_df, brain_regions_to_use, fitted_models, all_covariates)

    cases_z_score_df.to_csv(f'{save_dir}\\preclindementia{len(cases_z_score_df)}_{len(brain_regions_to_use)}_regions_z_scores.csv')
    dementia_df.to_csv(f'{save_dir}\\preclindementia{len(dementia_df)}_raw_data.csv') 
