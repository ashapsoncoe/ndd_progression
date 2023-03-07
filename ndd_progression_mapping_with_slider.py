import pickle
from scipy.stats import mode
import plotly.graph_objects as go
import numpy as np
from plotly.subplots import make_subplots
from collections import Counter

sustain_output_dir = 'C:\\Users\\alexs\\Documents\\acf_work\\pc_dementia70_all_130_cortical_regions_3events_25startpoints_2maxclusters_100000iterations_130biomarkers\\pickle_files\\pc_dementia70_all_130_cortical_regions_subtype1.pickle'


brain_regions_xyz_lookup = {
    "Right Frontal Pole": (6.300000000000001, 65.10000000000001, 1.05),
    "Right Insular Cortex": (42.0, 14.7, -5.25),
    "Right Superior Frontal Gyrus": (25.200000000000003, 21.0, 59.85),
    "Right Middle Frontal Gyrus": (42.0, 18.9, 43.05),
    "Right Inferior Frontal Gyrus, pars triangularis": (52.5, 29.4, 11.55),
    "Right Inferior Frontal Gyrus, pars opercularis": (52.5, 16.8, 26.25),
    "Right Precentral Gyrus": (3.0, -25.200000000000003, 70.35000000000001),
    "Right Temporal Pole": (33.6, 12.6, -36.75),
    "Right Superior Temporal Gyrus, anterior division": (58.8, -8.4, -5.25),
    "Right Superior Temporal Gyrus, posterior division": (63.0, -23.1, 1.05),
    "Right Middle Temporal Gyrus, anterior division": (60.900000000000006, -8.4, -17.85),
    "Right Middle Temporal Gyrus, posterior division": (65.10000000000001, -25.200000000000003, -9.45),
    "Right Middle Temporal Gyrus, temporooccipital part": (56.7, -58.8, 7.35),
    "Right Inferior Temporal Gyrus, anterior division": (48.3, -4.2, -38.85),
    "Right Inferior Temporal Gyrus, posterior division": (52.5, -44.1, -17.85),
    "Right Inferior Temporal Gyrus, temporooccipital part": (52.5, -58.8, -11.55),
    "Right Postcentral Gyrus": (14.7, -44.1, 72.45),
    "Right Superior Parietal Lobule": (35.7, -50.400000000000006, 59.85),
    "Right Supramarginal Gyrus, anterior division": (58.8, -31.5, 38.85),
    "Right Supramarginal Gyrus, posterior division": (56.7, -50.400000000000006, 34.65),
    "Right Angular Gyrus": (50.400000000000006, -54.6, 43.05),
    "Right Lateral Occipital Cortex, superior division": (31.5, -73.5, 47.25),
    "Right Lateral Occipital Cortex, inferior division": (48.3, -73.5, 7.35),
    "Right Intracalcarine Cortex": (4.2, -84.0, 11.55),
    "Right Frontal Medial Cortex": (3.0, 44.1, -15.75),
    "Right Juxtapositional Lobule Cortex (formerly Supplementary Motor Cortex)": (3.0, 0.0, 59.85),
    "Right Subcallosal Cortex": (3.0, 23.1, -15.75),
    "Right Paracingulate Gyrus": (3.0, 35.7, 36.75),
    "Right Cingulate Gyrus, anterior division": (3.0, -4.2, 47.25),
    "Right Cingulate Gyrus, posterior division": (3.0, -44.1, 40.95),
    "Right Precuneous Cortex": (3.0, -67.2, 43.05),
    "Right Cuneal Cortex": (3.0, -84.0, 30.450000000000003),
    "Right Frontal Orbital Cortex": (42.0, 29.4, -11.55),
    "Right Parahippocampal Gyrus, anterior division": (25.200000000000003, -2.1, -34.65),
    "Right Parahippocampal Gyrus, posterior division": (27.3, -37.8, -13.65),
    "Right Lingual Gyrus": (8.4, -86.10000000000001, -3.1500000000000004),
    "Right Temporal Fusiform Cortex, anterior division": (35.7, -4.2, -40.95),
    "Right Temporal Fusiform Cortex, posterior division": (39.9, -33.6, -22.05),
    "Right Temporal Occipital Fusiform Cortex": (35.7, -54.6, -11.55),
    "Right Occipital Fusiform Gyrus": (25.200000000000003, -84.0, -9.45),
    "Right Frontal Operculum Cortex": (44.1, 23.1, 5.25),
    "Right Central Opercular Cortex": (54.6, -4.2, 11.55),
    "Right Parietal Operculum Cortex": (58.8, -27.3, 26.25),
    "Right Planum Polare": (52.5, -6.300000000000001, 3.1500000000000004),
    "Right Heschl's Gyrus (includes H1 and H2)": (46.2, -23.1, 9.45),
    "Right Planum Temporale": (60.900000000000006, -21.0, 11.55),
    "Right Supracalcarine Cortex": (3.0, -88.2, 15.75),
    "Right Occipital Pole": (16.8, -100.8, -1.05),
    "Left Frontal Pole": (-6.300000000000001, 65.10000000000001, 1.05),
    "Left Insular Cortex": (-42.0, 14.7, -5.25),
    "Left Superior Frontal Gyrus": (-25.200000000000003, 21.0, 59.85),
    "Left Middle Frontal Gyrus": (-42.0, 18.9, 43.05),
    "Left Inferior Frontal Gyrus, pars triangularis": (-52.5, 29.4, 11.55),
    "Left Inferior Frontal Gyrus, pars opercularis": (-52.5, 16.8, 26.25),
    "Left Precentral Gyrus": (-3.0, -25.200000000000003, 70.35000000000001),
    "Left Temporal Pole": (-33.6, 12.6, -36.75),
    "Left Superior Temporal Gyrus, anterior division": (-58.8, -8.4, -5.25),
    "Left Superior Temporal Gyrus, posterior division": (-63.0, -23.1, 1.05),
    "Left Middle Temporal Gyrus, anterior division": (-60.900000000000006, -8.4, -17.85),
    "Left Middle Temporal Gyrus, posterior division": (-65.10000000000001, -25.200000000000003, -9.45),
    "Left Middle Temporal Gyrus, temporooccipital part": (-56.7, -58.8, 7.35),
    "Left Inferior Temporal Gyrus, anterior division": (-48.3, -4.2, -38.85),
    "Left Inferior Temporal Gyrus, posterior division": (-52.5, -44.1, -17.85),
    "Left Inferior Temporal Gyrus, temporooccipital part": (-52.5, -58.8, -11.55),
    "Left Postcentral Gyrus": (-14.7, -44.1, 72.45),
    "Left Superior Parietal Lobule": (-35.7, -50.400000000000006, 59.85),
    "Left Supramarginal Gyrus, anterior division": (-58.8, -31.5, 38.85),
    "Left Supramarginal Gyrus, posterior division": (-56.7, -50.400000000000006, 34.65),
    "Left Angular Gyrus": (-50.400000000000006, -54.6, 43.05),
    "Left Lateral Occipital Cortex, superior division": (-31.5, -73.5, 47.25),
    "Left Lateral Occipital Cortex, inferior division": (-48.3, -73.5, 7.35),
    "Left Intracalcarine Cortex": (-4.2, -84.0, 11.55),
    "Left Frontal Medial Cortex": (-3.0, 44.1, -15.75),
    "Left Juxtapositional Lobule Cortex (formerly Supplementary Motor Cortex)": (-3.0, 0.0, 59.85),
    "Left Subcallosal Cortex": (-3.0, 23.1, -15.75),
    "Left Paracingulate Gyrus": (-3.0, 35.7, 36.75),
    "Left Cingulate Gyrus, anterior division": (-3.0, -4.2, 47.25),
    "Left Cingulate Gyrus, posterior division": (-3.0, -44.1, 40.95),
    "Left Precuneous Cortex": (-3.0, -67.2, 43.05),
    "Left Cuneal Cortex": (-3.0, -84.0, 30.450000000000003),
    "Left Frontal Orbital Cortex": (-42.0, 29.4, -11.55),
    "Left Parahippocampal Gyrus, anterior division": (-25.200000000000003, -2.1, -34.65),
    "Left Parahippocampal Gyrus, posterior division": (-27.3, -37.8, -13.65),
    "Left Lingual Gyrus": (-8.4, -86.10000000000001, -3.1500000000000004),
    "Left Temporal Fusiform Cortex, anterior division": (-35.7, -4.2, -40.95),
    "Left Temporal Fusiform Cortex, posterior division": (-39.9, -33.6, -22.05),
    "Left Temporal Occipital Fusiform Cortex": (-35.7, -54.6, -11.55),
    "Left Occipital Fusiform Gyrus": (-25.200000000000003, -84.0, -9.45),
    "Left Frontal Operculum Cortex": (-44.1, 23.1, 5.25),
    "Left Central Opercular Cortex": (-54.6, -4.2, 11.55),
    "Left Parietal Operculum Cortex": (-58.8, -27.3, 26.25),
    "Left Planum Polare": (-52.5, -6.300000000000001, 3.1500000000000004),
    "Left Heschl's Gyrus (includes H1 and H2)": (-46.2, -23.1, 9.45),
    "Left Planum Temporale": (-60.900000000000006, -21.0, 11.55),
    "Left Supracalcarine Cortex": (-3.0, -88.2, 15.75),
    "Left Occipital Pole": (-16.8, -100.8, -1.05),
    "Left Lateral Ventrical": (-23.1, -44.1, 13.65),
    "Left Thalamus": (-10.5, -23.1, 9.45),
    "Left Caudate": (-12.6, 14.7, 11.55),
    "Left Putamen": (-23.1, 8.4, -1.05),
    "Left Pallidum": (-18.9, -2.1, 1.05),
    "Left Hippocampus": (-29.4, -16.8, -17.85),
    "Left Amygdala": (-25.200000000000003, -2.1, -19.95),
    "Left Accumbens": (-10.5, 12.6, -5.25),
    "Right Thalamus": (14.7, -25.200000000000003, 9.45),
    "Right Caudate": (14.7, 12.6, 15.75),
    "Right Putamen": (23.1, 10.5, -1.05),
    "Right Pallidum": (18.9, -2.1, -1.05),
    "Right Hippocampus": (29.4, -14.7, -17.85),
    "Right Amygdala": (27.3, 0.0, -19.95),
    "Right Accumbens": (10.5, 10.5, -5.25),
}

oxford_to_biobank_brain_region_lookup = {
    "Right Frontal Pole": 'volume_of_grey_matter_in_frontal_pole_right_f25783_2_0',
    "Right Insular Cortex": 'volume_of_grey_matter_in_insular_cortex_right_f25785_2_0',
    "Right Superior Frontal Gyrus": 'volume_of_grey_matter_in_superior_frontal_gyrus_right_f25787_2_0',
    "Right Middle Frontal Gyrus": 'volume_of_grey_matter_in_middle_frontal_gyrus_right_f25789_2_0',
    "Right Inferior Frontal Gyrus, pars triangularis": 'volume_of_grey_matter_in_inferior_frontal_gyrus_pars_triangularis_right_f25791_2_0',
    "Right Inferior Frontal Gyrus, pars opercularis": 'volume_of_grey_matter_in_inferior_frontal_gyrus_pars_opercularis_right_f25793_2_0',
    "Right Precentral Gyrus": 'volume_of_grey_matter_in_precentral_gyrus_right_f25795_2_0',
    "Right Temporal Pole": 'volume_of_grey_matter_in_temporal_pole_right_f25797_2_0',
    "Right Superior Temporal Gyrus, anterior division": 'volume_of_grey_matter_in_superior_temporal_gyrus_anterior_division_right_f25799_2_0',
    "Right Superior Temporal Gyrus, posterior division": 'volume_of_grey_matter_in_superior_temporal_gyrus_posterior_division_right_f25801_2_0',
    "Right Middle Temporal Gyrus, anterior division": 'volume_of_grey_matter_in_middle_temporal_gyrus_anterior_division_right_f25803_2_0',
    "Right Middle Temporal Gyrus, posterior division": 'volume_of_grey_matter_in_middle_temporal_gyrus_posterior_division_right_f25805_2_0',
    "Right Middle Temporal Gyrus, temporooccipital part": 'volume_of_grey_matter_in_middle_temporal_gyrus_temporooccipital_part_right_f25807_2_0',
    "Right Inferior Temporal Gyrus, anterior division": 'volume_of_grey_matter_in_inferior_temporal_gyrus_anterior_division_right_f25809_2_0',
    "Right Inferior Temporal Gyrus, posterior division": 'volume_of_grey_matter_in_inferior_temporal_gyrus_posterior_division_right_f25811_2_0',
    "Right Inferior Temporal Gyrus, temporooccipital part": 'volume_of_grey_matter_in_inferior_temporal_gyrus_temporooccipital_part_right_f25813_2_0',
    "Right Postcentral Gyrus": 'volume_of_grey_matter_in_postcentral_gyrus_right_f25815_2_0',
    "Right Superior Parietal Lobule": 'volume_of_grey_matter_in_superior_parietal_lobule_right_f25817_2_0',
    "Right Supramarginal Gyrus, anterior division": 'volume_of_grey_matter_in_supramarginal_gyrus_anterior_division_right_f25819_2_0',
    "Right Supramarginal Gyrus, posterior division": 'volume_of_grey_matter_in_supramarginal_gyrus_posterior_division_right_f25821_2_0',
    "Right Angular Gyrus": 'volume_of_grey_matter_in_angular_gyrus_right_f25823_2_0',
    "Right Lateral Occipital Cortex, superior division": 'volume_of_grey_matter_in_lateral_occipital_cortex_superior_division_right_f25825_2_0',
    "Right Lateral Occipital Cortex, inferior division": 'volume_of_grey_matter_in_lateral_occipital_cortex_inferior_division_right_f25827_2_0',
    "Right Intracalcarine Cortex": 'volume_of_grey_matter_in_intracalcarine_cortex_right_f25829_2_0',
    "Right Frontal Medial Cortex": 'volume_of_grey_matter_in_frontal_medial_cortex_right_f25831_2_0',
    "Right Juxtapositional Lobule Cortex (formerly Supplementary Motor Cortex)": 'volume_of_grey_matter_in_juxtapositional_lobule_cortex_formerly_supplementary_motor_cortex_right_f25833_2_0',
    "Right Subcallosal Cortex": 'volume_of_grey_matter_in_subcallosal_cortex_right_f25835_2_0',
    "Right Paracingulate Gyrus": 'volume_of_grey_matter_in_paracingulate_gyrus_right_f25837_2_0',
    "Right Cingulate Gyrus, anterior division": 'volume_of_grey_matter_in_cingulate_gyrus_anterior_division_right_f25839_2_0',
    "Right Cingulate Gyrus, posterior division": 'volume_of_grey_matter_in_cingulate_gyrus_posterior_division_right_f25841_2_0',
    "Right Precuneous Cortex": 'volume_of_grey_matter_in_precuneous_cortex_right_f25843_2_0',
    "Right Cuneal Cortex": 'volume_of_grey_matter_in_cuneal_cortex_right_f25845_2_0',
    "Right Frontal Orbital Cortex": 'volume_of_grey_matter_in_frontal_orbital_cortex_right_f25847_2_0',
    "Right Parahippocampal Gyrus, anterior division": 'volume_of_grey_matter_in_parahippocampal_gyrus_anterior_division_right_f25849_2_0',
    "Right Parahippocampal Gyrus, posterior division": 'volume_of_grey_matter_in_parahippocampal_gyrus_posterior_division_right_f25851_2_0',
    "Right Lingual Gyrus": 'volume_of_grey_matter_in_lingual_gyrus_right_f25853_2_0',
    "Right Temporal Fusiform Cortex, anterior division": 'volume_of_grey_matter_in_temporal_fusiform_cortex_anterior_division_right_f25855_2_0',
    "Right Temporal Fusiform Cortex, posterior division": 'volume_of_grey_matter_in_temporal_fusiform_cortex_posterior_division_right_f25857_2_0',
    "Right Temporal Occipital Fusiform Cortex": 'volume_of_grey_matter_in_temporal_occipital_fusiform_cortex_right_f25859_2_0',
    "Right Occipital Fusiform Gyrus": 'volume_of_grey_matter_in_occipital_fusiform_gyrus_right_f25861_2_0',
    "Right Frontal Operculum Cortex": 'volume_of_grey_matter_in_frontal_operculum_cortex_right_f25863_2_0',
    "Right Central Opercular Cortex": 'volume_of_grey_matter_in_central_opercular_cortex_right_f25865_2_0',
    "Right Parietal Operculum Cortex": 'volume_of_grey_matter_in_parietal_operculum_cortex_right_f25867_2_0',
    "Right Planum Polare": 'volume_of_grey_matter_in_planum_polare_right_f25869_2_0',
    "Right Heschl's Gyrus (includes H1 and H2)": 'volume_of_grey_matter_in_heschls_gyrus_includes_h1_and_h2_right_f25871_2_0',
    "Right Planum Temporale": 'volume_of_grey_matter_in_planum_temporale_right_f25873_2_0',
    "Right Supracalcarine Cortex": 'volume_of_grey_matter_in_supracalcarine_cortex_right_f25875_2_0',
    "Right Occipital Pole": 'volume_of_grey_matter_in_occipital_pole_right_f25877_2_0',
    "Left Frontal Pole": 'volume_of_grey_matter_in_frontal_pole_left_f25782_2_0',
    "Left Insular Cortex": 'volume_of_grey_matter_in_insular_cortex_left_f25784_2_0',
    "Left Superior Frontal Gyrus": 'volume_of_grey_matter_in_superior_frontal_gyrus_left_f25786_2_0',
    "Left Middle Frontal Gyrus": 'volume_of_grey_matter_in_middle_frontal_gyrus_left_f25788_2_0',
    "Left Inferior Frontal Gyrus, pars triangularis": 'volume_of_grey_matter_in_inferior_frontal_gyrus_pars_triangularis_left_f25790_2_0',
    "Left Inferior Frontal Gyrus, pars opercularis": 'volume_of_grey_matter_in_inferior_frontal_gyrus_pars_opercularis_left_f25792_2_0',
    "Left Precentral Gyrus": 'volume_of_grey_matter_in_precentral_gyrus_left_f25794_2_0',
    "Left Temporal Pole": 'volume_of_grey_matter_in_temporal_pole_left_f25796_2_0',
    "Left Superior Temporal Gyrus, anterior division": 'volume_of_grey_matter_in_superior_temporal_gyrus_anterior_division_left_f25798_2_0',
    "Left Superior Temporal Gyrus, posterior division": 'volume_of_grey_matter_in_superior_temporal_gyrus_posterior_division_left_f25800_2_0',
    "Left Middle Temporal Gyrus, anterior division": 'volume_of_grey_matter_in_middle_temporal_gyrus_anterior_division_left_f25802_2_0',
    "Left Middle Temporal Gyrus, posterior division": 'volume_of_grey_matter_in_middle_temporal_gyrus_posterior_division_left_f25804_2_0',
    "Left Middle Temporal Gyrus, temporooccipital part": 'volume_of_grey_matter_in_middle_temporal_gyrus_temporooccipital_part_left_f25806_2_0',
    "Left Inferior Temporal Gyrus, anterior division": 'volume_of_grey_matter_in_inferior_temporal_gyrus_anterior_division_left_f25808_2_0',
    "Left Inferior Temporal Gyrus, posterior division": 'volume_of_grey_matter_in_inferior_temporal_gyrus_posterior_division_left_f25810_2_0',
    "Left Inferior Temporal Gyrus, temporooccipital part": 'volume_of_grey_matter_in_inferior_temporal_gyrus_temporooccipital_part_left_f25812_2_0',
    "Left Postcentral Gyrus": 'volume_of_grey_matter_in_postcentral_gyrus_left_f25814_2_0',
    "Left Superior Parietal Lobule": 'volume_of_grey_matter_in_superior_parietal_lobule_left_f25816_2_0',
    "Left Supramarginal Gyrus, anterior division": 'volume_of_grey_matter_in_supramarginal_gyrus_anterior_division_left_f25818_2_0',
    "Left Supramarginal Gyrus, posterior division": 'volume_of_grey_matter_in_supramarginal_gyrus_posterior_division_left_f25820_2_0',
    "Left Angular Gyrus": 'volume_of_grey_matter_in_angular_gyrus_left_f25822_2_0',
    "Left Lateral Occipital Cortex, superior division": 'volume_of_grey_matter_in_lateral_occipital_cortex_superior_division_left_f25824_2_0',
    "Left Lateral Occipital Cortex, inferior division": 'volume_of_grey_matter_in_lateral_occipital_cortex_inferior_division_left_f25826_2_0',
    "Left Intracalcarine Cortex": 'volume_of_grey_matter_in_intracalcarine_cortex_left_f25828_2_0',
    "Left Frontal Medial Cortex": 'volume_of_grey_matter_in_frontal_medial_cortex_left_f25830_2_0',
    "Left Juxtapositional Lobule Cortex (formerly Supplementary Motor Cortex)": 'volume_of_grey_matter_in_juxtapositional_lobule_cortex_formerly_supplementary_motor_cortex_left_f25832_2_0',
    "Left Subcallosal Cortex": 'volume_of_grey_matter_in_subcallosal_cortex_left_f25834_2_0',
    "Left Paracingulate Gyrus": 'volume_of_grey_matter_in_paracingulate_gyrus_left_f25836_2_0',
    "Left Cingulate Gyrus, anterior division": 'volume_of_grey_matter_in_cingulate_gyrus_anterior_division_left_f25838_2_0',
    "Left Cingulate Gyrus, posterior division": 'volume_of_grey_matter_in_cingulate_gyrus_posterior_division_left_f25840_2_0',
    "Left Precuneous Cortex": 'volume_of_grey_matter_in_precuneous_cortex_left_f25842_2_0',
    "Left Cuneal Cortex": 'volume_of_grey_matter_in_cuneal_cortex_left_f25844_2_0',
    "Left Frontal Orbital Cortex": 'volume_of_grey_matter_in_frontal_orbital_cortex_left_f25846_2_0',
    "Left Parahippocampal Gyrus, anterior division": 'volume_of_grey_matter_in_parahippocampal_gyrus_anterior_division_left_f25848_2_0',
    "Left Parahippocampal Gyrus, posterior division": 'volume_of_grey_matter_in_parahippocampal_gyrus_posterior_division_left_f25850_2_0',
    "Left Lingual Gyrus": 'volume_of_grey_matter_in_lingual_gyrus_left_f25852_2_0',
    "Left Temporal Fusiform Cortex, anterior division": 'volume_of_grey_matter_in_temporal_fusiform_cortex_anterior_division_left_f25854_2_0',
    "Left Temporal Fusiform Cortex, posterior division": 'volume_of_grey_matter_in_temporal_fusiform_cortex_posterior_division_left_f25856_2_0',
    "Left Temporal Occipital Fusiform Cortex": 'volume_of_grey_matter_in_temporal_occipital_fusiform_cortex_left_f25858_2_0',
    "Left Occipital Fusiform Gyrus": 'volume_of_grey_matter_in_occipital_fusiform_gyrus_left_f25860_2_0',
    "Left Frontal Operculum Cortex": 'volume_of_grey_matter_in_frontal_operculum_cortex_left_f25862_2_0',
    "Left Central Opercular Cortex": 'volume_of_grey_matter_in_central_opercular_cortex_left_f25864_2_0',
    "Left Parietal Operculum Cortex": 'volume_of_grey_matter_in_parietal_operculum_cortex_left_f25866_2_0',
    "Left Planum Polare": 'volume_of_grey_matter_in_planum_polare_left_f25868_2_0',
    "Left Heschl's Gyrus (includes H1 and H2)": 'volume_of_grey_matter_in_heschls_gyrus_includes_h1_and_h2_left_f25870_2_0',
    "Left Planum Temporale": 'volume_of_grey_matter_in_planum_temporale_left_f25872_2_0',
    "Left Supracalcarine Cortex": 'volume_of_grey_matter_in_supracalcarine_cortex_left_f25874_2_0',
    "Left Occipital Pole": 'volume_of_grey_matter_in_occipital_pole_left_f25876_2_0',
    "Left Lateral Ventrical": None,
    "Left Thalamus": 'volume_of_thalamus_left_f25011_2_0',
    "Left Caudate": 'volume_of_caudate_left_f25013_2_0',
    "Left Putamen": 'volume_of_putamen_left_f25015_2_0',
    "Left Pallidum": 'volume_of_pallidum_left_f25017_2_0',
    "Left Hippocampus": 'volume_of_hippocampus_left_f25019_2_0',
    "Left Amygdala": 'volume_of_amygdala_left_f25021_2_0',
    "Left Accumbens": 'volume_of_accumbens_left_f25023_2_0',
    "Right Thalamus": 'volume_of_thalamus_right_f25012_2_0',
    "Right Caudate": 'volume_of_caudate_right_f25014_2_0',
    "Right Putamen": 'volume_of_putamen_right_f25016_2_0',
    "Right Pallidum": 'volume_of_pallidum_right_f25018_2_0',
    "Right Hippocampus": 'volume_of_hippocampus_right_f25020_2_0',
    "Right Amygdala": 'volume_of_amygdala_right_f25022_2_0',
    "Right Accumbens": 'volume_of_accumbens_right_f25024_2_0',
}

mean_volumes_from_all_ukbiobank_data = {
    'volume_of_brain_greywhite_matter_normalised_for_head_size_f25009_2_0': 1492871.3999042145,
    'volume_of_accumbens_left_f25023_2_0': 491.9314894636015,
    'volume_of_accumbens_right_f25024_2_0': 385.4747126436782,
    'volume_of_amygdala_left_f25021_2_0': 1262.9178400383141,
    'volume_of_amygdala_right_f25022_2_0': 1227.4833333333333,
    'volume_of_caudate_left_f25013_2_0': 3375.8710009578544,
    'volume_of_caudate_right_f25014_2_0': 3559.135009578544,
    'volume_of_hippocampus_left_f25019_2_0': 3772.4200909961687,
    'volume_of_hippocampus_right_f25020_2_0': 3886.9115660919542,
    'volume_of_pallidum_left_f25017_2_0': 1754.2928639846743,
    'volume_of_pallidum_right_f25018_2_0': 1798.1109674329502,
    'volume_of_putamen_left_f25015_2_0': 4758.1919300766285,
    'volume_of_putamen_right_f25016_2_0': 4815.195785440613,
    'volume_of_thalamus_left_f25011_2_0': 7742.561613984674,
    'volume_of_thalamus_right_f25012_2_0': 7550.337859195402,
    'volume_of_grey_matter_in_amygdala_left_f25888_2_0': 1823.5626583572798,
    'volume_of_grey_matter_in_amygdala_right_f25889_2_0': 2091.410982926245,
    'volume_of_grey_matter_in_angular_gyrus_left_f25822_2_0': 4066.052129549809,
    'volume_of_grey_matter_in_angular_gyrus_right_f25823_2_0': 5497.562513888888,
    'volume_of_grey_matter_in_brainstem_f25892_2_0': 4871.6031714559385,
    'volume_of_grey_matter_in_caudate_left_f25880_2_0': 3049.5875771072792,
    'volume_of_grey_matter_in_caudate_right_f25881_2_0': 3267.376090517241,
    'volume_of_grey_matter_in_central_opercular_cortex_left_f25864_2_0': 3756.1118026819922,
    'volume_of_grey_matter_in_central_opercular_cortex_right_f25865_2_0': 3721.25317289272,
    'volume_of_grey_matter_in_cingulate_gyrus_anterior_division_left_f25838_2_0': 5120.492307950191,
    'volume_of_grey_matter_in_cingulate_gyrus_anterior_division_right_f25839_2_0': 5654.862340756705,
    'volume_of_grey_matter_in_cingulate_gyrus_posterior_division_left_f25840_2_0': 5187.6138144157085,
    'volume_of_grey_matter_in_cingulate_gyrus_posterior_division_right_f25841_2_0': 5425.868168582376,
    'volume_of_grey_matter_in_crus_i_cerebellum_left_f25900_2_0': 10588.674779214558,
    'volume_of_grey_matter_in_crus_i_cerebellum_right_f25902_2_0': 11443.20489750958,
    'volume_of_grey_matter_in_crus_i_cerebellum_vermis_f25901_2_0': 1.995365453735632,
    'volume_of_grey_matter_in_crus_ii_cerebellum_left_f25903_2_0': 8016.2734906609185,
    'volume_of_grey_matter_in_crus_ii_cerebellum_right_f25905_2_0': 7813.044504070883,
    'volume_of_grey_matter_in_crus_ii_cerebellum_vermis_f25904_2_0': 402.3834063936781,
    'volume_of_grey_matter_in_cuneal_cortex_left_f25844_2_0': 1936.3126017241377,
    'volume_of_grey_matter_in_cuneal_cortex_right_f25845_2_0': 2306.48200308908,
    'volume_of_grey_matter_in_frontal_medial_cortex_left_f25830_2_0': 1872.9129475095783,
    'volume_of_grey_matter_in_frontal_medial_cortex_right_f25831_2_0': 1877.8278523946362,
    'volume_of_grey_matter_in_frontal_operculum_cortex_left_f25862_2_0': 1501.4164695641762,
    'volume_of_grey_matter_in_frontal_operculum_cortex_right_f25863_2_0': 1332.5900014128351,
    'volume_of_grey_matter_in_frontal_orbital_cortex_left_f25846_2_0': 6591.903667624521,
    'volume_of_grey_matter_in_frontal_orbital_cortex_right_f25847_2_0': 5999.679586446361,
    'volume_of_grey_matter_in_frontal_pole_left_f25782_2_0': 23120.77861350575,
    'volume_of_grey_matter_in_frontal_pole_right_f25783_2_0': 26191.424750957853,
    'volume_of_grey_matter_in_heschls_gyrus_includes_h1_and_h2_left_f25870_2_0': 1181.755563888889,
    'volume_of_grey_matter_in_heschls_gyrus_includes_h1_and_h2_right_f25871_2_0': 1023.228639295977,
    'volume_of_grey_matter_in_hippocampus_left_f25886_2_0': 4192.833788314176,
    'volume_of_grey_matter_in_hippocampus_right_f25887_2_0': 4356.1453527298845,
    'volume_of_grey_matter_in_iiv_cerebellum_left_f25893_2_0': 1859.1471291427201,
    'volume_of_grey_matter_in_iiv_cerebellum_right_f25894_2_0': 2113.7296772270115,
    'volume_of_grey_matter_in_ix_cerebellum_left_f25915_2_0': 1900.2684668103448,
    'volume_of_grey_matter_in_ix_cerebellum_right_f25917_2_0': 2138.5812608237547,
    'volume_of_grey_matter_in_ix_cerebellum_vermis_f25916_2_0': 434.5490069444445,
    'volume_of_grey_matter_in_inferior_frontal_gyrus_pars_opercularis_left_f25792_2_0': 2636.328695162835,
    'volume_of_grey_matter_in_inferior_frontal_gyrus_pars_opercularis_right_f25793_2_0': 2460.567199928161,
    'volume_of_grey_matter_in_inferior_frontal_gyrus_pars_triangularis_left_f25790_2_0': 2433.254338362069,
    'volume_of_grey_matter_in_inferior_frontal_gyrus_pars_triangularis_right_f25791_2_0': 2253.230289152299,
    'volume_of_grey_matter_in_inferior_temporal_gyrus_anterior_division_left_f25808_2_0': 1430.3411371958816,
    'volume_of_grey_matter_in_inferior_temporal_gyrus_anterior_division_right_f25809_2_0': 1494.3015217193486,
    'volume_of_grey_matter_in_inferior_temporal_gyrus_posterior_division_left_f25810_2_0': 4122.68875881226,
    'volume_of_grey_matter_in_inferior_temporal_gyrus_posterior_division_right_f25811_2_0': 4168.8910052681995,
    'volume_of_grey_matter_in_inferior_temporal_gyrus_temporooccipital_part_left_f25812_2_0': 3007.412608477011,
    'volume_of_grey_matter_in_inferior_temporal_gyrus_temporooccipital_part_right_f25813_2_0': 3824.5731848659,
    'volume_of_grey_matter_in_insular_cortex_left_f25784_2_0': 6297.955900383141,
    'volume_of_grey_matter_in_insular_cortex_right_f25785_2_0': 6281.489784003831,
    'volume_of_grey_matter_in_intracalcarine_cortex_left_f25828_2_0': 2591.8643419540226,
    'volume_of_grey_matter_in_intracalcarine_cortex_right_f25829_2_0': 2704.8055531609193,
    'volume_of_grey_matter_in_juxtapositional_lobule_cortex_formerly_supplementary_motor_cortex_left_f25832_2_0': 2854.67335704023,
    'volume_of_grey_matter_in_juxtapositional_lobule_cortex_formerly_supplementary_motor_cortex_right_f25833_2_0': 2760.29794973659,
    'volume_of_grey_matter_in_lateral_occipital_cortex_inferior_division_left_f25826_2_0': 7186.773551484675,
    'volume_of_grey_matter_in_lateral_occipital_cortex_inferior_division_right_f25827_2_0': 7710.068363266284,
    'volume_of_grey_matter_in_lateral_occipital_cortex_superior_division_left_f25824_2_0': 16314.085372605363,
    'volume_of_grey_matter_in_lateral_occipital_cortex_superior_division_right_f25825_2_0': 15627.432477490422,
    'volume_of_grey_matter_in_lingual_gyrus_left_f25852_2_0': 6517.590555076628,
    'volume_of_grey_matter_in_lingual_gyrus_right_f25853_2_0': 6978.891064655173,
    'volume_of_grey_matter_in_middle_frontal_gyrus_left_f25788_2_0': 10038.98818295019,
    'volume_of_grey_matter_in_middle_frontal_gyrus_right_f25789_2_0': 9533.25558572797,
    'volume_of_grey_matter_in_middle_temporal_gyrus_anterior_division_left_f25802_2_0': 1822.7613205818966,
    'volume_of_grey_matter_in_middle_temporal_gyrus_anterior_division_right_f25803_2_0': 1607.4108485632185,
    'volume_of_grey_matter_in_middle_temporal_gyrus_posterior_division_left_f25804_2_0': 5305.740872126436,
    'volume_of_grey_matter_in_middle_temporal_gyrus_posterior_division_right_f25805_2_0': 5474.197325431034,
    'volume_of_grey_matter_in_middle_temporal_gyrus_temporooccipital_part_left_f25806_2_0': 3478.8636226293106,
    'volume_of_grey_matter_in_middle_temporal_gyrus_temporooccipital_part_right_f25807_2_0': 4705.820189415708,
    'volume_of_grey_matter_in_occipital_fusiform_gyrus_left_f25860_2_0': 3898.4716645114936,
    'volume_of_grey_matter_in_occipital_fusiform_gyrus_right_f25861_2_0': 3706.269475574713,
    'volume_of_grey_matter_in_occipital_pole_left_f25876_2_0': 8490.04285177203,
    'volume_of_grey_matter_in_occipital_pole_right_f25877_2_0': 8155.542771791187,
    'volume_of_grey_matter_in_pallidum_left_f25884_2_0': 41.37469550402298,
    'volume_of_grey_matter_in_pallidum_right_f25885_2_0': 60.32417843685345,
    'volume_of_grey_matter_in_paracingulate_gyrus_left_f25836_2_0': 5709.250458333334,
    'volume_of_grey_matter_in_paracingulate_gyrus_right_f25837_2_0': 5655.135579501916,
    'volume_of_grey_matter_in_parahippocampal_gyrus_anterior_division_left_f25848_2_0': 2932.5161264367816,
    'volume_of_grey_matter_in_parahippocampal_gyrus_anterior_division_right_f25849_2_0': 2999.063654693487,
    'volume_of_grey_matter_in_parahippocampal_gyrus_posterior_division_left_f25850_2_0': 1693.3593374042148,
    'volume_of_grey_matter_in_parahippocampal_gyrus_posterior_division_right_f25851_2_0': 1374.1023039511497,
    'volume_of_grey_matter_in_parietal_operculum_cortex_left_f25866_2_0': 2050.2156706417627,
    'volume_of_grey_matter_in_parietal_operculum_cortex_right_f25867_2_0': 2034.736277394636,
    'volume_of_grey_matter_in_planum_polare_left_f25868_2_0': 1383.323078184866,
    'volume_of_grey_matter_in_planum_polare_right_f25869_2_0': 1459.699612691571,
    'volume_of_grey_matter_in_planum_temporale_left_f25872_2_0': 2038.0356292863987,
    'volume_of_grey_matter_in_planum_temporale_right_f25873_2_0': 1609.8279427681991,
    'volume_of_grey_matter_in_postcentral_gyrus_left_f25814_2_0': 11139.299945402297,
    'volume_of_grey_matter_in_postcentral_gyrus_right_f25815_2_0': 10396.118636015324,
    'volume_of_grey_matter_in_precentral_gyrus_left_f25794_2_0': 13813.295377155173,
    'volume_of_grey_matter_in_precentral_gyrus_right_f25795_2_0': 13533.91541690613,
    'volume_of_grey_matter_in_precuneous_cortex_left_f25842_2_0': 10051.614373323755,
    'volume_of_grey_matter_in_precuneous_cortex_right_f25843_2_0': 10488.23519157088,
    'volume_of_grey_matter_in_putamen_left_f25882_2_0': 1758.7816908548855,
    'volume_of_grey_matter_in_putamen_right_f25883_2_0': 2172.2886767480845,
    'volume_of_grey_matter_in_subcallosal_cortex_left_f25834_2_0': 2934.7828026819925,
    'volume_of_grey_matter_in_subcallosal_cortex_right_f25835_2_0': 2722.634839798851,
    'volume_of_grey_matter_in_superior_frontal_gyrus_left_f25786_2_0': 11029.73433500958,
    'volume_of_grey_matter_in_superior_frontal_gyrus_right_f25787_2_0': 9601.393433189654,
    'volume_of_grey_matter_in_superior_parietal_lobule_left_f25816_2_0': 5106.775981082375,
    'volume_of_grey_matter_in_superior_parietal_lobule_right_f25817_2_0': 4625.4533445881225,
    'volume_of_grey_matter_in_superior_temporal_gyrus_anterior_division_left_f25798_2_0': 1462.5773278975096,
    'volume_of_grey_matter_in_superior_temporal_gyrus_anterior_division_right_f25799_2_0': 1466.6233962164752,
    'volume_of_grey_matter_in_superior_temporal_gyrus_posterior_division_left_f25800_2_0': 2575.698804861111,
    'volume_of_grey_matter_in_superior_temporal_gyrus_posterior_division_right_f25801_2_0': 3114.7377598180074,
    'volume_of_grey_matter_in_supracalcarine_cortex_left_f25874_2_0': 502.9838701628352,
    'volume_of_grey_matter_in_supracalcarine_cortex_right_f25875_2_0': 715.4900074233717,
    'volume_of_grey_matter_in_supramarginal_gyrus_anterior_division_left_f25818_2_0': 3187.2966041666664,
    'volume_of_grey_matter_in_supramarginal_gyrus_anterior_division_right_f25819_2_0': 3126.6137823275863,
    'volume_of_grey_matter_in_supramarginal_gyrus_posterior_division_left_f25820_2_0': 4632.058149664751,
    'volume_of_grey_matter_in_supramarginal_gyrus_posterior_division_right_f25821_2_0': 5439.643244971265,
    'volume_of_grey_matter_in_temporal_fusiform_cortex_anterior_division_left_f25854_2_0': 1506.1701021072797,
    'volume_of_grey_matter_in_temporal_fusiform_cortex_anterior_division_right_f25855_2_0': 1371.666467552682,
    'volume_of_grey_matter_in_temporal_fusiform_cortex_posterior_division_left_f25856_2_0': 3946.2718951149427,
    'volume_of_grey_matter_in_temporal_fusiform_cortex_posterior_division_right_f25857_2_0': 3235.8127945402302,
    'volume_of_grey_matter_in_temporal_occipital_fusiform_cortex_left_f25858_2_0': 2621.944698754789,
    'volume_of_grey_matter_in_temporal_occipital_fusiform_cortex_right_f25859_2_0': 3335.9393096264357,
    'volume_of_grey_matter_in_temporal_pole_left_f25796_2_0': 9426.347454022989,
    'volume_of_grey_matter_in_temporal_pole_right_f25797_2_0': 9369.772328304598,
    'volume_of_grey_matter_in_thalamus_left_f25878_2_0': 2696.2326063218393,
    'volume_of_grey_matter_in_thalamus_right_f25879_2_0': 2829.750096264368,
    'volume_of_grey_matter_in_v_cerebellum_left_f25895_2_0': 2623.839216954023,
    'volume_of_grey_matter_in_v_cerebellum_right_f25896_2_0': 2620.711854166667,
    'volume_of_grey_matter_in_vi_cerebellum_left_f25897_2_0': 6765.1035143678155,
    'volume_of_grey_matter_in_vi_cerebellum_right_f25899_2_0': 6637.434237308429,
    'volume_of_grey_matter_in_vi_cerebellum_vermis_f25898_2_0': 1507.4173147270114,
    'volume_of_grey_matter_in_viiia_cerebellum_left_f25909_2_0': 3816.072221024904,
    'volume_of_grey_matter_in_viiia_cerebellum_right_f25911_2_0': 3881.6708249521075,
    'volume_of_grey_matter_in_viiia_cerebellum_vermis_f25910_2_0': 903.5337245689655,
    'volume_of_grey_matter_in_viiib_cerebellum_left_f25912_2_0': 2682.8462961925293,
    'volume_of_grey_matter_in_viiib_cerebellum_right_f25914_2_0': 2725.981952250958,
    'volume_of_grey_matter_in_viiib_cerebellum_vermis_f25913_2_0': 444.4938587883142,
    'volume_of_grey_matter_in_viib_cerebellum_left_f25906_2_0': 3878.5710912356326,
    'volume_of_grey_matter_in_viib_cerebellum_right_f25908_2_0': 4123.628586661877,
    'volume_of_grey_matter_in_viib_cerebellum_vermis_f25907_2_0': 139.18211291666665,
    'volume_of_grey_matter_in_ventral_striatum_left_f25890_2_0': 553.1454708093869,
    'volume_of_grey_matter_in_ventral_striatum_right_f25891_2_0': 579.8904034482757,
    'volume_of_grey_matter_in_x_cerebellum_left_f25918_2_0': 491.9838716714559,
    'volume_of_grey_matter_in_x_cerebellum_vermis_f25919_2_0': 231.58319302921453,
    'volume_of_grey_matter_in_x_cerebellum_right_f25920_2_0': 452.5021734674329,
    'volume_of_av_left_hemisphere_f26684_2_0': 109.93122173850574,
    'volume_of_av_right_hemisphere_f26710_2_0': 124.55668724377395,
    'volume_of_accessorybasalnucleus_left_hemisphere_f26602_2_0': 245.95485646551722,
    'volume_of_accessorybasalnucleus_right_hemisphere_f26612_2_0': 263.2141912595785,
    'volume_of_anterioramygdaloidareaaaa_left_hemisphere_f26603_2_0': 48.05749210488506,
    'volume_of_anterioramygdaloidareaaaa_right_hemisphere_f26613_2_0': 52.882797104885064,
    'volume_of_basalnucleus_left_hemisphere_f26601_2_0': 413.6360530411877,
    'volume_of_basalnucleus_right_hemisphere_f26611_2_0': 436.10156431992334,
    'volume_of_ca1body_left_hemisphere_f26622_2_0': 143.1933341954023,
    'volume_of_ca1body_right_hemisphere_f26644_2_0': 154.9515501340996,
    'volume_of_ca1head_left_hemisphere_f26626_2_0': 558.6239953304598,
    'volume_of_ca1head_right_hemisphere_f26648_2_0': 584.6132437739464,
    'volume_of_ca3body_left_hemisphere_f26632_2_0': 110.52813864703066,
    'volume_of_ca3body_right_hemisphere_f26654_2_0': 122.70481995450191,
    'volume_of_ca3head_left_hemisphere_f26637_2_0': 151.71085433189654,
    'volume_of_ca3head_right_hemisphere_f26659_2_0': 161.69286401580462,
    'volume_of_ca4body_left_hemisphere_f26635_2_0': 139.03582993295018,
    'volume_of_ca4body_right_hemisphere_f26657_2_0': 146.7074882926245,
    'volume_of_ca4head_left_hemisphere_f26634_2_0': 143.5053952538314,
    'volume_of_ca4head_right_hemisphere_f26656_2_0': 148.56510199233716,
    'volume_of_cl_left_hemisphere_f26681_2_0': 29.61296826628352,
    'volume_of_cl_right_hemisphere_f26706_2_0': 30.440316733716468,
    'volume_of_cm_left_hemisphere_f26670_2_0': 235.56546032088121,
    'volume_of_cm_right_hemisphere_f26694_2_0': 230.79288735632184,
    'volume_of_cem_left_hemisphere_f26677_2_0': 49.564354372605365,
    'volume_of_cem_right_hemisphere_f26703_2_0': 53.78027873084291,
    'volume_of_centralnucleus_left_hemisphere_f26604_2_0': 48.554767459291185,
    'volume_of_centralnucleus_right_hemisphere_f26614_2_0': 52.4162367887931,
    'volume_of_corticalnucleus_left_hemisphere_f26606_2_0': 25.02213168893678,
    'volume_of_corticalnucleus_right_hemisphere_f26616_2_0': 28.60250647078544,
    'volume_of_corticoamygdaloidtransitio_left_hemisphere_f26607_2_0': 166.32952703783525,
    'volume_of_corticoamygdaloidtransitio_right_hemisphere_f26617_2_0': 174.26640607998084,
    'volume_of_gcmldgbody_left_hemisphere_f26633_2_0': 161.0556441235632,
    'volume_of_gcmldgbody_right_hemisphere_f26655_2_0': 167.67636515086207,
    'volume_of_gcmldghead_left_hemisphere_f26631_2_0': 173.97625420019156,
    'volume_of_gcmldghead_right_hemisphere_f26653_2_0': 180.88468511733717,
    'volume_of_hata_left_hemisphere_f26638_2_0': 59.31331937883142,
    'volume_of_hata_right_hemisphere_f26660_2_0': 64.29340893678162,
    'volume_of_hippocampaltail_left_hemisphere_f26620_2_0': 571.5356441570881,
    'volume_of_hippocampaltail_right_hemisphere_f26642_2_0': 596.392864559387,
    'volume_of_lsg_left_hemisphere_f26668_2_0': 22.302168262212646,
    'volume_of_lsg_right_hemisphere_f26692_2_0': 19.374975247605366,
    'volume_of_ld_left_hemisphere_f26712_2_0': 22.743955342361108,
    'volume_of_ld_right_hemisphere_f26713_2_0': 22.343662531369734,
    'volume_of_lgn_left_hemisphere_f26665_2_0': 159.32070577586208,
    'volume_of_lgn_right_hemisphere_f26688_2_0': 185.03523672413795,
    'volume_of_lp_left_hemisphere_f26687_2_0': 116.62438316331419,
    'volume_of_lp_right_hemisphere_f26711_2_0': 105.95226964798852,
    'volume_of_lateralnucleus_left_hemisphere_f26600_2_0': 641.4885287835249,
    'volume_of_lateralnucleus_right_hemisphere_f26610_2_0': 662.0358525862069,
    'volume_of_mdl_left_hemisphere_f26676_2_0': 256.50752382662836,
    'volume_of_mdl_right_hemisphere_f26700_2_0': 261.35142818486594,
    'volume_of_mdm_left_hemisphere_f26673_2_0': 665.7699721743294,
    'volume_of_mdm_right_hemisphere_f26697_2_0': 668.6192484913795,
    'volume_of_mgn_left_hemisphere_f26664_2_0': 111.70593910919541,
    'volume_of_mgn_right_hemisphere_f26689_2_0': 114.97343308908046,
    'volume_of_mvre_left_hemisphere_f26679_2_0': 8.241431588122603,
    'volume_of_mvre_right_hemisphere_f26702_2_0': 7.916214710967433,
    'volume_of_medialnucleus_left_hemisphere_f26605_2_0': 19.90029499161878,
    'volume_of_medialnucleus_right_hemisphere_f26615_2_0': 24.00992480316092,
    'volume_of_medulla_whole_brain_f26716_2_0': 4663.477644875478,
    'volume_of_midbrain_whole_brain_f26719_2_0': 6064.66280842912,
    'volume_of_paralaminarnucleus_left_hemisphere_f26608_2_0': 49.83722413793103,
    'volume_of_paralaminarnucleus_right_hemisphere_f26618_2_0': 52.312634420498085,
    'volume_of_pc_left_hemisphere_f26685_2_0': 2.987544899090038,
    'volume_of_pc_right_hemisphere_f26708_2_0': 3.039167497868774,
    'volume_of_pf_left_hemisphere_f26674_2_0': 57.757842399425286,
    'volume_of_pf_right_hemisphere_f26698_2_0': 58.15735886733716,
    'volume_of_pons_whole_brain_f26717_2_0': 15201.601240181993,
    'volume_of_pt_left_hemisphere_f26683_2_0': 6.338298745210728,
    'volume_of_pt_right_hemisphere_f26709_2_0': 5.892044485392719,
    'volume_of_pua_left_hemisphere_f26672_2_0': 197.6359312092912,
    'volume_of_pua_right_hemisphere_f26696_2_0': 220.2351295258621,
    'volume_of_pui_left_hemisphere_f26666_2_0': 158.28681052442533,
    'volume_of_pui_right_hemisphere_f26690_2_0': 183.75190429118774,
    'volume_of_pul_left_hemisphere_f26682_2_0': 161.38645214080458,
    'volume_of_pul_right_hemisphere_f26705_2_0': 193.0972446599617,
    'volume_of_pum_left_hemisphere_f26667_2_0': 910.2953481321839,
    'volume_of_pum_right_hemisphere_f26691_2_0': 972.8925136494253,
    'volume_of_scp_whole_brain_f26718_2_0': 282.23874363026823,
    'volume_of_va_left_hemisphere_f26678_2_0': 360.00836077586206,
    'volume_of_va_right_hemisphere_f26701_2_0': 363.33714875478927,
    'volume_of_vamc_left_hemisphere_f26675_2_0': 26.76921195881226,
    'volume_of_vamc_right_hemisphere_f26699_2_0': 27.982293199233716,
    'volume_of_vla_left_hemisphere_f26671_2_0': 574.3993635296935,
    'volume_of_vla_right_hemisphere_f26695_2_0': 592.8655079262453,
    'volume_of_vlp_left_hemisphere_f26686_2_0': 756.7288857040229,
    'volume_of_vlp_right_hemisphere_f26707_2_0': 774.1216143917624,
    'volume_of_vm_left_hemisphere_f26680_2_0': 18.77943698443487,
    'volume_of_vm_right_hemisphere_f26704_2_0': 18.81103822437739,
    'volume_of_vpl_left_hemisphere_f26669_2_0': 798.3065539032567,
    'volume_of_vpl_right_hemisphere_f26693_2_0': 800.7542744731802,
    'volume_of_wholeamygdala_left_hemisphere_f26609_2_0': 1658.780882519157,
    'volume_of_wholeamygdala_right_hemisphere_f26619_2_0': 1745.842115565134,
    'volume_of_wholebrainstem_whole_brain_f26720_2_0': 26211.980687260537,
    'volume_of_wholehippocampalbody_left_hemisphere_f26639_2_0': 1289.803384363027,
    'volume_of_wholehippocampalbody_right_hemisphere_f26661_2_0': 1320.8464958572797,
    'volume_of_wholehippocampalhead_left_hemisphere_f26640_2_0': 1803.0921332614942,
    'volume_of_wholehippocampalhead_right_hemisphere_f26662_2_0': 1866.0917801245212,
    'volume_of_wholehippocampus_left_hemisphere_f26641_2_0': 3664.4311391283527,
    'volume_of_wholehippocampus_right_hemisphere_f26663_2_0': 3783.331148227969,
    'volume_of_wholethalamus_left_hemisphere_f26714_2_0': 5817.570126915709,
    'volume_of_wholethalamus_right_hemisphere_f26715_2_0': 6040.073895354406,
    'volume_of_fimbria_left_hemisphere_f26636_2_0': 65.38483652658046,
    'volume_of_fimbria_right_hemisphere_f26658_2_0': 59.65855792227011,
    'volume_of_hippocampalfissure_left_hemisphere_f26624_2_0': 138.49003076867817,
    'volume_of_hippocampalfissure_right_hemisphere_f26646_2_0': 147.77417095545977,
    'volume_of_molecularlayerhpbody_left_hemisphere_f26630_2_0': 209.53463333333335,
    'volume_of_molecularlayerhpbody_right_hemisphere_f26652_2_0': 217.91501872844827,
    'volume_of_molecularlayerhphead_left_hemisphere_f26629_2_0': 305.83145026341,
    'volume_of_molecularlayerhphead_right_hemisphere_f26651_2_0': 311.6732665708812,
    'volume_of_parasubiculum_left_hemisphere_f26628_2_0': 67.23662710488507,
    'volume_of_parasubiculum_right_hemisphere_f26650_2_0': 66.87338363266285,
    'volume_of_presubiculumbody_left_hemisphere_f26627_2_0': 183.4751362308429,
    'volume_of_presubiculumbody_right_hemisphere_f26649_2_0': 170.8626309698276,
    'volume_of_presubiculumhead_left_hemisphere_f26625_2_0': 146.28436965038316,
    'volume_of_presubiculumhead_right_hemisphere_f26647_2_0': 148.01619085488508,
    'volume_of_subiculumbody_left_hemisphere_f26621_2_0': 277.59581755268204,
    'volume_of_subiculumbody_right_hemisphere_f26643_2_0': 280.37006619492337,
    'volume_of_subiculumhead_left_hemisphere_f26623_2_0': 196.60987171455938,
    'volume_of_subiculumhead_right_hemisphere_f26645_2_0': 199.47964090756705,
}


def get_positional_var(samples_sequence, Z_vals):
        
    # Get the number of subtypes
    N_S = samples_sequence.shape[0]

    # Get the number of features/biomarkers
    N_bio = Z_vals.shape[0]

    # Unravel the stage zscores from Z_vals
    stage_zscore = Z_vals.T.flatten()
    IX_select = np.nonzero(stage_zscore)[0]
    stage_zscore = stage_zscore[IX_select][None, :]

    # Get the z-scores and their number
    zvalues = np.unique(stage_zscore)
    N_z = len(zvalues)

    # Extract which biomarkers have which zscores/stages
    stage_biomarker_index = np.tile(np.arange(N_bio), (N_z,))
    stage_biomarker_index = stage_biomarker_index[IX_select]

    # Z-score colour definition. 
    colour_mat = np.array([[1, 0, 0], [1, 0, 1], [0, 0, 1], [0.5, 0, 1], [0, 1, 1], [0, 1, 0.5]])[:N_z]

    if colour_mat.shape[0] > N_z:
        raise ValueError(f"Colours are only defined for {len(colour_mat)} z-scores!")

    confus_matrices_per_subtype = []
    confus_matrices_colours_per_subtype = []

    for i in range(N_S):
        
        this_samples_sequence = samples_sequence[i,:,:].T
        N = this_samples_sequence.shape[1]

        # Construct confusion matrix (vectorized)
        # We compare `this_samples_sequence` against each position
        # Sum each time it was observed at that point in the sequence
        # And normalize for number of samples/sequences
        confus_matrix = (this_samples_sequence==np.arange(N)[:, None, None]).sum(1) / this_samples_sequence.shape[0]

        # Define the confusion matrix to insert the colours
        # Use 1s to start with all white
        confus_matrix_c = np.ones((N_bio, N, 3))

        # Loop over each z-score event
        for j, z in enumerate(zvalues):

            # Determine which colours to alter
            # I.e. red (1,0,0) means removing green & blue channels
            # according to the certainty of red (representing z-score 1)
            alter_level = colour_mat[j] == 0

            # Extract the uncertainties for this z-score
            confus_matrix_zscore = confus_matrix[(stage_zscore==z)[0]]

            # Subtract the certainty for this colour
            confus_matrix_c[np.ix_(stage_biomarker_index[(stage_zscore==z)[0]], range(N), alter_level)] -= np.tile(
                confus_matrix_zscore.reshape((stage_zscore==z).sum(), N, 1),
                (1, 1, alter_level.sum())
            )

        confus_matrices_colours_per_subtype.append(confus_matrix_c)
        confus_matrices_per_subtype.append(confus_matrix)

    return confus_matrices_per_subtype, confus_matrices_colours_per_subtype




if __name__ == '__main__':

    with open(sustain_output_dir, 'rb') as fp:
        sustain_output = pickle.load(fp)

    confus_matrices_per_subtype, confus_matrices_colours_per_subtype = get_positional_var(sustain_output['samples_sequence'], sustain_output['Z_vals'])

    biobank_to_oxford_brain_region_lookup = {v: k for k, v in oxford_to_biobank_brain_region_lookup.items()}



subtypes = range(len(sustain_output['samples_sequence']))

type_counts = Counter([int(x) for x in sustain_output['ml_subtype'].flatten()])

if len(subtypes) <= 3:
    rows=1
    cols=len(subtypes)
else:
    assert len(subtypes) <= 6
    rows = 2
    cols = len(subtypes)/2

fig = make_subplots(
    rows = rows,
    cols = cols,
    specs=[[{"type": "scatter3d"}, {"type": "scatter3d"}]],
    subplot_titles=tuple([f'Subtype {x+1} (n = {type_counts[x]})' for x in subtypes]),
    )



for subtype in subtypes:

    if subtype <= 2:
        row = 1
        col = subtype+1
    else:
        row = 2
        col = subtype-3+1


    print('subtype', subtype)
    modes = np.array(mode(sustain_output['samples_sequence'][subtype], axis=1)[0]).flatten()
    medians = np.median(sustain_output['samples_sequence'][subtype], axis=1)
    medians = medians.reshape(sustain_output['Z_vals'].shape, order='F')


    for pos, r in enumerate(medians):

        try:
            assert sorted(r) == list(r)
        except AssertionError:
            print(pos, r)


    z_vals_per_stage = {}

    for stage in range(sustain_output['samples_sequence'][subtype].shape[0]):

        z_vals = []

        for br_prog, br_z in zip(medians, sustain_output['Z_vals']):
            br_prog = [0] + list(br_prog)
            br_z = [0] + list(br_z)
            ix = br_prog.index(max([x for x in br_prog if stage>=x]))
            z_val = round(br_z[ix], 2)
            z_vals.append(z_val)
        
        z_vals_per_stage[stage+1] = z_vals

    num_stages = len(z_vals_per_stage.keys())
    print(num_stages)

    regions_plotted = [x for x in sustain_output['region_names'] if x in biobank_to_oxford_brain_region_lookup]
    ix_regions_included = [sustain_output['region_names'].index(x) for x in regions_plotted]

    medians = medians[ix_regions_included, :]
    Z_vals = sustain_output['Z_vals'][ix_regions_included, :]

    rel_locs = [brain_regions_xyz_lookup[biobank_to_oxford_brain_region_lookup[reg]] for reg in regions_plotted]
    x_coords, y_coords, z_coords = zip(*rel_locs)
    raw_sizes = [mean_volumes_from_all_ukbiobank_data[reg] for reg in regions_plotted]

    sizes = [x/max(raw_sizes)*100 for x in raw_sizes]
    labels = [biobank_to_oxford_brain_region_lookup[reg] for reg in regions_plotted]

    frames = []
    # Add traces, one for each slider step
    for stage in z_vals_per_stage.keys():

        _ = fig.add_trace(
            go.Scatter3d(
                visible=False, 
                x=x_coords, 
                y=y_coords, 
                z=z_coords, 
                mode='markers',
                text=labels,
                textposition="top center",
                marker=dict(
                    size=sizes,
                    color=z_vals_per_stage[stage],           # set color to an array/list of desired values
                    coloraxis="coloraxis",
                    #colorscale='Viridis',   # choose a colorscale
                    #color_continuous_scale=[(0, "blue"), (0.43, "green"), (0.86, "red"), (1.38, 'orange')],
                    opacity=0.8
                )

                ),

                row=row,
                col=col)

    # Make starting trace visible
    fig.data[subtype*num_stages].visible = True


# Create and add slider
steps = []
for i in range(num_stages):
    step = dict(
        method="update",
        args=[{"visible": [False] * len(fig.data)},
            {"title": "SUSTAIN Stage: " + str(i+1)}],  # layout attribute
    )

    for subtype in subtypes:
        step["args"][0]["visible"][i+num_stages*subtype] = True  # Toggle i'th trace to "visible"

    steps.append(step)

sliders = [dict(
    active=0,
    currentvalue={},#"prefix": "SUSTAIN Stage: "},
    pad={"t": 50},
    steps=steps
)]

_ = fig.update_layout(sliders=sliders, coloraxis=dict(colorscale='Bluered_r',cmin=0, cmax = sustain_output['Z_vals'].max()), showlegend=False)

fig.show()



'''For 3D sync - didn't work though

import plotly.graph_objs as go
import pandas as pd
from plotly import tools

# Read data from a csv
z_data = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/api_docs/mt_bruno_elevation.csv')

trace1 = dict(type='surface', scene='scene1', z=z_data.values)

trace2 = dict(type='surface', scene='scene2', z=z_data.values, colorscale='viridis')

f= tools.make_subplots(rows=1, cols=2, specs=[[{'is_3d': True}, {'is_3d': True}]])

f.append_trace(trace1, 1, 1)
f.append_trace(trace2, 1, 2)

fig = go.FigureWidget(f)

def cam_change(layout, camera):
    fig.layout.scene2.camera = camera

fig.layout.scene1.on_change(cam_change, 'camera')
'''






'''
import numpy as np
import plotly.graph_objects as go

x = [1,3,4,5,6,2,6]
y = [6,2,3,7,1,3,6]
z = [4,1,7,3,8,4,3]
c = [
    [1,4,8,3,5,2,3],
    [1,4,2,3,1,2,3],
    [1,4,8,3,5,2,3],
    [1,7,8,4,5,2,3],
    [1,4,8,3,5,1,3],
    [8,4,8,2,5,2,3],
]



# Create figure
fig = go.Figure(go.Scatter3d(x=[], y=[], z=[], mode="markers", marker=dict(size=10)))

frames = [go.Frame(data = [go.Scatter3d(x=x,y=y, z=z, marker=dict(color = c[k]))], traces= [1], name=f'frame{k}') for k  in  range(len(x)-1)]

fig.update(frames=frames)

sliders = [
    {"pad": {"b": 10, "t": 60},
     "len": 0.9,
     "x": 0.1,
     "y": 0,
     
     "steps": [
                 {"args": [[f.name], {
                                        "frame": {"duration": 0},
                                        "mode": "immediate",
                                        "fromcurrent": True,
                                        "transition": {"duration": 0, "easing": "linear"},
                                        }
                                        ],
                  "label": str(k),
                  "method": "animate",
                  } for k, f in enumerate(fig.frames)
              ]
     }
        ]

fig.update_layout(sliders=sliders, coloraxis={"colorscale": [(0, "red"), (0.5, "yellow"), (1, "green")]})

fig.show()


from plotly.subplots import make_subplots
import plotly.graph_objects as go

fig = make_subplots(rows=1, cols=2, shared_yaxes=True)

fig.add_trace(go.Bar(x=[1, 2, 3], y=[4, 5, 6],
                    marker=dict(color=[4, 5, 6], coloraxis="coloraxis")),
              1, 1)

fig.add_trace(go.Bar(x=[1, 2, 3], y=[2, 3, 5],
                    marker=dict(color=[2, 3, 5], coloraxis="coloraxis")),
              1, 2)

fig.update_layout(coloraxis=dict(colorscale='Bluered_r'), showlegend=False)
fig.show()
'''






















