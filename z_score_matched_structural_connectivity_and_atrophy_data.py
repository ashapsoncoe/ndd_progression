import json
import pandas as pd
import os
import zipfile
import numpy as np
import itertools
import random


all_struc_conn_path = '/home_dir/Connectome_aparc_and_Tian_Subcortex_S1_3T_all_case_controls_15082024'
all_atrophy_data_dir = '/home_dir/UKBW_78867r674582_AlexConnectome_CaseControl_v150824_aparc.csv'
sc_label_order_dir = '/home_dir/UKB-connectomics-main/data/templates/atlases/labels'
save_path = '/home_dir/'
parcellation = 'aparc'
adjust_for_tiv = False
right_handed_only = False

keys_to_match = [
        'R_Sex_f31', 
        'R_Age_f21003_2',
        'R_Ethnicity_f21000',
        'R_Handedness_f1707_0',
        'R_Centre_f54_2',
    ]


subcortical_sc_and_fc_to_biobank_labels = {
    'HIP-rh':  'right_hippocampus',
    'AMY-rh': 'right_amygdala',
    'pTHA-rh': 'right_posterior_thalamus',  # note no biobank label for this
    'aTHA-rh':  'right_anterior_thalamus', # note no biobank label for this
    'THA-rh': 'right_thalamus',
    'NAc-rh':  'right_accumbens',
    'GP-rh':  'right_pallidum',
    'PUT-rh':  'right_putamen',
    'CAU-rh':  'right_caudate',
    'HIP-lh': 'left_hippocampus',
    'AMY-lh':  'left_amygdala',
    'pTHA-lh':  'left_posterior_thalamus', # note no biobank label for this
    'aTHA-lh':  'left_anterior_thalamus', # note no biobank label for this
    'THA-lh':   'left_thalamus',
    'NAc-lh':  'left_accumbens',
    'GP-lh':  'left_pallidum',
    'PUT-lh':  'left_putamen',
    'CAU-lh': 'left_caudate',

}

biobank_labels_to_mics = {
    'aparc': { # unmatched from MICS: medial_wall, ctx-lh-corpuscallosum, ctx-lh-temporalpole, medial_wall, ctx-rh-corpuscallosum, ctx-rh-temporalpole
        'bankssts_left_hemisphere': 'ctx-lh-bankssts',
        'bankssts_right_hemisphere': 'ctx-rh-bankssts',
        'caudalanteriorcingulate_left_hemisphere': 'ctx-lh-caudalanteriorcingulate',
        'caudalanteriorcingulate_right_hemisphere': 'ctx-rh-caudalanteriorcingulate',
        'caudalmiddlefrontal_left_hemisphere': 'ctx-lh-caudalmiddlefrontal',
        'caudalmiddlefrontal_right_hemisphere': 'ctx-rh-caudalmiddlefrontal',
        'cuneus_left_hemisphere': 'ctx-lh-cuneus',
        'cuneus_right_hemisphere': 'ctx-rh-cuneus',
        'entorhinal_left_hemisphere': 'ctx-lh-entorhinal',
        'entorhinal_right_hemisphere': 'ctx-rh-entorhinal',
        'frontalpole_left_hemisphere': 'ctx-lh-frontalpole',
        'frontalpole_right_hemisphere': 'ctx-rh-frontalpole',
        'fusiform_left_hemisphere': 'ctx-lh-fusiform',
        'fusiform_right_hemisphere': 'ctx-rh-fusiform',
        'inferiorparietal_left_hemisphere': 'ctx-lh-inferiorparietal',
        'inferiorparietal_right_hemisphere': 'ctx-rh-inferiorparietal',
        'inferiortemporal_left_hemisphere': 'ctx-lh-inferiortemporal',
        'inferiortemporal_right_hemisphere': 'ctx-rh-inferiortemporal',
        'insula_left_hemisphere': 'ctx-lh-insula',
        'insula_right_hemisphere': 'ctx-rh-insula',
        'isthmuscingulate_left_hemisphere': 'ctx-lh-isthmuscingulate',
        'isthmuscingulate_right_hemisphere': 'ctx-rh-isthmuscingulate',
        'lateraloccipital_left_hemisphere': 'ctx-lh-lateraloccipital',
        'lateraloccipital_right_hemisphere': 'ctx-rh-lateraloccipital',
        'lateralorbitofrontal_left_hemisphere': 'ctx-lh-lateralorbitofrontal',
        'lateralorbitofrontal_right_hemisphere': 'ctx-rh-lateralorbitofrontal',
        'lingual_left_hemisphere': 'ctx-lh-lingual',
        'lingual_right_hemisphere': 'ctx-rh-lingual',
        'medialorbitofrontal_left_hemisphere': 'ctx-lh-medialorbitofrontal',
        'medialorbitofrontal_right_hemisphere': 'ctx-rh-medialorbitofrontal',
        'middletemporal_left_hemisphere': 'ctx-lh-middletemporal',
        'middletemporal_right_hemisphere': 'ctx-rh-middletemporal',
        'paracentral_left_hemisphere': 'ctx-lh-paracentral',
        'paracentral_right_hemisphere': 'ctx-rh-paracentral',
        'parahippocampal_left_hemisphere': 'ctx-lh-parahippocampal',
        'parahippocampal_right_hemisphere': 'ctx-rh-parahippocampal',
        'parsopercularis_left_hemisphere': 'ctx-lh-parsopercularis',
        'parsopercularis_right_hemisphere': 'ctx-rh-parsopercularis',
        'parsorbitalis_left_hemisphere': 'ctx-lh-parsorbitalis',
        'parsorbitalis_right_hemisphere': 'ctx-rh-parsorbitalis',
        'parstriangularis_left_hemisphere': 'ctx-lh-parstriangularis',
        'parstriangularis_right_hemisphere': 'ctx-rh-parstriangularis',
        'pericalcarine_left_hemisphere': 'ctx-lh-pericalcarine',
        'pericalcarine_right_hemisphere': 'ctx-rh-pericalcarine',
        'postcentral_left_hemisphere': 'ctx-lh-postcentral',
        'postcentral_right_hemisphere': 'ctx-rh-postcentral',
        'posteriorcingulate_left_hemisphere': 'ctx-lh-posteriorcingulate',
        'posteriorcingulate_right_hemisphere': 'ctx-rh-posteriorcingulate',
        'precentral_left_hemisphere': 'ctx-lh-precentral',
        'precentral_right_hemisphere': 'ctx-rh-precentral',
        'precuneus_left_hemisphere': 'ctx-lh-precuneus',
        'precuneus_right_hemisphere': 'ctx-rh-precuneus',
        'rostralanteriorcingulate_left_hemisphere': 'ctx-lh-rostralanteriorcingulate',
        'rostralanteriorcingulate_right_hemisphere': 'ctx-rh-rostralanteriorcingulate',
        'rostralmiddlefrontal_left_hemisphere': 'ctx-lh-rostralmiddlefrontal',
        'rostralmiddlefrontal_right_hemisphere': 'ctx-rh-rostralmiddlefrontal',
        'superiorfrontal_left_hemisphere': 'ctx-lh-superiorfrontal',
        'superiorfrontal_right_hemisphere': 'ctx-rh-superiorfrontal',
        'superiorparietal_left_hemisphere': 'ctx-lh-superiorparietal',
        'superiorparietal_right_hemisphere': 'ctx-rh-superiorparietal',
        'superiortemporal_left_hemisphere': 'ctx-lh-superiortemporal',
        'superiortemporal_right_hemisphere': 'ctx-rh-superiortemporal',
        'supramarginal_left_hemisphere': 'ctx-lh-supramarginal',
        'supramarginal_right_hemisphere': 'ctx-rh-supramarginal',
        'transversetemporal_left_hemisphere': 'ctx-lh-transversetemporal',
        'transversetemporal_right_hemisphere': 'ctx-rh-transversetemporal',
    },
    'aparc-a2009s': {   #Unmatched from MICS: 'ctx_rh_Medial_wall', 'ctx_lh_Medial_wall'
        'gscingulant_left_hemisphere': 'ctx_lh_G_and_S_cingul-Ant',
        'gscingulant_right_hemisphere': 'ctx_rh_G_and_S_cingul-Ant',
        'gscingulmidant_left_hemisphere': 'ctx_lh_G_and_S_cingul-Mid-Ant',
        'gscingulmidant_right_hemisphere': 'ctx_rh_G_and_S_cingul-Mid-Ant',
        'gscingulmidpost_left_hemisphere': 'ctx_lh_G_and_S_cingul-Mid-Post',
        'gscingulmidpost_right_hemisphere': 'ctx_rh_G_and_S_cingul-Mid-Post',
        'gsfrontomargin_left_hemisphere': 'ctx_lh_G_and_S_frontomargin',
        'gsfrontomargin_right_hemisphere': 'ctx_rh_G_and_S_frontomargin',
        'gsoccipitalinf_left_hemisphere': 'ctx_lh_G_and_S_occipital_inf',
        'gsoccipitalinf_right_hemisphere': 'ctx_rh_G_and_S_occipital_inf',
        'gsparacentral_left_hemisphere': 'ctx_lh_G_and_S_paracentral',
        'gsparacentral_right_hemisphere': 'ctx_rh_G_and_S_paracentral',
        'gssubcentral_left_hemisphere': 'ctx_lh_G_and_S_subcentral',
        'gssubcentral_right_hemisphere': 'ctx_rh_G_and_S_subcentral',
        'gstransvfrontopol_left_hemisphere': 'ctx_lh_G_and_S_transv_frontopol',
        'gstransvfrontopol_right_hemisphere': 'ctx_rh_G_and_S_transv_frontopol',
        'ginslgscentins_left_hemisphere': 'ctx_lh_G_Ins_lg_and_S_cent_ins',
        'ginslgscentins_right_hemisphere': 'ctx_rh_G_Ins_lg_and_S_cent_ins',
        'gcingulpostdorsal_left_hemisphere': 'ctx_lh_G_cingul-Post-dorsal',
        'gcingulpostdorsal_right_hemisphere': 'ctx_rh_G_cingul-Post-dorsal',
        'gcingulpostventral_left_hemisphere': 'ctx_lh_G_cingul-Post-ventral',
        'gcingulpostventral_right_hemisphere': 'ctx_rh_G_cingul-Post-ventral',
        'gcuneus_left_hemisphere': 'ctx_lh_G_cuneus',
        'gcuneus_right_hemisphere': 'ctx_rh_G_cuneus',
        'gfrontinfopercular_left_hemisphere': 'ctx_lh_G_front_inf-Opercular',
        'gfrontinfopercular_right_hemisphere': 'ctx_rh_G_front_inf-Opercular',
        'gfrontinforbital_left_hemisphere': 'ctx_lh_G_front_inf-Orbital',
        'gfrontinforbital_right_hemisphere': 'ctx_rh_G_front_inf-Orbital',
        'gfrontinftriangul_left_hemisphere': 'ctx_lh_G_front_inf-Triangul',
        'gfrontinftriangul_right_hemisphere': 'ctx_rh_G_front_inf-Triangul',
        'gfrontmiddle_left_hemisphere': 'ctx_lh_G_front_middle',
        'gfrontmiddle_right_hemisphere': 'ctx_rh_G_front_middle',
        'gfrontsup_left_hemisphere': 'ctx_lh_G_front_sup',
        'gfrontsup_right_hemisphere': 'ctx_rh_G_front_sup',
        'ginsularshort_left_hemisphere': 'ctx_lh_G_insular_short',
        'ginsularshort_right_hemisphere': 'ctx_rh_G_insular_short',
        'goctemplatfusifor_left_hemisphere': 'ctx_lh_G_oc-temp_lat-fusifor',
        'goctemplatfusifor_right_hemisphere': 'ctx_rh_G_oc-temp_lat-fusifor',
        'goctempmedlingual_left_hemisphere': 'ctx_lh_G_oc-temp_med-Lingual',
        'goctempmedlingual_right_hemisphere': 'ctx_rh_G_oc-temp_med-Lingual',
        'goctempmedparahip_left_hemisphere': 'ctx_lh_G_oc-temp_med-Parahip',
        'goctempmedparahip_right_hemisphere': 'ctx_rh_G_oc-temp_med-Parahip',
        'goccipitalmiddle_left_hemisphere': 'ctx_lh_G_occipital_middle',
        'goccipitalmiddle_right_hemisphere': 'ctx_rh_G_occipital_middle',
        'goccipitalsup_left_hemisphere': 'ctx_lh_G_occipital_sup',
        'goccipitalsup_right_hemisphere': 'ctx_rh_G_occipital_sup',
        'gorbital_left_hemisphere': 'ctx_lh_G_orbital',
        'gorbital_right_hemisphere': 'ctx_rh_G_orbital',
        'gparietinfangular_left_hemisphere': 'ctx_lh_G_pariet_inf-Angular',
        'gparietinfangular_right_hemisphere': 'ctx_rh_G_pariet_inf-Angular',
        'gparietinfsupramar_left_hemisphere': 'ctx_lh_G_pariet_inf-Supramar',
        'gparietinfsupramar_right_hemisphere': 'ctx_rh_G_pariet_inf-Supramar',
        'gparietalsup_left_hemisphere': 'ctx_lh_G_parietal_sup',
        'gparietalsup_right_hemisphere': 'ctx_rh_G_parietal_sup',
        'gpostcentral_left_hemisphere': 'ctx_lh_G_postcentral',
        'gpostcentral_right_hemisphere': 'ctx_rh_G_postcentral',
        'gprecentral_left_hemisphere': 'ctx_lh_G_precentral',
        'gprecentral_right_hemisphere': 'ctx_rh_G_precentral',
        'gprecuneus_left_hemisphere': 'ctx_lh_G_precuneus',
        'gprecuneus_right_hemisphere': 'ctx_rh_G_precuneus',
        'grectus_left_hemisphere': 'ctx_lh_G_rectus',
        'grectus_right_hemisphere': 'ctx_rh_G_rectus',
        'gsubcallosal_left_hemisphere': 'ctx_lh_G_subcallosal',
        'gsubcallosal_right_hemisphere': 'ctx_rh_G_subcallosal',
        'gtempsupgttransv_left_hemisphere': 'ctx_lh_G_temp_sup-G_T_transv',
        'gtempsupgttransv_right_hemisphere': 'ctx_rh_G_temp_sup-G_T_transv',
        'gtempsuplateral_left_hemisphere': 'ctx_lh_G_temp_sup-Lateral',
        'gtempsuplateral_right_hemisphere': 'ctx_rh_G_temp_sup-Lateral',
        'gtempsupplanpolar_left_hemisphere': 'ctx_lh_G_temp_sup-Plan_polar',
        'gtempsupplanpolar_right_hemisphere': 'ctx_rh_G_temp_sup-Plan_polar',
        'gtempsupplantempo_left_hemisphere': 'ctx_lh_G_temp_sup-Plan_tempo',
        'gtempsupplantempo_right_hemisphere': 'ctx_rh_G_temp_sup-Plan_tempo',
        'gtemporalinf_left_hemisphere': 'ctx_lh_G_temporal_inf',
        'gtemporalinf_right_hemisphere': 'ctx_rh_G_temporal_inf',
        'gtemporalmiddle_left_hemisphere': 'ctx_lh_G_temporal_middle',
        'gtemporalmiddle_right_hemisphere': 'ctx_rh_G_temporal_middle',
        'latfisanthorizont_left_hemisphere': 'ctx_lh_Lat_Fis-ant-Horizont',
        'latfisanthorizont_right_hemisphere': 'ctx_rh_Lat_Fis-ant-Horizont',
        'latfisantvertical_left_hemisphere': 'ctx_lh_Lat_Fis-ant-Vertical',
        'latfisantvertical_right_hemisphere': 'ctx_rh_Lat_Fis-ant-Vertical',
        'latfispost_left_hemisphere': 'ctx_lh_Lat_Fis-post',
        'latfispost_right_hemisphere': 'ctx_rh_Lat_Fis-post',
        'poleoccipital_left_hemisphere': 'ctx_lh_Pole_occipital',
        'poleoccipital_right_hemisphere': 'ctx_rh_Pole_occipital',
        'poletemporal_left_hemisphere': 'ctx_lh_Pole_temporal',
        'poletemporal_right_hemisphere': 'ctx_rh_Pole_temporal',
        'scalcarine_left_hemisphere': 'ctx_lh_S_calcarine',
        'scalcarine_right_hemisphere': 'ctx_rh_S_calcarine',
        'scentral_left_hemisphere': 'ctx_lh_S_central',
        'scentral_right_hemisphere': 'ctx_rh_S_central',
        'scingulmarginalis_left_hemisphere': 'ctx_lh_S_cingul-Marginalis',
        'scingulmarginalis_right_hemisphere': 'ctx_rh_S_cingul-Marginalis',
        'scircularinsulaant_left_hemisphere': 'ctx_lh_S_circular_insula_ant',
        'scircularinsulaant_right_hemisphere': 'ctx_rh_S_circular_insula_ant',
        'scircularinsulainf_left_hemisphere': 'ctx_lh_S_circular_insula_inf',
        'scircularinsulainf_right_hemisphere': 'ctx_rh_S_circular_insula_inf',
        'scircularinsulasup_left_hemisphere': 'ctx_lh_S_circular_insula_sup',
        'scircularinsulasup_right_hemisphere': 'ctx_rh_S_circular_insula_sup',
        'scollattransvant_left_hemisphere': 'ctx_lh_S_collat_transv_ant',
        'scollattransvant_right_hemisphere': 'ctx_rh_S_collat_transv_ant',
        'scollattransvpost_left_hemisphere': 'ctx_lh_S_collat_transv_post',
        'scollattransvpost_right_hemisphere': 'ctx_rh_S_collat_transv_post',
        'sfrontinf_left_hemisphere': 'ctx_lh_S_front_inf',
        'sfrontinf_right_hemisphere': 'ctx_rh_S_front_inf',
        'sfrontmiddle_left_hemisphere': 'ctx_lh_S_front_middle',
        'sfrontmiddle_right_hemisphere': 'ctx_rh_S_front_middle',
        'sfrontsup_left_hemisphere': 'ctx_lh_S_front_sup',
        'sfrontsup_right_hemisphere': 'ctx_rh_S_front_sup',
        'sintermprimjensen_left_hemisphere': 'ctx_lh_S_interm_prim-Jensen',
        'sintermprimjensen_right_hemisphere': 'ctx_rh_S_interm_prim-Jensen',
        'sintraparietptrans_left_hemisphere': 'ctx_lh_S_intrapariet_and_P_trans',
        'sintraparietptrans_right_hemisphere': 'ctx_rh_S_intrapariet_and_P_trans',
        'socmiddlelunatus_left_hemisphere': 'ctx_lh_S_oc_middle_and_Lunatus',
        'socmiddlelunatus_right_hemisphere': 'ctx_rh_S_oc_middle_and_Lunatus',
        'socsuptransversal_left_hemisphere': 'ctx_lh_S_oc_sup_and_transversal',
        'socsuptransversal_right_hemisphere': 'ctx_rh_S_oc_sup_and_transversal',
        'soctemplat_left_hemisphere': 'ctx_lh_S_oc-temp_lat',
        'soctemplat_right_hemisphere': 'ctx_rh_S_oc-temp_lat',
        'soctempmedlingual_left_hemisphere': 'ctx_lh_S_oc-temp_med_and_Lingual',
        'soctempmedlingual_right_hemisphere': 'ctx_rh_S_oc-temp_med_and_Lingual',
        'soccipitalant_left_hemisphere': 'ctx_lh_S_occipital_ant',
        'soccipitalant_right_hemisphere': 'ctx_rh_S_occipital_ant',
        'sorbitalhshaped_left_hemisphere': 'ctx_lh_S_orbital-H_Shaped',
        'sorbitalhshaped_right_hemisphere': 'ctx_rh_S_orbital-H_Shaped',
        'sorbitallateral_left_hemisphere': 'ctx_lh_S_orbital_lateral',
        'sorbitallateral_right_hemisphere': 'ctx_rh_S_orbital_lateral',
        'sorbitalmedolfact_left_hemisphere': 'ctx_lh_S_orbital_med-olfact',
        'sorbitalmedolfact_right_hemisphere': 'ctx_rh_S_orbital_med-olfact',
        'sparietooccipital_left_hemisphere': 'ctx_lh_S_parieto_occipital',
        'sparietooccipital_right_hemisphere': 'ctx_rh_S_parieto_occipital',
        'spericallosal_left_hemisphere': 'ctx_lh_S_pericallosal',
        'spericallosal_right_hemisphere': 'ctx_rh_S_pericallosal',
        'spostcentral_left_hemisphere': 'ctx_lh_S_postcentral',
        'spostcentral_right_hemisphere': 'ctx_rh_S_postcentral',
        'sprecentralinfpart_left_hemisphere': 'ctx_lh_S_precentral-inf-part',
        'sprecentralinfpart_right_hemisphere': 'ctx_rh_S_precentral-inf-part',
        'sprecentralsuppart_left_hemisphere': 'ctx_lh_S_precentral-sup-part',
        'sprecentralsuppart_right_hemisphere': 'ctx_rh_S_precentral-sup-part',
        'ssuborbital_left_hemisphere': 'ctx_lh_S_suborbital',
        'ssuborbital_right_hemisphere': 'ctx_rh_S_suborbital',
        'ssubparietal_left_hemisphere': 'ctx_lh_S_subparietal',
        'ssubparietal_right_hemisphere': 'ctx_rh_S_subparietal',
        'stemporalinf_left_hemisphere': 'ctx_lh_S_temporal_inf',
        'stemporalinf_right_hemisphere': 'ctx_rh_S_temporal_inf',
        'stemporalsup_left_hemisphere': 'ctx_lh_S_temporal_sup',
        'stemporalsup_right_hemisphere': 'ctx_rh_S_temporal_sup',
        'stemporaltransverse_left_hemisphere': 'ctx_lh_S_temporal_transverse',
        'stemporaltransverse_right_hemisphere': 'ctx_rh_S_temporal_transverse',
    },  



    'subcortical': {
        'accumbens_left': 'Left-Accumbens-area',
        'accumbens_right': 'Right-Accumbens-area',
        'amygdala_left': 'Left-Amygdala',
        'amygdala_right': 'Right-Amygdala',
        'caudate_left': 'Left-Caudate',
        'caudate_right': 'Right-Caudate',
        'hippocampus_left': 'Left-Hippocampus',
        'hippocampus_right': 'Right-Hippocampus',
        'pallidum_left': 'Left-Pallidum',
        'pallidum_right': 'Right-Pallidum',
        'putamen_left': 'Left-Putamen',
        'putamen_right': 'Right-Putamen',
        'thalamus_left': 'Left-Thalamus-Proper',
        'thalamus_right': 'Right-Thalamus-Proper',
}
}



if __name__ == '__main__':

    # Load data
    all_cases_and_controls_df = pd.read_csv(all_atrophy_data_dir, index_col = 'EID78867', low_memory=False)   # Need to use EIDs78867 version to be compatible with downloaded connectivity data


    # Tidy up column names:
    all_regions = list(biobank_labels_to_mics[parcellation].keys()) + list(biobank_labels_to_mics['subcortical'].keys())

    all_regions_all_measures = []

    for c in all_cases_and_controls_df.columns:
        for m in ('area', 'volume', 'mean_thickness'):
            if m in c and c.replace(f'{m}_of_', '')[:-11] in all_regions:
                if m=='volume' and adjust_for_tiv:
                    all_cases_and_controls_df[c[:-11]] = all_cases_and_controls_df[c] / all_cases_and_controls_df['R_TIV_f25009f25003_2']
                else:
                    all_cases_and_controls_df.rename(columns={c: c[:-11]}, inplace=True)

                all_regions_all_measures.append(c[:-11])

    if right_handed_only:
            control_df = all_cases_and_controls_df.query(f"ML_C42C240Xf41270f20002_Dementia == 0 & R_Handedness_f1707_0 == 'Right-handed'")
            dementia_df = all_cases_and_controls_df.query(f"ML_C42C240Xf41270f20002_Dementia == 1 & R_Handedness_f1707_0 == 'Right-handed'")
    else:
        control_df = all_cases_and_controls_df.query(f"ML_C42C240Xf41270f20002_Dementia == 0")
        dementia_df = all_cases_and_controls_df.query(f"ML_C42C240Xf41270f20002_Dementia == 1")

    control_eids = set(control_df.index)
    dementia_eids = set(dementia_df.index)


    # Get SC labels:
    sc_region_names = []

    for f in (f'{parcellation}.label_list.txt', 'Tian_Subcortex_S1_3T_label.txt'):

        with open(f'{sc_label_order_dir}\\{f}', 'r') as file:
            for line in file:
                # Strip leading/trailing whitespace from the line
                line = line.strip()
                # Check if the line contains a region name
                if not line[0].isdigit() and len(line.split()) == 1:
                    sc_region_names.append(line)


    # Load structural connectivity, rename to biobank regional labels, average posterior and anterior thalamus SC

    all_struc_conn_data = {}

    for eid in dementia_eids | control_eids:

        if os.path.exists(f'{all_struc_conn_path}\\{eid}_31021_2_0.zip'):

            with zipfile.ZipFile(f'{all_struc_conn_path}\\{eid}_31021_2_0.zip', 'r') as zip_ref:

                for k in ('sift2_fbc', 'mean_FA'):

                    if not eid in all_struc_conn_data:
                        all_struc_conn_data[eid] = {}

                    with zip_ref.open(f'connectome_{k}_10M.csv', 'r') as f:

                        all_struc_conn_data[eid][k] = pd.read_csv(f, index_col=None, header=None, names=sc_region_names)
                        all_struc_conn_data[eid][k].index = sc_region_names

                        all_struc_conn_data[eid][k]['THA-lh'] = (all_struc_conn_data[eid][k]['aTHA-lh'] + all_struc_conn_data[eid][k]['pTHA-lh'] ) / 2
                        all_struc_conn_data[eid][k]['THA-rh'] = (all_struc_conn_data[eid][k]['aTHA-rh'] + all_struc_conn_data[eid][k]['pTHA-rh'] ) / 2
                        all_struc_conn_data[eid][k] = all_struc_conn_data[eid][k].drop(['aTHA-lh', 'pTHA-lh', 'aTHA-rh', 'pTHA-rh'], axis=1)

                        all_struc_conn_data[eid][k].loc['THA-lh'] = all_struc_conn_data[eid][k].loc[['aTHA-lh', 'pTHA-lh']].mean()
                        all_struc_conn_data[eid][k].loc['THA-rh'] = all_struc_conn_data[eid][k].loc[['aTHA-rh', 'pTHA-rh']].mean()
                        all_struc_conn_data[eid][k] = all_struc_conn_data[eid][k].drop(['aTHA-lh', 'pTHA-lh', 'aTHA-rh', 'pTHA-rh'], axis=0)

                        assert list(all_struc_conn_data[eid][k].columns) == list(all_struc_conn_data[eid][k].index)

                        new_names = [subcortical_sc_and_fc_to_biobank_labels[k] if k in subcortical_sc_and_fc_to_biobank_labels else k for k in all_struc_conn_data[eid][k].columns]
                        all_struc_conn_data[eid][k].columns = new_names
                        all_struc_conn_data[eid][k].index = new_names


    # For each case, z-score each stuctural connection and atrophy measure vs. controls. 

    z_scored_data = {x: {} for x in ('sc_mean_FA', 'sc_sift2_fbc', 'vol')} 

    all_regions = list(list(list(all_struc_conn_data.values())[0].values())[0].columns)

    random_matched_control_eids = set()
    case_ids_to_matched_control_ids = {}

    for pos, eid in enumerate(dementia_eids):

        print(f'Z scoring eid {pos+1} of {len(dementia_eids)}')

        # Z score data:
        vals = [all_cases_and_controls_df.loc[eid, x] for x in keys_to_match]

        query_string = ' & '.join([f"{x} == '{v}'" if type(v)==str else f"{x} == {v}" for x,v in zip(keys_to_match, vals)])

        matched_controls_this_indiv = control_df.query(query_string)

        if not eid in all_struc_conn_data: continue

        if len(matched_controls_this_indiv) < 10: continue

        if eid in matched_controls_this_indiv.index:
            matched_controls_this_indiv = matched_controls_this_indiv.drop([eid], axis=0)

        matched_eids = matched_controls_this_indiv.index

        random_control_eid = random.choice([x for x in matched_eids if x in all_struc_conn_data])
        random_matched_control_eids.add(random_control_eid)
        case_ids_to_matched_control_ids[eid] = random_control_eid


        for id1 in (eid, random_control_eid):

            for k in all_struc_conn_data[id1]:
                z_scored_data[f'sc_{k}'][id1] = pd.DataFrame(index = all_regions, columns=all_regions)

            
            z_scored_data['vol'][id1] = {}

            for pos, region1 in enumerate(all_regions):

                biobank_lookup_key  ='volume_of_' + region1.split('_')[1] + '_' + region1.split('_')[0] 

                if not region1 in subcortical_sc_and_fc_to_biobank_labels.values():

                    biobank_lookup_key = biobank_lookup_key + '_hemisphere' 

                if biobank_lookup_key in all_cases_and_controls_df.columns:

                    matched_region1_volumes = matched_controls_this_indiv[biobank_lookup_key]

                    z_scored_data['vol'][id1][region1] = (all_cases_and_controls_df.loc[id1, biobank_lookup_key] - np.mean(matched_region1_volumes)) / np.std(matched_region1_volumes)


                for region2 in all_regions:
                    
                    if id1 in all_struc_conn_data:

                        for k in all_struc_conn_data[id1]:

                            matched_values = [all_struc_conn_data[x][k].loc[region1, region2] for x in matched_eids]

                            if all_struc_conn_data[id1][k].loc[region1, region2] == 0 or set(matched_values) == {np.float64(0)}:
                                continue

                            z_scored_data[f'sc_{k}'][id1].loc[region1, region2] = (all_struc_conn_data[id1][k].loc[region1, region2] - np.mean(matched_values)) / np.std(matched_values)
                        

    # Combine all z-scored data into a dataframe and save to disc
    for dtype, selected_eids in (('all_dementia', dementia_eids), ('random_matched_controls', random_matched_control_eids)):

        eids_overlap = set.intersection(selected_eids, set(z_scored_data['sc_sift2_fbc'].keys()), set(z_scored_data['sc_mean_FA'].keys()), set(z_scored_data['vol'].keys())) # set(z_scored_data['fc'].keys()), set(z_scored_data['sc_streamline_count'].keys()),  removed

        
        # Assemble all data:
        all_data = []
        biomarker_names = []


        for eid in eids_overlap:

            this_eid_vol_names = [f'{x}_vol' for x in z_scored_data['vol'][eid].keys()]
            this_eid_vol_values  = list(z_scored_data['vol'][eid].values())

            this_eid_names = this_eid_vol_names 
            this_eid_data = this_eid_vol_values 

            for k in ('sift2_fbc', 'mean_FA'): 

                sc_combos = [(a,b) for a, b in itertools.combinations(z_scored_data[f'sc_{k}'][eid].columns, 2) if a!=b]

                this_eid_sc_names = [f'{a}_{b}_sc_{k}' for a,b in sc_combos]
                this_eid_sc_values = [z_scored_data[f'sc_{k}'][eid].loc[a,b] for a,b in sc_combos]

                this_eid_names.extend(this_eid_sc_names)
                this_eid_data.extend(this_eid_sc_values)

            biomarker_names.append(this_eid_names)
            all_data.append(this_eid_data)


        all_z_scored_df = pd.DataFrame(all_data, columns = biomarker_names[0])


        # Need to convert EIDs to EIDs version 59138 to be compatible with previous subtypes analysis
        eids_78867_to_59138 = {a:b for a,b in zip(all_cases_and_controls_df.index, all_cases_and_controls_df['EID59138'])}

        eids_overlap_59138_version = [eids_78867_to_59138[x] for x in eids_overlap]

        all_z_scored_df.index = eids_overlap_59138_version

        all_z_scored_df.to_csv(f'{save_path}\\all_volumetric_and_structural_connectivity_z_scores_{dtype}_{parcellation}.csv')


    case_ids_to_matched_control_ids = {eids_78867_to_59138[k]: eids_78867_to_59138[case_ids_to_matched_control_ids[k]] for k in case_ids_to_matched_control_ids}

    with open(f'{save_path}\\case_ids_to_matched_control_ids.json', 'w') as fp:
        json.dump(case_ids_to_matched_control_ids, fp)


