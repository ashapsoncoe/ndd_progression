import pandas as pd
import os
import pySuStaIn
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import StratifiedKFold


input_data_z_table_path =  '/data/home/wpw131/preclindementia152_23_regions_z_scores.csv' 
save_dir = '/data/home/wpw131/ebs_age_age2_TIV_centre_sep_sex_23_regions'  
all_data_path = '/data/home/wpw131/UKBW_EBM_78867r672482_070823.csv' 
dataset_name = 'preclin_AD_152_23_regions'
input_dataset_are_controls = False
use_multiple_processors = True
num_events_per_biomarker = 1
set_single_z_to_cases_mean = True
N_startpoints = 25
N_S_max = 4
N_iterations_MCMC = int(1e6)
N_folds = 10


if __name__ == '__main__':

    input_data_z_df = pd.read_csv(input_data_z_table_path, index_col=0)
    all_data_df = pd.read_csv(all_data_path)
    dementia_lookup = {all_data_df.at[i, 'EID']: all_data_df.at[i, 'ML_C42C240Xf41270f20002_Dementia'] for i in all_data_df.index}
    input_data_z_arr = np.array(input_data_z_df) 
    mri_fields = list(input_data_z_df.columns)

    # multiply data for decreasing biomarkers by -1, unless known controls, in which case multiply all biomarkers by -1
    if input_dataset_are_controls:
        IS_decreasing = np.mean(input_data_z_arr,axis=0)<np.inf
    else:
        IS_decreasing = np.mean(input_data_z_arr,axis=0)<0

    input_data_z_arr[np.tile(IS_decreasing,(input_data_z_arr.shape[0],1))] = -1*input_data_z_arr[np.tile(IS_decreasing,(input_data_z_arr.shape[0],1))]
   
    # Set Z-scores for each biomarker
    positive_vals_only_flat = input_data_z_arr[input_data_z_arr>0]
    pooled_95th_percentile = np.percentile(positive_vals_only_flat, 95)
    Z_max = np.array([pooled_95th_percentile]*input_data_z_arr.shape[1])  

    # If number of events per biomarker = 1, set to the mean of the cases:
    if num_events_per_biomarker == 1 and set_single_z_to_cases_mean:
        ix = [list(input_data_z_df.index).index(x) for x in input_data_z_df.index if dementia_lookup[x]==1]
        cases_only_arr = input_data_z_arr[ix]
        mean_each_biomarker = np.mean(input_data_z_arr[ix], axis=0)
        Z_vals = np.array([[np.mean(mean_each_biomarker)]]*input_data_z_arr.shape[1])  
    else:
        Z_vals = np.array([[np.percentile(positive_vals_only_flat, p) for p in np.arange(0, 100, 100/(num_events_per_biomarker+1))][1:]]*input_data_z_arr.shape[1]) 

    print('Z score events', Z_vals)


    output_folder = save_dir + f'//{dataset_name}_{num_events_per_biomarker}events_{N_startpoints}startpoints_{N_S_max}maxclusters_{N_iterations_MCMC}iterations_{input_data_z_arr.shape[1]}biomarkers'
    
    if set_single_z_to_cases_mean == False:
        output_folder = output_folder + '_z_by_percentiles'
    
    print('Data will be saved in', output_folder)


    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    else:
        print('existing data found')

    sustain_input = pySuStaIn.ZscoreSustain(input_data_z_arr, Z_vals, Z_max, mri_fields, N_startpoints, N_S_max, N_iterations_MCMC, output_folder, dataset_name, use_multiple_processors)

    # run the sustain algorithm with the inputs set in sustain_input above
    s_output = sustain_input.run_sustain_algorithm()
    samples_sequence, samples_f, ml_subtype, prob_ml_subtype, ml_stage, prob_ml_stage, prob_subtype_stage = s_output

    # run cross validation
    print('starting cross validation')
    
    labels = 1 * np.ones(len(input_data_z_arr), dtype=int) 
    cv = StratifiedKFold(n_splits=N_folds, shuffle=True)
    cv_it = cv.split(input_data_z_arr, labels)
    f = [test for train, test in cv_it]
    test_idxs = np.array([x[:min([len(a) for a in f])] for x in f])

    CVIC, loglike_matrix = sustain_input.cross_validate_sustain_model(test_idxs)

    with open(f'{save_dir}/cross_validation_results_{N_folds}_fold.pkl', 'wb') as fp:
        pickle.dump([CVIC, loglike_matrix], fp)
