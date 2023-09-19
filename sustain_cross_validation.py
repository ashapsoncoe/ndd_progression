import pickle
import os
import pandas as pd
import numpy as np
import pySuStaIn
import numpy as np
from sklearn.model_selection import StratifiedKFold

save_dir = '/data/home/wpw131/ebs_age_age2_TIV_centre_sep_sex_23_regions/preclin_AD_152_23_regions_1events_25startpoints_4maxclusters_10000iterations_crossvalidation' 
original_data = '/data/home/wpw131/preclindementia152_23_regions_z_scores.csv' 
num_events_per_biomarker = 1
N_S_max  = 4
N_iterations_MCMC = 10000
startpoints = 25


if __name__ == '__main__':

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    original_df = pd.read_csv(original_data, index_col=0)

    # Re-obtain z-values:
    input_data_z_arr = np.array(original_df)
    IS_decreasing = np.mean(input_data_z_arr,axis=0)<0
    input_data_z_arr[np.tile(IS_decreasing,(input_data_z_arr.shape[0],1))] = -1*input_data_z_arr[np.tile(IS_decreasing,(input_data_z_arr.shape[0],1))]
    positive_vals_only_flat = input_data_z_arr[input_data_z_arr>0]
    pooled_95th_percentile = np.percentile(positive_vals_only_flat, 95)
    Z_max = np.array([pooled_95th_percentile]*input_data_z_arr.shape[1])  
    Z_vals = np.array([[np.percentile(positive_vals_only_flat, p) for p in np.arange(0, 100, 100/(num_events_per_biomarker+1))][1:]]*input_data_z_arr.shape[1])   #Needs updating to work with events per biomarker > 1

    sustain_input = pySuStaIn.ZscoreSustain(
        input_data_z_arr, 
        Z_vals, 
        Z_max, 
        list(original_df.columns), 
        startpoints, 
        N_S_max, 
        N_iterations_MCMC, 
        save_dir, 
        'crossvalidation_output', 
        True
        )


    N_folds = 10
    labels = 1 * np.ones(len(input_data_z_arr), dtype=int) 
    cv = StratifiedKFold(n_splits=N_folds, shuffle=True)
    cv_it = cv.split(input_data_z_arr, labels)
    f = [test for train, test in cv_it]
    test_idxs = np.array([x[:min([len(a) for a in f])] for x in f])

    CVIC, loglike_matrix = sustain_input.cross_validate_sustain_model(test_idxs)

    with open(f'{save_dir}/cross_validation_results.pkl', 'wb') as fp:
        pickle.dump([CVIC, loglike_matrix], fp)