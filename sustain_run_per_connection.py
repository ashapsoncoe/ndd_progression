import pandas as pd
import os
import pySuStaIn
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import StratifiedKFold
import itertools
from scipy import stats
from collections import Counter
import json
from scipy.stats import norm
from scipy.optimize import fsolve


input_data_z_table_path = '/home_dir/all_volumetric_and_structural_connectivity_z_scores_all_dementia_aparc.csv'
input_data_z_table_path_c = '/home_dir/all_volumetric_and_structural_connectivity_z_scores_random_matched_controls_aparc.csv' 
case_control_lookup_path  = '/home_dir/case_ids_to_matched_control_ids.json'
dementia_subtype_lookup_path = '/home_dir/EID_stage_subtype_2_sustain_subtypes.csv' 
save_dir = '/home_dir' 
use_multiple_processors = True
run_cross_validation = True
num_events_per_biomarker = 1
N_startpoints = 25
N_S_max = 2
N_iterations_MCMC = int(1e5)
N_folds = 4
phenotypes_to_include = {'vol': 'Atrophy', 'sc_sift2_fbc': 'Connection Fibre Bundle Capacity'} #'sc_mean_FA': 'Connnection Fractional Anisotropy', 
filter_by_zscore = False
filter_by_cohensD = True
sig_threshold = 0.3
min_phenotype_data_count = 20
plot_histograms_of_separation = True
cohort = 'sustain_type1_dementia' # Options: 'all_dementia', 'post_dx_dementia', 'sustain_type1_dementia', 'sustain_type2_dementia', 'sustain_type1and2_dementia'


# Function to find the intersection of two Gaussian distributions
def find_intersection(mean1, std1, mean2, std2):
    def diff(x):
        return norm.pdf(x, mean1, std1) - norm.pdf(x, mean2, std2)

    return fsolve(diff, (mean1 + mean2) / 2, maxfev=10000)[0]  # Start search at midpoint




if __name__ == '__main__':

    if filter_by_zscore:
        temp = f'zscore{abs(sig_threshold)}'

    if filter_by_cohensD:
        temp = f'cohensD{abs(sig_threshold)}'

    output_folder = f"{save_dir}/sustain_connectivity_cohort_{cohort}_{N_startpoints}_nsp_{N_iterations_MCMC}_iter_{num_events_per_biomarker}_events_filtered_{'_'.join(phenotypes_to_include.keys())}_{temp}"

    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    with open(case_control_lookup_path, 'r') as fp:
        case_control_lookup_dict = json.load(fp)

    input_data_z_df_cases = pd.read_csv(input_data_z_table_path, index_col=0, low_memory=False)
    input_data_z_df_controls = pd.read_csv(input_data_z_table_path_c, index_col=0, low_memory=False)

    # Filter by subtype:
    dementia_subtype_lookup_df = pd.read_csv(dementia_subtype_lookup_path, index_col=1)

    if cohort == 'all_dementia':
        eids_to_use  = set(input_data_z_df_cases.index)

    if cohort == 'sustain_type1_dementia':
        eids_to_use = set(dementia_subtype_lookup_df[(dementia_subtype_lookup_df['subtype'] == 0) & (dementia_subtype_lookup_df['stage'] > 0)].index)

    if cohort == 'sustain_type2_dementia':
        eids_to_use = set(dementia_subtype_lookup_df[(dementia_subtype_lookup_df['subtype'] == 1) & (dementia_subtype_lookup_df['stage'] > 0)].index)

    if cohort == 'sustain_type1and2_dementia':
        eids_to_use = set(dementia_subtype_lookup_df[(dementia_subtype_lookup_df['stage'] > 0)].index)

    input_data_z_df_cases = input_data_z_df_cases.loc[[x for x in eids_to_use if x in input_data_z_df_cases.index]]

    matched_control_indices = [case_control_lookup_dict[str(x)] for x in input_data_z_df_cases.index]

    input_data_z_df_controls = input_data_z_df_controls.loc[matched_control_indices]

    all_regions = set(['_'.join(x.split('_')[:-1]) for x in input_data_z_df_cases.columns if 'vol' in x])

    all_combinations_of_regions = [x for x in itertools.combinations(all_regions, 2) if x[0] != x[1]]


    for region1, region2 in all_combinations_of_regions:
     
        if os.path.exists(f'{output_folder}/pickle_files'):
            for f in os.listdir(f'{output_folder}/pickle_files'):
                os.remove(f'{output_folder}/pickle_files/{f}')
            os.rmdir(f'{output_folder}/pickle_files')
            

        if os.path.exists(f'{output_folder}/sustain_output_{region1}_{region2}_{N_S_max}_subtypes.pkl'): 
            print(f'Data exists for {region1} and {region2}, skipping')
            continue

        else:
            print(f'Data does not exist for {region1} and {region2}')
        
        
        

        biomarkers = []

        for phenotype in phenotypes_to_include:

            if phenotype == 'vol':
                biomarkers.extend([x for x in input_data_z_df_cases.columns if (region1 in x or region2 in x) and phenotype in x])

            else:
                biomarkers.extend([x for x in input_data_z_df_cases.columns if region1 in x and region2 in x and phenotype in x])

        # Get data, remove nan values and and multiply by -1 if decreasing in the cases group:

        input_df_this_combo_cases = input_data_z_df_cases[biomarkers].dropna()
        input_df_this_combo_controls = input_data_z_df_controls[biomarkers].dropna()

        if len(input_df_this_combo_cases) < min_phenotype_data_count or len(input_df_this_combo_controls) < min_phenotype_data_count:
            #print(f'Skipping {region1} and {region2} due to insufficient data')
            continue

        for biomarker in biomarkers:
            if np.mean(input_df_this_combo_cases[biomarker]) < 0:
                input_df_this_combo_cases[biomarker] = -1*input_df_this_combo_cases[biomarker]
                input_df_this_combo_controls[biomarker] = -1*input_df_this_combo_controls[biomarker]

        mean_vals_cases = np.mean(input_df_this_combo_cases, axis=0)
        mean_vals_controls = np.mean(input_df_this_combo_controls, axis=0)

        std_vals_cases = np.std(input_df_this_combo_cases, axis=0)
        std_vals_controls = np.std(input_df_this_combo_controls, axis=0)


        # If not meeting chosen cuttoff for signifance, skip this pair:
        if filter_by_zscore:

            if min(mean_vals_cases) < sig_threshold: continue


        if filter_by_cohensD:

            cohens_Ds = [abs(mean_control - mean_case) / np.sqrt((std_control**2 + std_case**2) / 2) for mean_control, std_control, mean_case, std_case in zip(mean_vals_controls, std_vals_controls, mean_vals_cases, std_vals_cases)]

            if min(cohens_Ds) < sig_threshold: continue


        # Get Z scores from the data by fitting gaussian distirbutions to the cases and controls and finding the intersection point:

        thresholds = []

        for biomarker in biomarkers:
            
            controls = input_df_this_combo_controls[biomarker]
            cases = input_df_this_combo_cases[biomarker]
            
            x = np.linspace(min(min(controls), min(cases)), max(max(controls), max(cases)), 1000)

            pdf_control = norm.pdf(x, mean_vals_controls[biomarker], std_vals_controls[biomarker])
            pdf_case = norm.pdf(x, mean_vals_cases[biomarker], std_vals_cases[biomarker])
            threshold = find_intersection(mean_vals_controls[biomarker], std_vals_controls[biomarker], mean_vals_cases[biomarker], std_vals_cases[biomarker])
            thresholds.append(threshold)


            if plot_histograms_of_separation:

                if not os.path.exists(f'{output_folder}/case_control_histograms'):
                    os.mkdir(f'{output_folder}/case_control_histograms')

                fig, ax = plt.subplots(1, 1, figsize=(15, 5), sharey=True)

                ax.hist(controls, bins=30, density=True, alpha=0.5, color='blue', edgecolor='black', label='Controls')
                ax.hist(cases, bins=30, density=True, alpha=0.5, color='red', edgecolor='black', label='Cases')
                ax.plot(x, pdf_control, 'b-', label='Control Fit')
                ax.plot(x, pdf_case, 'r-', label='Case Fit')
                ax.axvline(threshold, color='black', linestyle='--', label=f'Separator: {threshold:.2f}')
                ax.set_title(biomarker)
                ax.set_xlabel('Value')
                ax.legend()  # Ensure legend appears in each subplot
                fig.suptitle('Maximum Separator using Case-Control Gaussian Fits', fontsize=14)
                ax.set_ylabel('Density')
                plt.tight_layout()
                plt.savefig(f'{output_folder}/case_control_histograms/{biomarker}.png')

        if min(thresholds) < 0: continue

        input_df_this_combo_cases.columns = ['Connection Fibre Bundle Capacity' if 'fbc' in col else col.replace('_', ' ').title().replace('Vol', 'Atrophy') for col in input_df_this_combo_cases.columns]
        input_data_z_arr = np.array(input_df_this_combo_cases) 
        col_names = list(input_df_this_combo_cases.columns)

        Z_vals = np.array([[a] for a in thresholds])
        Z_max = np.array([np.percentile([a for a in input_data_z_arr[:,x] if a>0], 95) for x in range(3)])  

      
        for num_subtypes_to_find in range(1, N_S_max+1):

            sustain_input = pySuStaIn.ZscoreSustain(input_data_z_arr, Z_vals, Z_max, col_names, N_startpoints, num_subtypes_to_find, N_iterations_MCMC, output_folder, '', use_multiple_processors)

            #overall_ml_likelihood = sustain_input._find_ml(sustain_input._AbstractSustain__sustainData)[2]

            s_output = sustain_input.run_sustain_algorithm()

            s_output_dict = {}
            s_output_dict['samples_sequence']  = s_output[0]
            s_output_dict['samples_f']  = s_output[1]
            s_output_dict['ml_subtype']  = s_output[2]
            s_output_dict['prob_ml_subtype']  = s_output[3]
            s_output_dict['ml_stage']  = s_output[4]
            s_output_dict['prob_ml_stage']  = s_output[5]
            s_output_dict['prob_subtype_stage']  = s_output[6]
            s_output_dict['z_vals']  = Z_vals
            s_output_dict['sorted_region_names'] = {}


            # create and save the sustain plot:
            for subtype in range(num_subtypes_to_find):

                this_samples_sequence = s_output_dict['samples_sequence'][subtype,:,:].T
                N = this_samples_sequence.shape[1]
                confus_matrix = (this_samples_sequence==np.arange(N)[:, None, None]).sum(1) / this_samples_sequence.shape[0]
                mean_positions = {bio_n: np.average(range(len(bio_v)), weights=bio_v) for bio_n, bio_v in zip(list(input_df_this_combo_cases.columns), confus_matrix)}
                sorted_region_names = sorted(mean_positions, key=mean_positions.get)
                s_output_dict['sorted_region_names'][f'subtype_{subtype}'] = sorted_region_names
                
                order_to_use = [list(input_df_this_combo_cases.columns).index(a) for a in sorted_region_names]

                labels1 = list(input_df_this_combo_cases.columns)

                if not (len(phenotypes_to_include.keys())*num_events_per_biomarker > 6):

                    figs, axs = pySuStaIn.ZscoreSustain.plot_positional_var(
                        np.array([s_output_dict['samples_sequence'][subtype]]), 
                        np.array([s_output_dict['samples_f'][subtype]]), 
                        s_output_dict['samples_f'].shape[1], 
                        Z_vals, 
                        biomarker_labels=labels1, 
                        ml_f_EM=None, 
                        cval=False, 
                        subtype_order=[0], 
                        biomarker_order=order_to_use, 
                        title_font_size=12, 
                        stage_font_size=20, 
                        stage_label='SuStaIn Stage', 
                        stage_rot=0, 
                        stage_interval=1, 
                        label_font_size=20, 
                        label_rot=0, 
                        cmap="original", 
                        biomarker_colours=None, 
                        figsize=(40,10), 
                        separate_subtypes=True, 
                        save_path= f'{output_folder}/positional_variance_diagram_{region1}_{region2}_{num_subtypes_to_find}_subtypes_subtype_{subtype}', 
                        save_kwargs={},

                        )
                

            with open(f'{output_folder}/sustain_output_{region1}_{region2}_{num_subtypes_to_find}_subtypes.pkl', 'wb') as fp:
                pickle.dump(s_output_dict, fp)


            # run cross validation
            if run_cross_validation:
                labels2 = 1 * np.ones(len(input_data_z_arr), dtype=int) 
                cv = StratifiedKFold(n_splits=N_folds, shuffle=True)
                cv_it = cv.split(input_data_z_arr, labels2)
                f = [test for train, test in cv_it]
                test_idxs = np.array([x[:min([len(a) for a in f])] for x in f])

                CVIC, loglike_matrix = sustain_input.cross_validate_sustain_model(test_idxs)

                with open(f'{output_folder}/cross_validation_results_{N_folds}_fold_{region1}_{region2}_{num_subtypes_to_find}_subtypes.pkl', 'wb') as fp:
                    pickle.dump([CVIC, loglike_matrix], fp)
                
    

    
 
    # Generate summary stats for completed sustain runs
    for num_subtypes_to_find in range(1, N_S_max+1):

        sorted_region_names = {}
        overall_ml_likelihoods = {}
        overall_cvics = {}
        all_z_vals = {}

        for f in os.listdir(output_folder):

            if 'sustain_output' in f and f'{num_subtypes_to_find}_subtypes' in f:

                with open(f'{output_folder}/{f}', 'rb') as fp:
                    c = pickle.load(fp)

                region_combo = f[15:-4]

                sorted_region_names[region_combo] = c['sorted_region_names']
                all_z_vals[region_combo] = c['z_vals']
                
                
            if 'cross_validation_results' in f and f'{num_subtypes_to_find}_subtypes' in f:

                with open(f'{output_folder}/{f}', 'rb') as fp:
                    c = pickle.load(fp)

                region_combo = f[32:-15]

                overall_cvics[region_combo] = float(c[0][num_subtypes_to_find-1])


        with open(f'{output_folder}/results_overall_ml_likelihoods_{num_subtypes_to_find}_subtypes.json', 'w') as fp:
            json.dump(overall_ml_likelihoods, fp)
            
        with open(f'{output_folder}/results_overall_cvics_{num_subtypes_to_find}_subtypes.json', 'w') as fp:
            json.dump(overall_cvics, fp)

        generic_labels = []

        for phenotype in phenotypes_to_include:
            if phenotype == 'vol':
                generic_labels.extend(['Region1 atrophy', 'Region2 atrophy'])
            else:
                generic_labels.append(phenotypes_to_include[phenotype])

        cols_for_z_df = [a for b in [[f'{l} Z{a+1}' for a in range(num_events_per_biomarker)] for l in generic_labels] for a in b]

        for combo in all_z_vals:
            all_z_vals[combo] = {a: b[0] for a,b in zip(cols_for_z_df, all_z_vals[combo])}


        all_zvals_df = pd.DataFrame(all_z_vals).transpose()

        all_zvals_df.to_csv(f'{output_folder}/results_all_z_vals.csv') # Same regardless of subtypes 


        subtype_names = list(sorted_region_names.values())[0].keys()
        sorted_orders_orig_names = {}
        sorted_orders_new_names = {}


        for subtype in subtype_names:
            sorted_orders_new_names.update({f'{x}_{subtype}': ['Region atrophy'  if 'atrophy' in a.lower() else a for a in sorted_region_names[x][subtype]] for x in sorted_region_names})
            sorted_orders_orig_names.update({f'{x}_{subtype}': sorted_region_names[x][subtype] for x in sorted_region_names})


        sorted_region_names_df = pd.DataFrame(sorted_orders_orig_names).transpose()
        sorted_region_names_df.columns = [f'Event {x+1}' for x in range(len(sorted_region_names_df.columns))]
        sorted_region_names_df.to_csv(f'{output_folder}/results_sorted_region_names_{num_subtypes_to_find}_subtypes.csv')

        # key_dict = sorted_orders_new_names
        # all_positions = [(a, [key_dict[x].index(a)+1 for x in key_dict]) for a in list(key_dict.values())[0]]
        # all_positions_df = pd.DataFrame(all_positions)
        # all_positions_df.index = key_dict.keys()
        # all_positions_df.to_csv(f'{output_folder}/results_all_positions_{num_subtypes_to_find}_subtypes.csv')


        counts = Counter([','.join([x for x in a]) for a in sorted_orders_new_names.values()])

        with open(f'{output_folder}/results_counts_of_order_combinations_{num_subtypes_to_find}_subtypes.json', 'w') as fp:
            json.dump(counts, fp)

        # friedman_chi_sq_stat, friedman_chi_sq_p_value = stats.friedmanchisquare(*all_positions.values())
        chi_sq_stat, chi_sq_p_value = stats.chisquare(list(counts.values()))

        summary_stats ={
            'all_orders_chi_sq_stat': chi_sq_stat,
            'all_orders_chi_sq_p_value': chi_sq_p_value,
            # 'all_positions_friedman_chi_sq_stat': friedman_chi_sq_stat,
            # 'all_positions_friedman_chi_sq_p_value': friedman_chi_sq_p_value,
            'mean_ml_likelihood': float(np.mean(list(overall_ml_likelihoods.values()))),
        }

        with open(f'{output_folder}/results_summary_stats_{num_subtypes_to_find}_subtypes.json', 'w') as fp:
            json.dump(summary_stats, fp)



