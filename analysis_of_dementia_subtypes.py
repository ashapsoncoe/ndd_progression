import os
import sys

working_dir = os.path.dirname(__file__)
sys.path.insert(0, working_dir)
os.chdir(working_dir)


import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pySuStaIn
import scipy
import statsmodels.stats.proportion as proportion
import numpy as np
import pylab
from collections import Counter
from scipy.stats import chi2_contingency

original_data_path = 'GLM_gaussian_max_deg_2_age_TIV_centre_grouped_regions_onlyage2_sep_sex\\preclindementia152_23_regions_z_scores.csv'
results_folder = 'preclin_AD_152_23_regions_1events_25startpoints_4maxclusters_1000000iterations_23biomarkers'
raw_data_file = 'UKBW_EBM_78867r674582_250923v2.csv' 
N_S_max  = 4
N_biomarkers = 23
Z_max = 2.40284599
Z_vals = [0.40208586]



def get_stats(dtype, raw_df, assigned_types, assigned_stages, num_types_to_use, exclude_stage_0=True, stage_0_only=False):

    type_frequencies = {x: [] for x in range(num_types_to_use)}

    for subtype, freq, stage in zip(assigned_types, raw_df[dtype], assigned_stages):

        if np.isnan(freq): 
            continue

        if stage == 0:
            if exclude_stage_0:
                continue
            else:
                type_frequencies[subtype].append(freq)
        else:
            if not stage_0_only:
                type_frequencies[subtype].append(freq)
        

    t, pval = scipy.stats.ttest_ind(type_frequencies[0], type_frequencies[1], equal_var=True)

    print(f'Subtype 1 mean {np.mean(type_frequencies[0])}, std {np.std(type_frequencies[0])}, n = {len(type_frequencies[0])}')

    print(f'Subtype 2 mean {np.mean(type_frequencies[1])}, std {np.std(type_frequencies[1])}, n = {len(type_frequencies[1])}')

    print(f"Welch's t-test: P value = {pval}, t = {t}")

    return t, pval, type_frequencies

def make_rainfall_plot(v_data, dtype):

    # From https://medium.com/mlearning-ai/getting-started-with-raincloud-plots-in-python-2ea5c2d01c11

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.set_position([0.5, 0.1, 0.2, 0.8])
    # Create a list of colors for the boxplots based on the number of features you have
    boxplots_colors = ['yellowgreen', 'olivedrab']

    # Boxplot data
    bp = ax.boxplot(v_data, patch_artist = True, vert = True, showfliers=False, whis=10**100)

    # Change to the desired color and add transparency
    for patch, color in zip(bp['boxes'], boxplots_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.4)

    # Create a list of colors for the violin plots based on the number of features you have
    violin_colors = ['thistle', 'orchid']

    # Violinplot data
    vp = ax.violinplot(v_data, points=500, 
                showmeans=False, showextrema=False, showmedians=False, vert=True)

    for idx, b in enumerate(vp['bodies']):
        # Get the center of the plot
        m = np.mean(b.get_paths()[0].vertices[:, 0])
        b.get_paths()[0].vertices[:, 0] = np.clip(b.get_paths()[0].vertices[:, 0], m, np.inf)
        # Change to the desired color
        b.set_color(violin_colors[idx])
   

    # Create a list of colors for the scatter plots based on the number of features you have
    scatter_colors = ['tomato', 'darksalmon']

    # Scatterplot data
    for idx, features in enumerate(v_data):
        # Add jitter effect so the features do not overlap on the y-axis
        y = np.full(len(features), idx + .8)
        idxs = np.arange(len(y))
        out = y.astype(float)
        out.flat[idxs] += np.random.uniform(low=-.05, high=.05, size=len(idxs))
        y = out
        plt.scatter(y, features, s=.3, c=scatter_colors[idx])

    plt.xticks(np.arange(1,3,1), ['Subtype 1', 'Subtype 2'])  # Set text labels.
    plt.ylabel(dtype)
    plt.show()


if __name__ == '__main__':


    original_df = pd.read_csv(f'{working_dir}\\{original_data_path}', index_col=0)

    Z_vals = np.array([Z_vals for i in range(N_biomarkers)])

    data_by_subtype = {}

    pickle_path = f'{working_dir}\\{results_folder}\\pickle_files'

    for f in os.listdir(pickle_path):

        subtype_n = int(f.split('subtype')[-1].split('.')[0])

        with open(f'{pickle_path}//{f}', 'rb') as fp:
            data_by_subtype[subtype_n] = pickle.load(fp)


    _ = plt.figure()

    # Plot all models:
    for i in data_by_subtype:

        for n in range(i+1):


            this_samples_sequence = data_by_subtype[i]['samples_sequence'][n,:,:].T
            N = this_samples_sequence.shape[1]
            confus_matrix = (this_samples_sequence==np.arange(N)[:, None, None]).sum(1) / this_samples_sequence.shape[0]
            mean_positions = {bio_n: np.average(range(len(bio_v)), weights=bio_v) for bio_n, bio_v in zip(list(original_df.columns), confus_matrix)}
            sorted_region_names = sorted(mean_positions, key=mean_positions.get)
            order_to_use = [list(original_df.columns).index(a) for a in sorted_region_names]


            figs, axs = pySuStaIn.ZscoreSustain.plot_positional_var(
                data_by_subtype[i]['samples_sequence'][n:n+1], 
                data_by_subtype[i]['samples_f'][n:n+1], 
                data_by_subtype[i]['samples_f'][n:n+1].shape[1], 
                Z_vals, 
                biomarker_labels=[x.replace('_', ' ').capitalize() for x in original_df.columns], 
                ml_f_EM=None, 
                cval=False, 
                subtype_order=None, 
                biomarker_order=order_to_use, 
                title_font_size=0, 
                stage_font_size=14, 
                stage_label='SuStaIn Stage', 
                stage_rot=0, 
                stage_interval=1, 
                label_font_size=14, 
                label_rot=0, 
                cmap="original", 
                biomarker_colours=None, 
                figsize=(10,10), 
                separate_subtypes=True, 
                save_path= f'{working_dir}\\{results_folder}\\positional_variance_diagram_{i+1}_subtypes_subtype_{n+1}', 
                save_kwargs={'dpi': 400}
                )

    
    plt.clf()

    N_iterations_MCMC = data_by_subtype[0]['samples_sequence'][0].shape[1]

    for s in data_by_subtype:

        loaded_variables            = data_by_subtype[s]
        samples_likelihood          = loaded_variables["samples_likelihood"]

        _ = plt.figure(0)
        _ = plt.plot(range(N_iterations_MCMC), samples_likelihood, label=f"{s+1} subtype model")
        _ = plt.figure(1)
        _ = plt.hist(samples_likelihood, label=f"{s+1} subtype model")
        
    _ = plt.figure(0)
    _ = plt.legend(loc='upper right')
    _ = plt.xlabel('MCMC samples')
    _ = plt.ylabel('Log likelihood')
    _ = plt.title('MCMC trace')
        
    _ = plt.figure(1)
    _ = plt.legend(loc='upper right')
    _ = plt.xlabel('Log likelihood')  
    _ = plt.ylabel('Number of samples')  
    _ = plt.title('Histograms of model likelihood')
    plt.show()


    with open(f'{working_dir}\\{results_folder}\\cross_validation_results_10_fold.pkl', 'rb') as fp:
        CVIC, loglike_matrix  = pickle.load(fp)



    # go through each subtypes model and plot the log-likelihood on the test set and the CVIC
    print("CVIC for each subtype model: " + str(CVIC))
    print("Average test set log-likelihood for each subtype model: " + str(np.mean(loglike_matrix, 0)))

    _ = plt.figure(1)    
    _ = plt.plot(range(1, N_S_max+1),CVIC)
    _ = plt.xticks(range(1, N_S_max+1))
    _ = plt.ylabel('CVIC')  
    _ = plt.xlabel('Subtypes model') 
    _ = plt.title('CVIC')

    _ = plt.figure(0)
    df_loglike = pd.DataFrame(data = loglike_matrix, columns = [str(i+1) for i in range(N_S_max)])
    df_loglike.boxplot(grid=False)
    for i in range(N_S_max):
        y = df_loglike[[str(i+1)]]
        x = np.random.normal(1+i, 0.04, size=len(y)) # Add some random "jitter" to the x-axis
        pylab.plot(x, y, 'r.', alpha=0.2)
    _ = plt.ylabel('Log likelihood')  
    _ = plt.xlabel('Subtypes model') 
    _ = plt.title('Test set log-likelihood across folds')
    plt.show()


    # Histogram of stage values by subtype

    alphabetical_regions = list(original_df.columns)
    alphabetical_regions.sort()

    for model_num in (0,1):

        assigned_types = [int(x[0]) for x in data_by_subtype[model_num]['ml_subtype']]
        assigned_stages = [int(x[0]) for x in data_by_subtype[model_num]['ml_stage']]

        for pattern_type in set(assigned_types):

            stages_of_this_type = [stage for stage, dtype in zip(assigned_stages, assigned_types) if dtype == pattern_type]

            c = Counter(stages_of_this_type)

            x_axis = range(len(alphabetical_regions)+1)
            y_axis = [c[a] if a in c else 0 for a in x_axis]

            _ = plt.bar(x_axis, y_axis)
            _ = plt.xticks(x_axis)
            _ = plt.ylabel('Frequency')  
            _ = plt.xlabel('Stage') 
            _ = plt.title(f'Distribution of cases of subtype {pattern_type+1} in {model_num+1}-subtype model')

            plt.show()


    # Studying epidemiological associations

    columns_to_use = [
        'total_volume_of_periventricular_white_matter_hyperintensities_f24485_2_0',                                                                                                                                                             
        'total_volume_of_deep_white_matter_hyperintensities_f24486_2_0',
        'standard_prs_for_alzheimers_disease_ad_f26206_0_0',
        'APOE4_alleles',
        'R_BMI_f21001_2',
        'EID',
        'R_SmokeEver_f20160_2',
        'R_RF_DemLC_4_Hypertension_ever', # yes and no
        'R_RF_DemLC_7_Diabetes_ever', # yes and no
        'ML_C42C240Xf41270_VascularDementia', # 1 and 0
        'ML_C42C240Xf41270_Alzheimers',
    ]

    raw_df = pd.read_csv(raw_data_file, usecols=columns_to_use)
    raw_df.index = raw_df['EID']
    raw_df_controls = raw_df.dropna(subset=['APOE4_alleles']).query('ML_C42C240Xf41270_VascularDementia != 1 and ML_C42C240Xf41270_Alzheimers != 1')
    raw_df = raw_df.loc[original_df.index, :]



    # Analyse ApoE4 alleles:
    print('Mean control ApoE4 alleles: ')
    apoe4_type_values = get_stats('APOE4_alleles', raw_df, assigned_types, assigned_stages, 2, exclude_stage_0=True, stage_0_only=False)[2]


    type1_values = [int(x) for x in apoe4_type_values[0]]
    type2_values = [int(x) for x in apoe4_type_values[1]]
    control_values = [int(x) for x in raw_df_controls.sample(n=len(type1_values)+len(type2_values), random_state=1)['APOE4_alleles']]


    scipy.stats.ttest_ind(type1_values, control_values)
    scipy.stats.ttest_ind(type2_values, control_values)


    type1_counts = Counter(type1_values)
    type2_counts = Counter(type2_values)
    control_counts = Counter(control_values)

    t1_counts = [type1_counts[0], type1_counts[1], type1_counts[2]]
    t2_counts = [type2_counts[0], type2_counts[1], type2_counts[2]]
    c_counts = [control_counts[0], control_counts[1], control_counts[2]]

    print('For type 1 vs type 2:') 
    chi2_contingency([t1_counts, t2_counts])

    print('For type 1 vs controls:') 
    chi2_contingency([t1_counts, c_counts])

    print('For type 2 vs controls:') 
    chi2_contingency([t2_counts, c_counts])


    # Plot continuous types:
    continuous_dtypes = {
        #'total_volume_of_white_matter_hyperintensities_from_t1_and_t2_flair_images_f25781_2_0': 'Total WMHI volume',
        'total_volume_of_periventricular_white_matter_hyperintensities_f24485_2_0': 'pWMHI volume (mm³)',                                                                                                                                                               
        'total_volume_of_deep_white_matter_hyperintensities_f24486_2_0':  'dWMHI volume (mm³)',
        'standard_prs_for_alzheimers_disease_ad_f26206_0_0': 'PRS-AD',
        'R_BMI_f21001_2': 'BMI (kg/m²)',

    }

    assigned_types = [int(x[0]) for x in data_by_subtype[1]['ml_subtype']]
    assigned_stages = [int(x[0]) for x in data_by_subtype[1]['ml_stage']]

    for dtype in continuous_dtypes:

        print(f'For {dtype}')
        t, pval, type_values = get_stats(dtype, raw_df, assigned_types, assigned_stages, 2, exclude_stage_0=True, stage_0_only=False)
        v_data = [type_values[0], type_values[1]]

        dtype_label = continuous_dtypes[dtype]

        make_rainfall_plot(v_data, dtype_label)


    binary_dtypes = [
        'R_SmokeEver_f20160_2',
        'R_RF_DemLC_4_Hypertension_ever', # yes and no
        'R_RF_DemLC_7_Diabetes_ever', # yes and no
        'ML_C42C240Xf41270_VascularDementia', # 1 and 0
        'ML_C42C240Xf41270_Alzheimers', # 1 and 0
    ]


    for dtype in binary_dtypes:

        raw_data = list(raw_df[dtype])

        type0_pos = len([x for x in zip(raw_data, assigned_types, assigned_stages) if x[2]!=0 and x[1]==0 and (x[0]=='Yes' or x[0]==1)])
        type0_neg = len([x for x in zip(raw_data, assigned_types, assigned_stages) if x[2]!=0 and x[1]==0 and (x[0]=='No' or x[0]==0)])
        type1_pos = len([x for x in zip(raw_data, assigned_types, assigned_stages) if x[2]!=0 and x[1]==1 and (x[0]=='Yes' or x[0]==1)])
        type1_neg = len([x for x in zip(raw_data, assigned_types, assigned_stages) if x[2]!=0 and x[1]==1 and (x[0]=='No' or x[0]==0)])

        positive = np.array([type0_pos, type1_pos])
        totals = np.array([type0_pos+type0_neg, type1_pos+type1_neg])

        chisq, pvalue, table = proportion.proportions_chisquare(positive, totals)
        print(f'For {dtype}')
        print(f'Subtype 1 prop {positive[0]/totals[0]}, n = {totals[0]}, positive cases = {positive[0]}')
        print(f'Subtype 2 prop {positive[1]/totals[1]}, n = {totals[1]}, positive cases = {positive[1]}')
        print(f"Chisq proportions test: P value = {pvalue}, chisq = {chisq}")
        print('')



