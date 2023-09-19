import pandas as pd
import os
import pySuStaIn
import numpy as np
import matplotlib.pyplot as plt
# from math import ceil

input_data_z_table_path =  '/data/home/wpw131/preclindementia152_23_regions_z_scores.csv' 
save_dir = '/data/home/wpw131/ebs_age_age2_TIV_centre_sep_sex_23_regions'  
all_data_path = '/data/home/wpw131/UKBW_EBM_78867r672482_070823.csv' 
dataset_name = 'preclin_AD_152_23_regions_1events_25startpoints_4maxclusters_10000iterations_23biomarkers'
input_dataset_are_controls = False
use_multiple_processors = True
num_events_per_biomarker = 1
set_single_z_to_cases_mean = True
N_startpoints = 25
N_S_max = 4
N_iterations_MCMC = int(1e6)


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
    #Z_vals = np.array([np.arange(0, ceil(max(x)), ceil(max(x))/(num_events_per_biomarker+1))[1:] for x in input_data_z_arr.T])     
    #Z_max = np.array([ceil(max(x)) for x in input_data_z_arr.T]) 

    output_folder = save_dir + f'//{dataset_name}_{num_events_per_biomarker}events_{N_startpoints}startpoints_{N_S_max}maxclusters_{N_iterations_MCMC}iterations_{input_data_z_arr.shape[1]}biomarkers'
    print('Data will be saved in', output_folder)
    # if os.path.exists(output_folder):
    #     shutil.rmtree(output_folder)

    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    sustain_input = pySuStaIn.ZscoreSustain(input_data_z_arr, Z_vals, Z_max, mri_fields, N_startpoints, N_S_max, N_iterations_MCMC, output_folder, dataset_name, use_multiple_processors)

    # run the sustain algorithm with the inputs set in sustain_input above
    s_output = sustain_input.run_sustain_algorithm()
    samples_sequence, samples_f, ml_subtype, prob_ml_subtype, ml_stage, prob_ml_stage, prob_subtype_stage = s_output


    _ = pySuStaIn.ZscoreSustain._plot_sustain_model(sustain_input,samples_sequence,samples_f,input_data_z_arr.shape[0], biomarker_labels=mri_fields, label_font_size=10, label_rot=70, figsize=(20,10))
    plt.savefig(f'{output_folder}//{dataset_name}_positional_variance_diagram.png')
    plt.clf()







# import rpy2 
# from rpy2 import robjects
# from rpy2.robjects.packages import importr
# base = importr("base")
# utils = importr("utils")

# robjects.r('''
#         # create a function `f`
#         f <- function(r, verbose=FALSE) {
#             if (verbose) {
#                 cat("I am calling f().\n")
#             }
#             2 * pi * r
#         }
#         # call the function `f` with argument value 3
#         f(3)
#         ''')


# r = robjects.r
 
# x = robjects.IntVector(range(10))
# y = r.rnorm(10)
 
# r.X11()
 
# r.layout(r.matrix(robjects.IntVector([1,2,3,2]), nrow=2, ncol=2))
# r.plot(r.runif(10), y, xlab="runif", ylab="foo/bar", col="red")




# cmap = get_cmap('Greys')


# figs, axs = plot_positional_var(samples_sequence, samples_f, input_data_z_arr.shape[0], Z_vals, biomarker_labels=None, ml_f_EM=None, cval=False, subtype_order=(0,1), biomarker_order=None, title_font_size=12, stage_font_size=10, stage_label='SuStaIn Stage', stage_rot=0, stage_interval=1, label_font_size=10, label_rot=0, cmap="original", biomarker_colours=None, figsize=None, separate_subtypes=False, save_path=None, save_kwargs={})


# def plot_positional_var(samples_sequence, samples_f, n_samples, Z_vals, biomarker_labels=None, ml_f_EM=None, cval=False, subtype_order=None, biomarker_order=None, title_font_size=12, stage_font_size=10, stage_label='SuStaIn Stage', stage_rot=0, stage_interval=1, label_font_size=10, label_rot=0, cmap="original", biomarker_colours=None, figsize=None, separate_subtypes=False, save_path=None, save_kwargs={}):
#     # Get the number of subtypes
#     N_S = samples_sequence.shape[0]
#     # Get the number of features/biomarkers
#     N_bio = Z_vals.shape[0]
#     # Check that the number of labels given match
#     if biomarker_labels is not None:
#         assert len(biomarker_labels) == N_bio
#     # Set subtype order if not given
#     if subtype_order is None:
#         # Determine order if info given
#         if ml_f_EM is not None:
#             subtype_order = np.argsort(ml_f_EM)[::-1]
#         # Otherwise determine order from samples_f
#         else:
#             subtype_order = np.argsort(np.mean(samples_f, 1))[::-1]
#     elif isinstance(subtype_order, tuple):
#         subtype_order = list(subtype_order)
#     # Unravel the stage zscores from Z_vals
#     stage_zscore = Z_vals.T.flatten()
#     IX_select = np.nonzero(stage_zscore)[0]
#     stage_zscore = stage_zscore[IX_select][None, :]
#     # Get the z-scores and their number
#     zvalues = np.unique(stage_zscore)
#     print(zvalues)
#     N_z = len(zvalues)
#     # Extract which biomarkers have which zscores/stages
#     stage_biomarker_index = np.tile(np.arange(N_bio), (N_z,))
#     stage_biomarker_index = stage_biomarker_index[IX_select]
#     # Warn user of reordering if labels and order given
#     if biomarker_labels is not None and biomarker_order is not None:
#         warnings.warn(
#             "Both labels and an order have been given. The labels will be reordered according to the given order!"
#         )
#     if biomarker_order is not None:
#         # self._plot_biomarker_order is not suited to zscore version
#         # Ignore for compatability, for now
#         # One option is to reshape, sum position, and lowest->highest determines order
#         if len(biomarker_order) > N_bio:
#             biomarker_order = np.arange(N_bio)
#     # Otherwise use default order
#     else:
#         biomarker_order = np.arange(N_bio)
#     # If no labels given, set dummy defaults
#     if biomarker_labels is None:
#         biomarker_labels = [f"Biomarker {i}" for i in range(N_bio)]
#     # Otherwise reorder according to given order (or not if not given)
#     else:
#         biomarker_labels = [biomarker_labels[i] for i in biomarker_order]
    
#     # Z-score colour definition
#     if cmap == "original":
#         # Hard-coded colours: hooray!
#         colour_mat = np.array([[1, 0, 0], [1, 0, 1], [0, 0, 1], [0.5, 0, 1], [0, 1, 1], [0, 1, 0.5]])[:N_z]
#         # We only have up to 5 default colours, so double-check
#         print(N_z, colour_mat.shape[0])
#         if colour_mat.shape[0] < N_z:
#             raise ValueError(f"Colours are only defined for {len(colour_mat)} z-scores!")
#     else:
#         raise NotImplementedError
#     '''
#     Note for future self/others: The use of any arbitrary colourmap is problematic, as when the same stage can have the same biomarker with different z-scores of different certainties, the colours need to mix in a visually informative way and there can be issues with RGB mixing/interpolation, particulary if there are >2 z-scores for the same biomarker at the same stage. It may be possible, but the end result may no longer be useful to look at.
#     '''

#     # Check biomarker label colours
#     # If custom biomarker text colours are given
#     if biomarker_colours is not None:
#         biomarker_colours = pySuStaIn.AbstractSustain.check_biomarker_colours(
#         biomarker_colours, biomarker_labels
#     )
#     # Default case of all-black colours
#     # Unnecessary, but skips a check later
#     else:
#         biomarker_colours = {i:"black" for i in biomarker_labels}

#     # Flag to plot subtypes separately
#     if separate_subtypes:
#         nrows, ncols = 1, 1
#     else:
#         # Determine number of rows and columns (rounded up)
#         if N_S == 1:
#             nrows, ncols = 1, 1
#         elif N_S < 3:
#             nrows, ncols = 1, N_S
#         elif N_S < 7:
#             nrows, ncols = 2, int(np.ceil(N_S / 2))
#         else:
#             nrows, ncols = 3, int(np.ceil(N_S / 3))
#     # Total axes used to loop over
#     total_axes = nrows * ncols
#     # Create list of single figure object if not separated
#     if separate_subtypes:
#         subtype_loops = N_S
#     else:
#         subtype_loops = 1
#     # Container for all figure objects
#     figs = []
#     # Loop over figures (only makes a diff if separate_subtypes=True)
#     for i in range(subtype_loops):
#         # Create the figure and axis for this subtype loop
#         fig, axs = plt.subplots(nrows, ncols, figsize=figsize)
#         figs.append(fig)
#         # Loop over each axis
#         for j in range(total_axes):
#             # Normal functionality (all subtypes on one plot)
#             if not separate_subtypes:
#                 i = j
#             # Handle case of a single array
#             if isinstance(axs, np.ndarray):
#                 ax = axs.flat[i]
#             else:
#                 ax = axs
#             # Check if i is superfluous
#             if i not in range(N_S):
#                 ax.set_axis_off()
#                 continue

#             this_samples_sequence = samples_sequence[subtype_order[i],:,:].T
#             N = this_samples_sequence.shape[1]

#             # Construct confusion matrix (vectorized)
#             # We compare `this_samples_sequence` against each position
#             # Sum each time it was observed at that point in the sequence
#             # And normalize for number of samples/sequences
#             confus_matrix = (this_samples_sequence==np.arange(N)[:, None, None]).sum(1) / this_samples_sequence.shape[0]

#             # Define the confusion matrix to insert the colours
#             # Use 1s to start with all white
#             confus_matrix_c = np.ones((N_bio, N, 3))

#             # Loop over each z-score event
#             for j, z in enumerate(zvalues):
#                 # Determine which colours to alter
#                 # I.e. red (1,0,0) means removing green & blue channels
#                 # according to the certainty of red (representing z-score 1)
#                 alter_level = colour_mat[j] == 0
#                 # Extract the uncertainties for this z-score
#                 confus_matrix_zscore = confus_matrix[(stage_zscore==z)[0]]
#                 # Subtract the certainty for this colour
#                 confus_matrix_c[
#                     np.ix_(
#                         stage_biomarker_index[(stage_zscore==z)[0]], range(N),
#                         alter_level
#                     )
#                 ] -= np.tile(
#                     confus_matrix_zscore.reshape((stage_zscore==z).sum(), N, 1),
#                     (1, 1, alter_level.sum())
#                 )
#             # Add axis title
#             if cval == False:
#                 temp_mean_f = np.mean(samples_f, 1)
#                 # Shuffle vals according to subtype_order
#                 # This defaults to previous method if custom order not given
#                 vals = temp_mean_f[subtype_order]

#                 if n_samples != np.inf:
#                     title_i = f"Subtype {i+1} (f={vals[i]:.2f}, n={np.round(vals[i] * n_samples):n})"
#                 else:
#                     title_i = f"Subtype {i+1} (f={vals[i]:.2f})"
#             else:
#                 title_i = f"Subtype {i+1} cross-validated"
#             # Plot the colourized matrix
#             ax.imshow(
#                 confus_matrix_c[biomarker_order, :, :],
#                 interpolation='nearest'
#             )
#             # Add the xticks and labels
#             stage_ticks = np.arange(0, N, stage_interval)
#             ax.set_xticks(stage_ticks)
#             ax.set_xticklabels(stage_ticks+1, fontsize=stage_font_size, rotation=stage_rot)
#             # Add the yticks and labels
#             ax.set_yticks(np.arange(N_bio))
#             # Add biomarker labels to LHS of every row only
#             if (i % ncols) == 0:
#                 ax.set_yticklabels(biomarker_labels, ha='right', fontsize=label_font_size, rotation=label_rot)
#                 # Set biomarker label colours
#                 for tick_label in ax.get_yticklabels():
#                     tick_label.set_color(biomarker_colours[tick_label.get_text()])
#             else:
#                 ax.set_yticklabels([])
#             # Make the event label slightly bigger than the ticks
#             ax.set_xlabel(stage_label, fontsize=stage_font_size+2)
#             ax.set_title(title_i, fontsize=title_font_size)
#         # Tighten up the figure
#         fig.tight_layout()
#         # Save if a path is given
#         if save_path is not None:
#             # Modify path for specific subtype if specified
#             # Don't modify save_path!
#             if separate_subtypes:
#                 save_name = f"{save_path}_subtype{i}"
#             else:
#                 save_name = f"{save_path}_all-subtypes"
#             # Handle file format, avoids issue with . in filenames
#             if "format" in save_kwargs:
#                 file_format = save_kwargs.pop("format")
#             # Default to png
#             else:
#                 file_format = "png"
#             # Save the figure, with additional kwargs
#             fig.savefig(
#                 f"{save_name}.{file_format}",
#                 **save_kwargs
#             )
#     return figs, axs
