import json
import pandas as pd
import matplotlib.pyplot as plt


home_dir = '\\home_dir\\sustain_connectivity_cohort_sustain_type1_dementia_25_nsp_100000_iter_1_events_filtered_vol_sc_sift2_fbc_cohensD0.3_2_subtypes'

if __name__ == "__main__":

    with open(f'{home_dir}\\results_overall_cvics_1_subtypes.json', 'r') as f:
        cvics1 = json.load(f)


    with open(f'{home_dir}\\results_overall_cvics_2_subtypes.json', 'r') as f:
        cvics2 = json.load(f)


    df = pd.read_csv(f'{home_dir}\\results_sorted_region_names_1_subtypes.csv', index_col=0)


    order_lookup = {}


    for i in df.index:

        k = i.replace('_1_subtypes_subtype_0', '')

        data = []

        for datum in df.loc[i]:
            if 'Atrophy' in datum:
                data.append('Region atrophy')
            else:
                if datum == 'Connection Fibre Bundle Capacity':
                    data.append('Reduced structural connectivity')

        data = ', '.join(data)

        order_lookup[k] = data


    col_lookup = {
        'Region atrophy, Region atrophy, Reduced structural connectivity': 'blue',
        'Region atrophy, Reduced structural connectivity, Region atrophy': 'red',
        'Reduced structural connectivity, Region atrophy, Region atrophy': 'green',
        
    }

    shared_data = set(order_lookup.keys()) | set(cvics1.keys()) | set(cvics2.keys())


    x = [cvics1[k] for k in shared_data]
    y = [cvics2[k] for k in shared_data]
    c = [col_lookup[order_lookup[k]] for k in shared_data]


    plt.scatter(x, y, c=c)
    plt.xlabel('CVIC')


    fig, ax = plt.subplots()

    # Reverse the col_lookup to label_lookup for legend
    label_lookup = {v: k for k, v in col_lookup.items()}

    for color in ['blue', 'red', 'green']: #set(col_lookup.values()):
        keys = [k for k in shared_data if col_lookup[order_lookup[k]] == color]
        x_vals = [cvics1[k] for k in keys]
        y_vals = [cvics2[k] for k in keys]
        label = label_lookup[color]
        ax.scatter(x_vals, y_vals, c=color, label=label, s=10)

    all_x = [cvics1[k] for k in shared_data]
    all_y = [cvics2[k] for k in shared_data]
    min_val = min(min(all_x), min(all_y))
    max_val = max(max(all_x), max(all_y))
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', lw=1)

    ax.set_xlabel('1 subtype model CVIC')
    ax.set_ylabel('2 subtype model CVIC')
    ax.legend(loc='best', fontsize=7)
    plt.savefig(f'{home_dir}\\1 vs 2 Subtype Model CVICs.png')
