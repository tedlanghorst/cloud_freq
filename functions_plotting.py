import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import textwrap 
from matplotlib.colors import TwoSlopeNorm, Normalize, LogNorm

#Plotting functions
def cloud_bar_plot(df_in, ax):
    df_grouped = df_in.groupby('quantile_bin')['cloud_class'].value_counts(normalize=True).unstack('cloud_class')
    df_grouped = df_grouped[(df_grouped.index > 0) & (df_grouped.index < 11)]
    
    #Rename columns
    cloud_class_names = ['Clear', 'Mixed', 'Cloudy']
    df_grouped.columns = cloud_class_names
    
    df_grouped.plot.bar(stacked=True, width=0.95, ax=ax)

    # Customize ticks
    ax.set_ylim([0,1])
    # new_labels = [f'{(label-1)*10:.0f}-{label*10:.0f}' for label in df_grouped.index]
    ax.set_xticks(np.linspace(1,9,5))
    new_labels = [f'{(label+1) * 0.1:.1f}' for label in ax.get_xticks()]
    
    ax.set_xticklabels(new_labels, rotation=0)
    
    # Set axis labels
    ax.set_ylabel('Proportion')
    ax.set_xlabel('Discharge Quantile')
    ax.legend(title='Cloud Class', loc='upper left', bbox_to_anchor=(1, 1.045))
    
    # Create a twin Axes to duplicate y labels
    # ax2 = ax.twinx()
    # ax2.set_yticks(ax.get_yticks())
    # ax2.set_ylim(ax.get_ylim())
    
def split_cloud_plot(sites, df, split_col, bins, colors, ax):
    ax.set_prop_cycle('color', colors)

    # Plot each site ID's time series
    for lower_limit, upper_limit in zip(bins, bins[1:]):
        # sites_size_bin = sites[(sites['lta_discharge']>=lower_limit) & (sites['lta_discharge']<upper_limit)]
        sites_size_bin = sites[(sites[split_col]>=lower_limit) & (sites[split_col]<upper_limit)]
        df_size_bin = df[df.index.get_level_values('id').isin(sites_size_bin.index)]

        df_grouped = df_size_bin.groupby('quantile_bin')['cloud_class'].value_counts(normalize=True).unstack('cloud_class')
        df_grouped = df_grouped[(df_grouped.index > 0) & (df_grouped.index < 11)]

        low_string = f'$10^{np.log10(lower_limit):1.0f}$'
        high_string = f'$10^{np.log10(upper_limit):1.0f}$'

        # label = f"{low_string}-{high_string}, n={len(sites_size_bin)}"
        label = f"{low_string}-{high_string}"
        
        ax.plot(df_grouped.index,df_grouped['No Cloud'],label=label)

    ax.set_xticks(np.linspace(2,10,5))
    new_labels = [f'{label * 0.1:.1f}' for label in ax.get_xticks()]
    ax.set_xticklabels(new_labels, rotation=0)

    ax.set_yticks(np.linspace(0.30,0.5,5))

    # Create a twin Axes to duplicate y labels
    # ax2 = ax.twinx()
    # ax2.set_yticks(ax.get_yticks())
    # ax2.set_ylim(ax.get_ylim())

    ax.set_xlabel('Discharge Quantile')
    ax.set_ylabel('Proportion of cloud-free images')
    
def obs_true_1to1(df_in, quant, ax):
    grouped = df_in .groupby(['id','cloud_binary'])
    mean_values = grouped['Q'].quantile(quant/100.0)
    cloud_values = (mean_values[mean_values.index.get_level_values('cloud_binary') == False]
                    .reset_index(level='cloud_binary', drop=True)
                    .rename(f"cloud_Q"))
    
    grouped = df_in.groupby(['id'])
    no_cloud_values = grouped['Q'].quantile(quant/100.0).rename(f"all_Q")
    
    valid_data = (cloud_values != 0) & (no_cloud_values != 0)
    cloud_values = cloud_values[valid_data]
    no_cloud_values = no_cloud_values[valid_data]

    quant_plot = pd.merge(cloud_values,no_cloud_values,left_index=True,right_index=True,how='left')
    quant_plot['ND'] = (quant_plot["cloud_Q"]-quant_plot["all_Q"])/quant_plot["all_Q"]
    
    norm = TwoSlopeNorm(vmin=-0.5, vcenter=0, vmax=0.5)
    cmap = 'RdBu'
    ax.scatter(quant_plot["all_Q"],quant_plot["cloud_Q"],c=quant_plot['ND'],s=0.25,cmap=cmap,norm=norm)
    
    # Set the same limits for both axes
    min_val = min(quant_plot[["all_Q","cloud_Q"]].min())
    max_val = max(quant_plot[["all_Q","cloud_Q"]].max())
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    ax.set_aspect('equal', 'box')
    
    # Add a 1-to-1 line
    ax.plot(ax.get_xlim(), ax.get_ylim(), ls="--", c=".3")
    
    # Log-transform the axes
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('')
    ax.set_ylabel('')
    
    return
    

def pfaf_level2_plot(df_in, col_plot, cmap, norm, ax):
    df_in.plot(col_plot, edgecolor='black', linewidth=0.2, ax=ax, cmap=cmap, norm=norm)
    ax.set_xlim([-1.4E7, 1.6E7])
    ax.set_ylim([-6E6, 1E7])
    ax.set_xticks([])
    ax.set_yticks([])
    
def global_reach_plot(df_in, col_plot, cmap, norm, ax):
    df_in.plot(col_plot, ax=ax, cmap=cmap, norm=norm)
    ax.set_xlim([-1.4E7, 1.6E7])
    ax.set_ylim([-6E6, 1E7])
    ax.set_xticks([])
    ax.set_yticks([])
    
def monthly_cloud_Q_plot(df_in, ax):
    mask = df_in['Q_norm']['count'] > (df_in['Q_norm']['count'].max() * 0.1)
    df_in = df_in[mask]
    
    # Plot the mean values
    ax.plot(df_in.index,df_in['cloudMask']['mean'], color='black')
    ax.plot(df_in.index,df_in['Q_norm']['mean'], color='blue')
    
    # Add shaded area for the uncertainty between p10 and p90
    ax.fill_between(df_in.index, 
                df_in['cloudMask']['p10'], 
                df_in['cloudMask']['p90'], 
                color='black', alpha=0.1, edgecolor='None')
    ax.fill_between(df_in.index, 
                df_in['Q_norm']['p10'], 
                df_in['Q_norm']['p90'], 
                color='blue', alpha=0.1, edgecolor='None')
    
    ax.set_ylim([0,1])
    ax.set_xlim([1,12])
    ax.set_xticks([])
    ax.set_yticks([])
    
    