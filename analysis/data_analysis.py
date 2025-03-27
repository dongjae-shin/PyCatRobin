import matplotlib.pyplot as plt
import mplcursors
import pandas as pd
import seaborn as sns
import numpy as np
import warnings

from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.pyplot import legend
from matplotlib.widgets import Button
from openpyxl.styles.builtins import styles

from data.extract import DataForGP

class DataAnalysis:
    def __init__(self, dataset: DataForGP = None):
        if dataset is None:
            raise ValueError("Please provide a dataset.")

        self.dataset = dataset
        self.unique_properties = None
        self.df_snr = None

    def compare_targets_std_dev(
            self, target_wise: bool = False, colormap: str = 'tab10', plot_module_hist: str = 'seaborn'
    ):
        """
        Compare the standard deviation of the target values for each column.

        Args:
            target_wise (bool): If True, compare standard deviations target-wise; otherwise, compare overall.
            colormap (str): The colormap to use for the plot.
            plot_module_hist (str): The module to use for plotting histograms. Either 'seaborn' or 'plotly'.

        Returns:
            None
        """
        if target_wise:
            columns = [col for col in self.dataset.df_stat.columns if '_std' in col]

            # extract the properties from the columns
            properties = [] # e.g., 'CO2 Conversion (%)', 'Selectivity to CO (%)'
            for col in columns:
                parts = col.rsplit('_', 2)
                if len(parts) > 2:
                    properties.append(parts[0])
            self.unique_properties = list(set(properties))

            # Plot the data
            for property in self.unique_properties: # Figures over properties
                columns_property = [col for col in columns if property in col]# Figures over properties
                # Create subplots
                nrows = (len(columns_property) + 2) // 3
                ncols = min(len(columns_property), 3)
                fig, axs = plt.subplots(nrows, ncols, figsize=(13, 8))

                # set title of the fig
                fig.suptitle(f'{property}', fontsize=16)

                axs = np.ravel(axs)
                n_plot = 0
                for i, column in enumerate(columns_property): # Subplots over methods
                    # Melt the DataFrame to long format for seaborn
                    df_melted = self.dataset.df_stat.melt(id_vars=['GroupID'],
                                                          value_vars=column,
                                                          var_name='Target',
                                                          value_name='Standard Deviation')
                    # Generate colors from a predefined colormap
                    cmap = plt.cm.get_cmap(colormap, len(df_melted['GroupID'].unique()))
                    colors = [cmap(i) for i in range(len(df_melted['GroupID'].unique()))]
                    colors[-1] = 'black'
                    # Plot the data
                    sns.barplot(data=df_melted, x='GroupID', y='Standard Deviation', ax=axs[i], palette=colors,
                                hue='GroupID', legend=False)
                    # Set the title of the subplot with the signal-to-noise ratio (SNR)
                    snr = df_melted.iloc[-1]['Standard Deviation'] / df_melted.iloc[:-1]['Standard Deviation'].max()
                    axs[i].set_title(f'{column.split("_")[1]} (SNR={snr:.2f})', fontsize=12)

                    # Show ylabel only for the leftmost axes, and  xlabel only for the lowest axes
                    if i % ncols != 0:
                        axs[i].set_ylabel('')
                    if i < (nrows - 1) * ncols:
                        axs[i].set_xlabel('')
                    n_plot += 1

                    # Make the axes itself a button
                    def on_click(event, col=column, ax=axs[i]):
                        if event.inaxes == ax:
                            self._generate_data_distribution(column=col.rstrip("_std"), cmap=colors,
                                                             plot_module=plot_module_hist)
                    fig.canvas.mpl_connect('button_press_event', on_click)

                # Hide the blank Axes: turn off Axes.axis if Axes order > number of plotted Axes
                for i in range(len(axs)):
                    if i + 1 > n_plot:
                        axs[i].axis('off')

                # Show a set of subplots for every property
                fig.tight_layout()
                plt.show()
        else:
            # Melt the DataFrame to long format for seaborn
            df_melted = self.dataset.df_stat.melt(id_vars=['GroupID'],
                                             value_vars=[col for col in self.dataset.df_stat.columns if '_std' in col],
                                             var_name='Target',
                                             value_name='Standard Deviation')
            df_melted['Target'] = df_melted['Target'].str.rstrip('_std')

            # Plot the data
            plt.figure(figsize=(12, 6))
            sns.barplot(data=df_melted, x='Target', y='Standard Deviation', hue='GroupID', palette=colormap)
            plt.xticks(rotation=45, ha='right')
            plt.title('Standard Deviation of Target Values by GroupID')
            plt.tight_layout()
            plt.show()

    def _generate_data_distribution(self, column: str, cmap: str = 'tab10', plot_module: str = 'seaborn'):

        # shallow copy: connected to the original df. it's like using nickname
        df_stat = self.dataset.df_stat.copy().reset_index(drop=True)
        i=1
        # Initialize the DataFrame with the first group's data
        df = pd.DataFrame(
            {'filename'       : df_stat[df_stat['GroupID'] == i]['filename'][i-1],
             'experiment_date': df_stat[df_stat['GroupID'] == i]['experiment_date'][i-1],
             'GroupID'        : [i] * len(df_stat[df_stat['GroupID'] == i][f'{column}_list'][i-1]),
             'location'       : df_stat[df_stat['GroupID'] == i]['location'][i-1],
             f'{column}'      : df_stat[df_stat['GroupID'] == i][f'{column}_list'][i-1]}
        ) # 'GroupID' -> len(): to extract number of groups + total
        # Concatenate other groups' data to the DataFrame in axis=0
        for i, group in enumerate(df_stat['GroupID'].unique()[1:]): # slicing: to exclude the group 'total'
            df = pd.concat((
                df,
                pd.DataFrame(
                {'filename'       : df_stat[df_stat['GroupID'] == group]['filename'][i+1],
                 'experiment_date': df_stat[df_stat['GroupID'] == group]['experiment_date'][i+1],
                 'GroupID'        : [group] * len(df_stat[df_stat['GroupID'] == group][f'{column}_list'][i+1]),
                 'location'       : df_stat[df_stat['GroupID'] == group]['location'][i+1],
                 f'{column}'      : df_stat[df_stat['GroupID'] == group][f'{column}_list'][i+1]}
                )
            ), axis=0)
        df.reset_index(drop=True, inplace=True)
        # set values of column, of which GroupID is 'total', to 'all'
        df.loc[df['GroupID'] == 'total', 'location'] = 'all'

        if plot_module == 'seaborn':
            # Plot a violinplot
            fig, axs = plt.subplots(nrows=2, ncols=1, sharex=True)

            hue_order = df['GroupID'].unique()
            sns.violinplot(
                df, x=column,  hue='GroupID', split=False, inner='stick', palette=cmap, ax=axs[0],
                legend=False, hue_order=hue_order
            )

            # Overlay a stripplot to the violinplot to differentiate the 'location'
            markers = ['X', 'o', 'P', '^', '*', 'v', 'D', 'P',]
            colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
            legend_elements = []
            locmarks = [[df['location'].unique()[i], markers[i], colors[i]] for i in range(len(df['location'].unique()))]
            # Manually implement the legend for the location since seaborn does not support it
            for locmark in locmarks:
                # Select the data for the location
                df_selected = df[df['location'] == locmark[0]].reset_index(drop=True)
                # Add dummy rows so df_selected has always all GroupID values -> use the same hue range with the violin plot
                for i, group_id in enumerate(hue_order):
                    with warnings.catch_warnings(action="ignore"):
                        df_selected.loc[len(df_selected)+i+1] = [None, None, group_id, 'dummy_location', None]#df_selected.iloc[0] # add a row to avoid the error of the last row
                # Plot the strip plot for the location with the corresponding marker
                strip = sns.stripplot(
                    df_selected, x=column,  hue='GroupID', jitter=True, dodge=True,
                    palette=[locmark[2]] * len(hue_order),
                    ax=axs[0], legend=False, size=5, marker=locmark[1],
                    linewidth=0.5, edgecolor='w', hue_order=hue_order,
                )
                # Add the legend element that is not tied to the plot but corresponds to locmark
                legend_elements.append(
                    Line2D(
                        [0], [0], label=locmark[0], marker=locmark[1], color='w',
                        markersize=9, markeredgecolor='w', markerfacecolor=locmark[2]
                    )
                )
            axs[0].legend(handles=legend_elements, title='Location')
            axs[0].set_yticks([])

            # Plot the histogram
            sns.histplot(
                df, x=column, hue='GroupID', shrink=0.95, multiple='stack', stat='count', palette=cmap,
                kde=False, ax=axs[1]
            )
            # Make the boundary of axs[1] the same as that of axs[0]
            axs[1].set_xlim(axs[0].get_xlim())
            axs[0].set_xlabel('')

            fig.suptitle(f'Distribution of {column}')
            plt.tight_layout()
            plt.show()

            # Add hover functionality
            cursor = mplcursors.cursor(strip, hover=True)

            # @cursor.connect("add")
            # def on_add(sel):
            #     sel.annotation.set(text=)

            plt.show()

        if plot_module == 'plotly':
            import plotly.express as px

            fig1 = px.histogram(df, x=column, color='GroupID', pattern_shape='location', marginal='rug',
                                hover_data=df.columns)
            fig2 = px.histogram(df, x=column, color='GroupID', pattern_shape='location', marginal='violin',
                                hover_data=df.columns)
            fig2.update_layout(plot_bgcolor='white')
            fig2.update_xaxes(
                mirror=True,
                ticks='outside',
                showline=True,
                linecolor='black',
                gridcolor='lightgrey'
            )
            fig2.update_yaxes(
                mirror=True,
                ticks='outside',
                showline=True,
                linecolor='black',
                gridcolor='lightgrey'
            )

            from dash import Dash, dcc, html, dash_table
            app = Dash()
            app.layout = [
                html.Div(children='My First App with Data and a Graph'),
                dcc.Graph(figure=fig1),
                dcc.Graph(figure=fig2),
                dash_table.DataTable(data=df.to_dict('records'), page_size=10)
            ]
            # app.run_server(debug=False, use_reloader=False)  # Turn off reloader if inside Jupyter
            app.run(debug=False, use_reloader=False)  # Turn off reloader if inside Jupyter

    def plot_heatmap_snr(
            self,
            properties: list[str] = None, methods: list[str] = None,
            vmax: float = None, vmin: float = 0, cmap: str = 'Reds'
    ):
        """
        Plot the heatmap of the signal-to-noise ratio (SNR) of the target values.

        Args:
            properties (list(str)): The list of properties to plot.
            methods (list(str)): The list of methods to plot.
            vmax (float): The maximum value of the colorbar.
            vmin (float): The minimum value of the colorbar.
            cmap (str): The colormap to use for the heatmap.

        Returns:
            None
        """
        # If properties and methods are not provided, use the unique properties
        if properties is None:
            if self.unique_properties is None:
                columns = [col for col in self.dataset.df_stat.columns if '_std' in col]
                # extract the properties from the columns
                properties = []  # e.g., 'CO2 Conversion (%)', 'Selectivity to CO (%)'
                for col in columns:
                    parts = col.rsplit('_', 2)
                    if len(parts) > 2:
                        properties.append(parts[0])
                self.unique_properties = list(set(properties))
            properties = self.unique_properties
        # If methods are not provided, use the unique methods
        if methods is None:
            methods = [col.split('_')[1] for col in self.dataset.df_stat.columns if '_std' in col]
            # make elements of methods unique
            methods = list(set(methods))

        # Make (len(properties) x (len(methods)) DataFrame consists of corresponding SNR values
        # Make the row index corresponds to methods and the column index corresponds to properties
        df_snr = pd.DataFrame(index=methods, columns=properties)
        for prop in properties:
            for method in methods:
                column = f'{prop}_{method}_std'
                snr = self.dataset.df_stat[column].iloc[-1] / self.dataset.df_stat[column].iloc[:-1].max()
                df_snr.loc[method, prop] = snr
        self.df_snr = df_snr
        # Ensure the DataFrame is filled with float values
        df_snr = df_snr.astype(float)

        # Plot the heatmap
        fig, ax = plt.subplots(figsize=(10.3, 10))
        # Set the font sizes for the plot
        plt.rcParams.update({'font.size': 15})
        vmax = df_snr.max().max() if vmax is None else vmax
        sns.heatmap(
            df_snr,
            annot=True, fmt='.2f',
            annot_kws={'fontsize': 12}, # set fontsize for the annotation
            cmap=cmap,
            cbar_kws={'label': 'Signal-to-Noise Ratio (SNR)'},
            vmax=vmax, vmin=vmin,
            ax=ax, # use the ax parameter to plot the heatmap on the provided axis
        )
        # Rotate xtick labels
        plt.xticks(ha='left', fontsize=15, rotation=-30)
        plt.yticks(ha='center', fontsize=15)
        plt.tight_layout()
        plt.show()

    def pearson_correlation_target(self):
            """
            Calculate the Pearson correlation coefficient between the target values.

            Returns:
                None
            """

            pass
