import matplotlib.pyplot as plt
import mplcursors
import pandas as pd
import seaborn as sns
import numpy as np
import warnings
from matplotlib.widgets import Button

from data.extract import DataForGP

class DataAnalysis:
    def __init__(self, dataset: DataForGP = None):
        if dataset is None:
            raise ValueError("Please provide a dataset.")

        self.dataset = dataset
        self.unique_properties = None

    def compare_targets_std_dev(self, target_wise: bool = False, colormap: str = 'tab10'):
        """
        Compare the standard deviation of the target values for each column.

        Args:
            target_wise (bool): If True, compare standard deviations target-wise; otherwise, compare overall.
            colormap (str): The colormap to use for the plot.

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
                fig, axs = plt.subplots(nrows, ncols, figsize=(13, 6))
                axs = np.ravel(axs)
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
                    axs[i].set_title(f'{column.rstrip("_std")}', fontsize=10)

                    # Show ylabel only for the leftmost axes, and  xlabel only for the lowest axes
                    if i % ncols != 0:
                        axs[i].set_ylabel('')
                    if i < (nrows - 1) * ncols:
                        axs[i].set_xlabel('')

                    # Make the axes itself a button
                    def on_click(event, col=column, ax=axs[i]):
                        if event.inaxes == ax:
                            self._generate_histogram(column=col.rstrip("_std"), cmap=colormap)
                    fig.canvas.mpl_connect('button_press_event', on_click)

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

    def _generate_histogram(self, column: str, cmap: str = 'tab10'):
        """
        Generate a histogram for the specified column.

        Args:
            column (str): The column to generate the histogram for.
            cmap (str): The colormap to use for the plot.

        Returns:
            None
        """
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
            test = df_stat[df_stat['GroupID'] == group]['experiment_date']
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
        # test = df[df['GroupID'] == 'total']['location'] #= 'all'

        # Plot the histogram
        # fig, ax = plt.subplots()

        import plotly.express as px

        # hist = sns.histplot(
        #     df, x=column, hue='GroupID', shrink=0.95, multiple='dodge', stat='count', palette=cmap, kde=True, ax=ax
        # )
        #
        # num_elements_group = df_stat[f'{column}_list'][:-1].apply(len).tolist() # slicing: excluding 'total' group
        # ax.set_ylim(0, np.max(num_elements_group))
        # plt.title(f'Histogram of {column}')

        fig1 = px.histogram(df, x=column, color='GroupID', pattern_shape='location', marginal='rug',
                            hover_data=df.columns)
        fig2 = px.histogram(df, x=column, color='GroupID', pattern_shape='location', marginal='violin',
                            hover_data=df.columns)
        fig2.update_layout(
            plot_bgcolor='white',
        )
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

        # plt.show()

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


