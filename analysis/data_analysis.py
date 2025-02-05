import matplotlib.pyplot as plt
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

    def compare_targets_std_dev(self, target_wise: bool = False, colormap: str = 'viridis'):
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
                    cmap = plt.cm.get_cmap('viridis', len(df_melted['GroupID'].unique()))
                    colors = [cmap(i) for i in range(len(df_melted['GroupID'].unique()))]
                    colors[-1] = 'hotpink'
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
                            self._generate_histogram(col.rstrip("_std"))
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

    def _generate_histogram(self, column: str):
        """
        Generate a histogram for the specified column.

        Args:
            column (str): The column to generate the histogram for.

        Returns:
            None
        """
        plt.figure()
        sns.histplot(
            self.dataset.df_us,
            x=column,
            hue='GroupID',
            kde=True
        )
        # sns.histplot(
        #     self.dataset.df_us_unique,
        #     x=column,
        #     hue='GroupID',
        #     kde=True
        # )
        plt.title(f'Histogram of {column}')
        plt.show()



