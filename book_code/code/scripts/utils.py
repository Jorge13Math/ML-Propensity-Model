import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s[%(name)s][%(levelname)s] %(message)s',
                    datefmt='[%Y-%m-%d][%H:%M:%S]')


logger = logging.getLogger(__name__)

class Utils():
    """Class to plot and explore data"""

    def __init__(self):
        logger.info('Class is initialized')


    def plot_variable(self,df,column,df_result=False):
        """
        Plot variable by column
        
        Args:
            :df: Dataframe
            :column: Column to plot
            :df_result: True or False to return dataframe created

        Returns:
            :df_result: Dataframe with values 
        
        """
        self.df_column = pd.DataFrame({f"{column}":eval(f"df.{column}").value_counts().index, 'Count':eval(f"df.{column}").value_counts().values})
        if df_result:
            return self.df_column
        else:
            self.df_column.plot(x=column, y='Count', kind='bar', legend=False, grid=True, figsize=(10, 5))
            plt.title(f"{column} for category ")
            plt.ylabel('Count')
            plt.show()

    def plot_variable_per_target(self,df,column,target,df_result=False):
        """
        Plot variable by column ang target
        
        Args:
            :df: Dataframe
            :column: Column to plot
            :target: Column that is a traget value in the dataframe
            :df_result: True or False to return dataframe created

        Returns:
            :df_result: Dataframe with values 
        
        """
        self.df_class = pd.DataFrame({'0':eval(f"df[df[target]==0].{column}").value_counts(), '1':eval(f"df[df[target]==1].{column}").value_counts()},index = eval(f"df.{column}").value_counts().index)
        if df_result:
            return self.df_class
        else:
            self.df_class.plot.bar(rot=90,grid=True, figsize=(10, 5))
            plt.title(f"{column} for category and class ")
            plt.ylabel('Count')
            plt.show()

    def plot_variables_nan(self,df):
        """
        Plot variable with NaN
        
        Args:
            :df: Dataframe
        
        """
        
        self.df_nan =pd.DataFrame({'Variables':df.isna().sum().index, 'Number_of_nan':df.isna().sum().values})
        self.df_nan.plot(x='Variables', y='Number_of_nan', kind='bar', legend=False, grid=True, figsize=(10, 5))
        plt.title('NaN for variable in data ')
        plt.ylabel('Count')
        plt.show()

    











