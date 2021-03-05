import pandas as pd
import numpy as np
import logging
from sklearn.impute import IterativeImputer
from sklearn.impute import KNNImputer

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s[%(name)s][%(levelname)s] %(message)s',
                    datefmt='[%Y-%m-%d][%H:%M:%S]')

logger = logging.getLogger(__name__)


class Preprocces:
    """Class to Preprocess data"""

    def __init__(self, data):
        self.data = data

    def fill_categorical_na(self, df):
        """
        Impute categorical NaN with two methods
        Args:
            :df: Dataframe
        Returns:
            :df_result: Dataframe without NaN
        """

        df_result = df.copy()

        imputer = KNNImputer(n_neighbors=2, weights="uniform")
        df_result.columns
        data_cat_imputed = imputer.fit_transform(df_result)
        
        for i in range(data_cat_imputed.shape[1]):
            df_result[df_result.columns[i]] = data_cat_imputed[:, i]

        return df_result

    def fill_numerical_na(self, columns, df):
        """
        Impute numerical NaN with IterativeImputer

        Args:
            :columns: Columns to impute NaN
            :df: Dataframe
        Returns:
            :df_result: Dataframe without NaN
        """

        imp = IterativeImputer(missing_values=np.nan, max_iter=15, random_state=0)

        imp.fit(df[columns])

        data_imputed = imp.transform(df[columns])
        for i in range(data_imputed.shape[1]):
            df[columns[i]] = data_imputed[:, i]
        df.year = round(df.year)
        df.rangoempleados = round(df.rangoempleados)

        return df

    def transform_time(self, df):
        """
        Transform time in company's years
        Args:
            :df: Dataframe
        Returns:
            :df['year']: New column from dataframe that contain company's years
        """

        df['year'] = pd.to_datetime(df.anyofundacion, format='%d/%m/%Y')
        df['year'] = 2021 - df.year.dt.year
        return df['year']

    def clean_dataframe(self, cols):
        """
        Clean dataframe and remove features
        
        Args:
            :cols: Columns to remove

        Returns:
            :df_result: Dataframe with column to train a model 
        
        """

        df = self.data
        logger.info('Shape of dataframe:' + str(df.shape))
        logger.info(f'Transform column anyofundacion to company years example: 2021-2017 --> 4 year')
        df['year'] = self.transform_time(df)
        logger.info(f'Remove variables {cols}')
        
        df = df.drop(columns=cols)
        logger.info('Shape of dataframe:' + str(df.shape))
        
        logger.info('Dataframe is cleaned')
        return df
