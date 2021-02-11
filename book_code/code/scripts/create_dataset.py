import os
import logging
import pandas as pd

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s[%(name)s][%(levelname)s] %(message)s',
                    datefmt='[%Y-%m-%d][%H:%M:%S]')


logger = logging.getLogger(__name__)

def genererate_dataframe(path):
    """
    Generate dataframe from folder dataset
    
    Args:
        :path: Folder's Path

    Returns:
        :df: Dataframe with values 
        :df_client: Dataframe with  id clients
    
    """
    logger.info('Loading files')
    data_file = 'BD_DC.csv'
    client_file = 'BD_clientes.csv'
    
    df = pd.read_csv(path+data_file,delim_whitespace=True)
    df_client = pd.read_csv(path+client_file)
    
    logger.info('Dataframes generated')
   
    return df, df_client
