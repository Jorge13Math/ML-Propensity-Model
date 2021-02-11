# Introduction
This repository aims to show the result and the methodology used to achieve the objectives of technical test. The tasks is:
* Find next potential clients, using Python programming language for their development.

# Getting Started

1. Installation process:
    * Install conda
    * conda create -n testdc python=3.6
    * conda activate testdc
    * pip install -r requirements.txt
    
1. Software dependencies
    * All python packages needed are included in `requirements.txt`

1. Run notebook
    * The notebook `Analysis.ipynb` contains all the process to achive the goal.
    * Notebook's structure:
        * Imports
        * Loading data
        * EDA
        * Preprocess data
        * Train and evaluate the a model
        * Making predictions

1. Generate book
    * Technicall test:
        * jb build book_code

# Repository structure 
The repository is structured in the following folders:
* book_code:
    * code: You will find the forlder scripts that contains the python scripts and a folder called "notebooks" that contains the notebook of all analysis. 

    * data: You will find the following datasets.
        * data_file = 'BD_DC.csv'
        * client_file = 'BD_clientes.csv'

    * models: You will find the models already trained.
        * model_lr.pickle

    * results: You will find the `predict_potencial_client.csv` that contains all information about the clients and sorted in descending order by potential customers.

# Book
* Code test: Go to `/book_code/_build/html/` and open `index.html` to see the notebook and all analysis in book format.
