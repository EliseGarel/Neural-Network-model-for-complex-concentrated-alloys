# Regression Neural Netwok on different databases

# 1) Import libraries

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from IPython.display import display, Markdown, clear_output
import fidle.pwk as ooo
from importlib import reload
import modele_fit as mf
from pathos.multiprocessing import ProcessingPool as Pool


# 2) Parameters of the neural network

frac_train=0.8
shape=(5,) 
liste_entree=['Zr_at','Nb_at','Mo_at','Ti_at','Cr_at']
liste_sortie= [['E (GPa)','H (GPa)'],
               ['E (GPa)','H (GPa)'],
               ['E (GPa)','H (GPa)'],
               ['E (GPa)','H (GPa)'],
               ['E (GPa)','H (GPa)','stdE (GPa)','stdH (GPa)']]

nb_layers=2
list_neurones=[20,20]
neurones_sortie= [2,2,2,2,4]
list_act=['relu','relu']
f_opt='rmsprop'
crit_loss='mse'
list_metrics=['mse','mae']
nb_epochs=200
filename=['A:/BASE DE DONNEES/Nanoindentation/Python_extrac/Raw_data.csv',
          'A:/BASE DE DONNEES/Nanoindentation/Python_extrac/Raw_data_corrected.csv',
          'A:/BASE DE DONNEES/Nanoindentation/Python_extrac/Threshold_data_with_outliers.csv',
          'A:/BASE DE DONNEES/Nanoindentation/Python_extrac/Data_without_outliers.csv',
          'A:/BASE DE DONNEES/Nanoindentation/Python_extrac/Data_without_outliers_averaged.csv']

best_model_dir=['Raw_data',
                'Raw_data_corrected',
                'Threshold_data_with_outliers',
                'Data_without_outliers',
                'Data_without_outliers_averaged']
          
best_model_name=['best_Raw_data',
                'best_Raw_data_corrected',
                'best_Threshold_data_with_outliers',
                'best_Data_without_outliers',
                'best_Data_without_outliers_averaged']     


