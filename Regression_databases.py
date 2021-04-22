##################################################
# Regression Neural Netwok on different databases#
##################################################

####################################################################################################
## 1) Import libraries

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

######################################################################################################
## 2) Parameters of the neural network

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


######################################################################################################
## 3) Run models

list_x_test=[]
list_y_test=[]
list_x_train=[]
list_y_train=[]

# Regression on the 5 different databases
for i in range (0,5):
    x_test, y_test, x_train, y_train=mf.regression ([filename[i],
                                                     frac_train, 
                                                     shape, 
                                                     liste_entree, 
                                                     liste_sortie[i], 
                                                     nb_layers,
                                                     list_neurones,
                                                     neurones_sortie[i],
                                                     list_act,
                                                     f_opt,
                                                     crit_loss,
                                                     list_metrics, 
                                                     nb_epochs,
                                                     best_model_dir[i], 
                                                     best_model_name[i]])
    list_x_test.append(x_test)
    list_y_test.append(y_test)
    list_x_train.append(x_train)
    list_y_train.append(y_train)


######################################################################################################
## 4) Evaluate best models

loaded_model=[]
score=[]

# Load the best model for each database as well as the x and y values used for test
# Score iscontaining the loss, mae and mse. 

for i in range (0,5):
    loaded_model.append(keras.models.load_model('./'+best_model_dir[i]+'/'+best_model_name[i]+'.h5'))
    loaded_model[i].summary()
    print("Loaded.")
    score = loaded_model[i].evaluate(list_x_test[i], list_y_test[i], verbose=0)
    print('Test loss      : {:5.4f}'.format(score[0]))
    print('Test mae  : {:5.4f}'.format(score[1]))
    print('Test mse  : {:5.4f}'.format(score[1]))
    
######################################################################################################   
## 5) Plot predictions vs experimental for all databases + residual historgram

# For the 5 databases, we plot the values of model predictions vs experimental values, for train and test values. 
# The residuals are the difference between these prediction and experimental values. The histogram show the distribution of these residuals.
for i in range (0,5):

    predictions_test = loaded_model[i].predict(list_x_test[i])
    predictions_train = loaded_model[i].predict(list_x_train[i])
    y_train=list_y_train[i]
    y_test=list_y_test[i]
   
    for j in range (0,len(y_train[1,:])):
        res_train=(predictions_train[:,j]- y_train[:,j])
        res_test=(predictions_test[:,j]- y_test[:,j])
        
        plt.plot(predictions_train[:,j],y_train[:,j],'+')
        plt.title(best_model_dir[i]+': ' + liste_sortie[i][j]+" prediction vs "+liste_sortie[i][j]+" experimental on training data")
        plt.savefig('./'+best_model_dir[i]+'/'+liste_sortie[i][j]+'_train')
        plt.show()
        
        plt.plot(predictions_test[:,j],y_test[:,j],'+')
        plt.title(best_model_dir[i]+':  '+liste_sortie[i][j]+" prediction vs "+liste_sortie[i][j]+" experimental on testing data")
        plt.savefig('./'+best_model_dir[i]+'/'+liste_sortie[i][j]+'_test')
        plt.show()
        
        plt.hist(res_train) 
        plt.title(best_model_dir[i]+': residus of '+liste_sortie[i][j] +' for train')
        plt.savefig('./'+best_model_dir[i]+'/'+'Residus '+liste_sortie[i][j]+'_train')
        plt.show()
        
        plt.hist(res_test)
        plt.title(best_model_dir[i]+': residus of '+liste_sortie[i][j] +' for test')
        plt.savefig('./'+best_model_dir[i]+'/''Residus '+liste_sortie[i][j]+'_test')
        plt.show()
