# -*- coding: utf-8 -*-
from tensorflow import keras
import tensorflow as tf
import datetime
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from IPython.display import display, Markdown
import fidle.pwk as ooo

def get_model_v1(shape,nb_layers,list_neurones,neurones_sortie,list_act,f_opt,crit_loss,list_metrics):
       model = keras.models.Sequential()
       for i in range (0,nb_layers):
           model.add(keras.layers.Dense(list_neurones[i], activation=list_act[i], input_shape=shape))
        
       model.add(keras.layers.Dense(neurones_sortie))
       model.compile(optimizer = f_opt,
                     loss      = crit_loss,
                     metrics   = list_metrics )
       return model



def regression (list_args):
    # List_args=nom_fichier,frac_train, shape, liste_entree, liste_sortie, nb_layers,list_neurones,neurones_sortie,list_act,f_opt,crit_loss,list_metrics, nb_epochs,best_model_dir, best_model_name
    [nom_fichier,frac_train, shape, liste_entree, liste_sortie, nb_layers,list_neurones,neurones_sortie,list_act,f_opt,crit_loss,list_metrics, nb_epochs,best_model_dir, best_model_name] =list_args
    
    data = pd.read_csv(nom_fichier,sep=',', header=0)

    display(data.head())
    print('Données manquantes : ',data.isna().sum().sum(), '  Shape is : ', data.shape)
    
    # Preparer donnees
    # ---- Split => train, test
    #
    data_train = data.sample(frac=0.8, axis=0)
    data_test  = data.drop(data_train.index)

    # ---- Split => x,y (medv is price)
    #
    x_train = data_train[liste_entree]
    y_train = data_train[liste_sortie]
    x_test  = data_test[liste_entree]
    y_test  = data_test[liste_sortie]

    print('Original data shape was : ',data.shape)
    print('x_train : ',x_train.shape, 'y_train : ',y_train.shape)
    print('x_test  : ',x_test.shape,  'y_test  : ',y_test.shape)
    
    display(x_train.describe().style.format("{0:.2f}").set_caption("Before normalization :"))

    mean = x_train.mean()
    std  = x_train.std()
    x_train = (x_train-mean)/std
    x_test  = (x_test-mean)/std

    display(x_train.describe().style.format("{0:.2f}").set_caption("After normalization :"))

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_test,  y_test  = np.array(x_test),  np.array(y_test)
    
    #Modele
    model=get_model_v1(shape,nb_layers,list_neurones,neurones_sortie,list_act,f_opt,crit_loss,list_metrics)
   
    model.summary()
    os.makedirs('./'+best_model_dir,   mode=0o750, exist_ok=True)
    save_dir = './'+best_model_dir+'/'+best_model_name+'.h5'
    savemodel_callback = keras.callbacks.ModelCheckpoint(filepath=save_dir, verbose=1, save_best_only=True)
    
    
    # Run Modele
    history = model.fit(x_train,
                    y_train,
                    epochs          = nb_epochs,
                    batch_size      = 10,
                    verbose         = 0,
                    validation_data = (x_test, y_test),
                    callbacks        = [savemodel_callback])
    
    ooo.plot_history(history, save_dir='./'+best_model_dir, plot={'MSE' :['mse', 'val_mse'],
                                                                  'MAE' :['mae', 'val_mae'],
                                                                  'LOSS':['loss','val_loss']} )
    plt.show()
    

    return x_test, y_test, x_train, y_train
  