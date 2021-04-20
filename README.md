# Neural-Network-model-for-complex-concentrated-alloys
Combinatorial design of complex concentrated alloys: a rational approach in thin film

## Databases content:
The databases are associating a composition measured by EDX with hardness and elastic modulus measurements. For Each composition, 5 nanoindentation test were carried out.


|   Composition (%at)    |   Compositions (%m)    | E (GPa) | H (GPa) |                
|----+----+----+----+----|------------------------|---------|---------|
| Zr | Nb | Mo | Ti | Cr | Zr | Nb | Mo | Ti | Cr |         |         |
|


## The databases:
  - Raw_data                          : acquired data without any treatment
  - Raw_data_corrected                : acquired data with hardness H an elastic modulus E correction with respect to calibration measurment on fused silica
  - Raw_data_threshold_with_outliers  : corrected data with a H and E threshold (no negative or zero values)
  - Raw_data_without_outliers         : Q_test to find extremum outliers with a 0.05 confidence index, lead after a Shapiro normality test. There are only 5 values of H and E for each composition value, thus the normality test is not lead in perfect condition, though Shhapiro is adapted to small 
                                      samples.
  - Raw_data_without_outliers_averages: the Raw_data_without_outliers are grouped per composition, the hardness and modulus valus are averged and the standard deviations stdH and stdE are computed and added to the database as 2 supplementary outputs
## The codes
### ``model_fit module``

This Python module contains a fonction that allows to create and train a regression model of which parameters are chosen previously.\
It contains: 
``regression``: list of arguments must contain : 
- filename : allow to read the database file
- frac_train : training fraction of the database ( eg: 0.8)
- shape : input layer shape 
- input_list : header of the database columns containing the input data
- output_list : headers of the database columns containing the output data
- nb_layers
- list_neurones
- neurones_sortie
- list_act
- f_opt
- crit_loss
- list_metrics
- nb_epochs
- best_model_dir
- best_model_name
    
