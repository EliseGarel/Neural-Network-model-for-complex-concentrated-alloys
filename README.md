# Neural-Network-model-for-complex-concentrated-alloys
Combinatorial design of complex concentrated alloys: a rational approach in thin film

Database content:
The databases are associating a composition measure by EDX with hardness and elastic modulus measurements. For Each composition, 5 nanoindentation test were carried out.

The databases: 
  Raw_data                          : acquired data without any treatment
  Raw_data_corrected                : acquired data with hardness H an elastic modulus E correction with respect to calibration measurment on fused silica
  Raw_data_threshold_with_outliers  : corrected data with a H and E threshold (no negative or zero values)
  Raw_data_without_outliers         : Q_test to find extremum outliers with a 0.05 confidence index, lead after a Shapiro normality test. There are only 5 values of H  
                                      and E for each composition value, thus the normality test is not lead in perfect condition, though Shhapiro is adapted to small 
                                      samples.
  Raw_data_without_outliers_averages: the Raw_data_without_outliers are grouped per composition, the hardness and modulus valus are averged and the standard deviations 
                                      stdH and stdE are computed
