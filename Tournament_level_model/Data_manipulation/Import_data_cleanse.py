# Data downloaded from https://www.advancedsportsanalytics.com/pga-raw-data

import pandas as pd

#Import Tournament data
raw_data_location = 'D:\\KF_Repo\\PGA_Golf\\Tournament_level_model\\Raw_data\\ASA All PGA Raw Data - Tourn Level.csv'
tournament = pd.read_csv(raw_data_location)

#Select fewer columns
reduce_columns = tournament[['Player_initial_last', 'tournament id', 'player id', 'hole_par',
       'strokes', 'n_rounds', 'made_cut', 
       'player', 'tournament name', 'course', 'date',
       'purse', 'season', 'no_cut', 'Finish', 'sg_putt', 'sg_arg', 'sg_app',
       'sg_ott', 'sg_t2g', 'sg_total']]

#Save updated file as csv
reduce_columns.to_csv('D:\\KF_Repo\\PGA_Golf\\Tournament_level_model\\Data_manipulation\\PGA_subset.csv', index=False)