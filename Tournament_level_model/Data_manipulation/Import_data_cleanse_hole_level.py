# Data downloaded from https://www.advancedsportsanalytics.com/pga-raw-data

import pandas as pd

#Import Tournament data
raw_data_location = 'D:\\KF_Repo\\PGA_Golf\\Tournament_level_model\\Raw_data\\ASA All PGA Raw Data - Hole Level.csv'
hole_level = pd.read_csv(raw_data_location)


#Select fewer columns
reduce_columns = hole_level[['player', 'player id', 'hole','par',
                             'Strokes Above Par',
                             'round',
                             'final position',
       'strokes', 'made_cut', 'major',
       'tournament name','tournament id', 'course', 'date',
       'purse', 'season']]

#Save updated file as csv
reduce_columns.to_csv('D:\\KF_Repo\\PGA_Golf\\Tournament_level_model\\Data_manipulation\\PGA_subset_hole_level.csv', index=False)