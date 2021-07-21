import pandas as pd

#Import Tournament data
tournament_location = 'D:\\KF_Repo\\PGA_Golf\\Tournament_level_model\\Data_manipulation\\PGA_subset.csv'
tournament_data = pd.read_csv(tournament_location)

# Import Hole Level data
hole_level_location = 'D:\\KF_Repo\\PGA_Golf\\Tournament_level_model\\Data_manipulation\\PGA_subset_hole_level.csv'
hole_level_data = pd.read_csv(hole_level_location)


#https://medium.com/@jamesmazzolajd/on-strokes-gained-explained-1e92758ef93d

#Strokes gained comes from shotlink data - uses PGA tour baseline (tour average)
#sg_ott - strokes gained off the tee on par 4s and par 5s
#sg_app - strokes gained on approach shots on par 4s and par 5s
#sg_arg - stokes gained around the green (within 30 yards)
#sg_t2g - stokes gained tee to green (ott + app + arg)
#sg_putt - strokes gained putting on the green
#sg_total

#Additional
#Stokes gained ball striking = sg ott + sg app

#Filter data for 2021 US Open
us_open_2021 = subset_data[subset_data['tournament id']==401243414]

#Compare Strokes
compare = us_open_2021[['sg_total','hole_par','strokes','made_cut']]
compare['score'] = compare['strokes'] - compare['hole_par']

#Jon Rahm finished 6 under
#3.78 strokes gained (better than tour average)














