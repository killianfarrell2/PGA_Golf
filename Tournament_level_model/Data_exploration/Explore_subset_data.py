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

#Filter data for 2020 PGA 
us_pga_2020 = tournament_data[tournament_data['tournament id']==401219481]
us_pga_2020['round_par'] = us_pga_2020['hole_par']/us_pga_2020['n_rounds']
us_pga_2020['Final_score'] = us_pga_2020['strokes'] - us_pga_2020['hole_par']


#Filter data for 2020 PGA 
us_pga_2020_hole = hole_level_data[hole_level_data['tournament id']==401219481]

R1 = us_pga_2020_hole[us_pga_2020_hole['round']==1]
R2 = us_pga_2020_hole[us_pga_2020_hole['round']==2]
R3 = us_pga_2020_hole[us_pga_2020_hole['round']==3]
R4 = us_pga_2020_hole[us_pga_2020_hole['round']==4]

#Break out into rounds
g1 = pd.DataFrame(R1.groupby(['player id'])['Strokes Above Par'].sum())
g1 = g1.rename(columns={'Strokes Above Par': 'R1'})

g2 = pd.DataFrame(R2.groupby(['player id'])['Strokes Above Par'].sum())
g2 = g2.rename(columns={'Strokes Above Par': 'R2'})

g3 = pd.DataFrame(R3.groupby(['player id'])['Strokes Above Par'].sum())
g3 = g3.rename(columns={'Strokes Above Par': 'R3'})

g4 = pd.DataFrame(R4.groupby(['player id'])['Strokes Above Par'].sum())
g4 = g4.rename(columns={'Strokes Above Par': 'R4'})

 
#Merge individual rounds
merge_rounds = pd.merge(us_pga_2020, g1, left_on=['player id'], right_index=True)
merge_rounds = pd.merge(merge_rounds, g2, left_on=['player id'], right_index=True)
merge_rounds = pd.merge(merge_rounds, g3, left_on=['player id'], right_index=True)
merge_rounds = pd.merge(merge_rounds, g4, left_on=['player id'], right_index=True)





#Jon Rahm finished 6 under
#3.78 strokes gained (better than tour average)














