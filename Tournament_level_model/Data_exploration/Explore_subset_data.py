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
g1 = g1.rename(columns={'Strokes Above Par': 'Round_Score'})
g1['Round'] = 1

g2 = pd.DataFrame(R2.groupby(['player id'])['Strokes Above Par'].sum())
g2 = g2.rename(columns={'Strokes Above Par': 'Round_Score'})
g2['Round'] = 2

g3 = pd.DataFrame(R3.groupby(['player id'])['Strokes Above Par'].sum())
g3 = g3.rename(columns={'Strokes Above Par': 'Round_Score'})
g3['Round'] = 3

g4 = pd.DataFrame(R4.groupby(['player id'])['Strokes Above Par'].sum())
g4 = g4.rename(columns={'Strokes Above Par': 'Round_Score'})
g4['Round'] = 4

 
#Merge individual rounds
merge_1 = pd.merge(us_pga_2020, g1, left_on=['player id'], right_index=True,how='left')
merge_2 = pd.merge(us_pga_2020, g2, left_on=['player id'], right_index=True,how='left')
merge_3 = pd.merge(us_pga_2020, g3, left_on=['player id'], right_index=True,how='left')
merge_4 = pd.merge(us_pga_2020, g4, left_on=['player id'], right_index=True,how='left')


#Union Dataframes
union = pd.concat([merge_1, merge_2, merge_3,merge_4])
#Get total round score
union['Round_total'] = union['round_par'] + union['Round_Score']

#Rename column
union = union.rename(columns={"player id": "player_id"})


#Downgraded Arviz to 0.11.
import pymc3 as pm
import arviz as az
import matplotlib.pyplot as plt

#Gaussian Inference - fat right tail
az.plot_kde(union['Round_total'].values, rug=True)
plt.yticks([1], alpha=0);


with pm.Model() as model:
    # global model parameters
    sd_att = pm.HalfStudentT("sd_att", nu=3, sigma=2.5)
    sd_def = pm.HalfStudentT("sd_def", nu=3, sigma=2.5)


#Set cores to 1
with model:
    trace = pm.sample(1000, tune=1000, cores=1)



#Plot Distribution of Round 1 scores to Par

import seaborn as sns

r1_graph = sns.displot(merge_rounds,x='R1',kind='kde',fill=True)
r2_graph = sns.displot(merge_rounds,x='R2',kind='kde',fill=True)

#Equation
#Poisson Distribution
# Round Score = Par of Course (Fixed) 
#+ Course Difficulty Effect (Fixed for all players) + Round(Fixed for all players)
#+ Player skill(Normal) + Made Cut after Round 2 + Stokes Gained



















