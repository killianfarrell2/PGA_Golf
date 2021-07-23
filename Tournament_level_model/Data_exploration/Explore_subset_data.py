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

import pandas as pd

#Import Tournament data
tournament_location = 'D:\\KF_Repo\\PGA_Golf\\Tournament_level_model\\Data_manipulation\\PGA_subset.csv'
tournament_data = pd.read_csv(tournament_location)

# Import Hole Level data
hole_level_location = 'D:\\KF_Repo\\PGA_Golf\\Tournament_level_model\\Data_manipulation\\PGA_subset_hole_level.csv'
hole_level_data = pd.read_csv(hole_level_location)


#Bad Initial Energy when using just 1 tournament - so combining all
tournament_data['round_par'] = tournament_data['hole_par']/tournament_data['n_rounds']
tournament_data['Final_score'] = tournament_data['strokes'] - tournament_data['hole_par']

# Divide data up into Rounds
R1 = hole_level_data[hole_level_data['round']==1]
R2 = hole_level_data[hole_level_data['round']==2]
R3 = hole_level_data[hole_level_data['round']==3]
R4 = hole_level_data[hole_level_data['round']==4]

#Break out into rounds
g1 = pd.DataFrame(R1.groupby(['tournament id','player id'])['Strokes Above Par'].sum())
g1 = g1.rename(columns={'Strokes Above Par': 'Round_Score'})
g1['Round'] = 1

g2 = pd.DataFrame(R2.groupby(['tournament id','player id'])['Strokes Above Par'].sum())
g2 = g2.rename(columns={'Strokes Above Par': 'Round_Score'})
g2['Round'] = 2

g3 = pd.DataFrame(R3.groupby(['tournament id','player id'])['Strokes Above Par'].sum())
g3 = g3.rename(columns={'Strokes Above Par': 'Round_Score'})
g3['Round'] = 3

g4 = pd.DataFrame(R4.groupby(['tournament id','player id'])['Strokes Above Par'].sum())
g4 = g4.rename(columns={'Strokes Above Par': 'Round_Score'})
g4['Round'] = 4

 
#Merge individual rounds
merge_1 = pd.merge(tournament_data, g1, left_on=['tournament id','player id'], right_index=True,how='inner')
merge_2 = pd.merge(tournament_data, g2, left_on=['tournament id','player id'], right_index=True,how='inner')
merge_3 = pd.merge(tournament_data, g3, left_on=['tournament id','player id'], right_index=True,how='inner')
merge_4 = pd.merge(tournament_data, g4, left_on=['tournament id','player id'], right_index=True,how='inner')


#Union Dataframes
union = pd.concat([merge_1, merge_2, merge_3,merge_4])
#Get total round score
union['Round_total'] = union['round_par'] + union['Round_Score']

#Drop column player id - Can't use golfer ID - get index 9621 is out of bounds error
union = union.drop(columns=['player id'])

# Create new column i_golfer
golfers = union.player.unique()
golfers = pd.DataFrame(golfers, columns=["golfer"])
golfers["i"] = golfers.index

# Add i column back to dataframe
union = pd.merge(union, golfers, left_on="player", right_on="golfer", how="left")
union = union.rename(columns={"i": "i_golfer"}).drop("golfer", 1)


# create a group using groupby
g_obs = union.groupby("player")['hole_par'].count()
  


#Downgraded Arviz to 0.11.
import pymc3 as pm
import arviz as az
import matplotlib.pyplot as plt

#Gaussian Inference - fat right tail
az.plot_kde(union['Round_total'].values, rug=True)
plt.yticks([1], alpha=0);

#Set values
golfers = union.i_golfer.values
golf_round = union.Round.values
observed_round_score = union.Round_total.values

#Get unique number of golfers
num_golfers = len(union.i_golfer.drop_duplicates())

#Drop dataframes
R1.drop(R1.index, inplace=True)
R2.drop(R2.index, inplace=True)
R3.drop(R3.index, inplace=True)
R4.drop(R4.index, inplace=True)
merge_1.drop(merge_1.index, inplace=True)
merge_2.drop(merge_2.index, inplace=True)
merge_3.drop(merge_3.index, inplace=True)
merge_4.drop(merge_4.index, inplace=True)

import theano.tensor as tt


#Normal Distribution - gives output of decimal numbers - we need whole numbers
#Poisson Distribution not acceptable as mean is not equal to variance

with pm.Model() as model:
    # global model parameters
    sd_att = pm.HalfStudentT("sd_att", nu=3, sigma=2.5)

    # golfer specific
    atts_star = pm.Normal("atts_star", mu=72, sigma=sd_att, shape=num_golfers)
   
    golfer_theta = atts_star[golfers]
    
    # likelihood of observed data
    golfer_score = pm.Poisson("golfer_score", mu=golfer_theta, observed=observed_round_score)

#golfer_score = pm.Normal("golfer_score", mu=golfer_theta,sigma=1, observed=observed_round_score)

#Set cores to 1
with model:
    trace = pm.sample(2000, tune=2000, cores=1)

# Create dataframe with trace
df_trace = pm.trace_to_dataframe(trace)


#Plot P{osterior of specific player}
pm.plot_posterior(trace['atts_star'][0])

#Right now output is decimal number - but we need round numbers
with model:
    pp_trace = pm.sample_posterior_predictive(trace,samples=50)




#Plot Distribution of Round 1 scores to Par

import seaborn as sns

r1_graph = sns.displot(merge_rounds,x='R1',kind='kde',fill=True)
r2_graph = sns.displot(merge_rounds,x='R2',kind='kde',fill=True)

#Equation
#Poisson Distribution
# Round Score = Par of Course (Fixed) 
#+ Course Difficulty Effect (Fixed for all players) + Round(Fixed for all players)
#+ Player skill(Normal) + Made Cut after Round 2 + Stokes Gained



















