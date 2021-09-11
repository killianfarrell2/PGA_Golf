import pandas as pd
#Downgraded Arviz to 0.11.
import pymc3 as pm
import arviz as az
import matplotlib.pyplot as plt
import theano.tensor as tt
import numpy as np

# Use a theano shared variable to be able to exchange the data the model runs on
from theano import shared

#Import Tournament data
data_location = 'D:\\KF_Repo\\PGA_Golf\\Tournament_level_model\\Data_manipulation\\model_data.csv'
data = pd.read_csv(data_location)

# Get count of rounds
g_count = pd.DataFrame(data.groupby("player")['hole_par'].count())
g_count = g_count.rename(columns={"hole_par": "Count_rounds"})

# Join count column
data = pd.merge(data, g_count, left_on="player", right_index=True, how="left")

#Filter out golfers with less than 28 rounds played
data = data[data["Count_rounds"]>=28]

#Drop column player id - Can't use player ID - get index 9621 is out of bounds error
data = data.drop(columns=['player id'])


#Split into training data with rough 80:20 split
training_data = data[data['date'] <'2020-10-01']
test_data = data[data['date'] >='2020-10-01']

# Create new column i_golfer for training data
golfers = training_data.player.unique()
golfers = pd.DataFrame(golfers, columns=["golfer"])
golfers["i"] = golfers.index

# Add i column back to dataframe
training_data = pd.merge(training_data, golfers, left_on="player", right_on="golfer", how="left")
training_data = training_data.rename(columns={"i": "i_golfer"}).drop("golfer", 1)


# Get golfers from test set
golfers_test = pd.DataFrame(test_data.player.unique())
# Rename column
golfers_test = golfers_test.rename(columns={0: "golfer"})

# Drop golfers that are already in golfers
cond = golfers_test["golfer"].isin(golfers['golfer'])
golfers_test.drop(golfers_test[cond].index, inplace = True)


# Add new golfers to golfers dataframe
golfers = golfers.append(golfers_test, ignore_index=True)
golfers["i"] = golfers.index

# Add i column back to dataframe
test_data = pd.merge(test_data, golfers, left_on="player", right_on="golfer", how="left")
test_data = test_data.rename(columns={"i": "i_golfer"}).drop("golfer", 1)


# Get Round 1 entries for each tournament
test_r1 = test_data[test_data['Round']==1]
# Select a tournament from round 1 - Augusta
tournament_r1 = test_r1[test_r1['tournament id']==401219478]

#Set values to be used as x
observed_golfers = training_data.i_golfer.values
# Set as shared so that can swap in golfers for test set
observed_golfers_shared = shared(observed_golfers)
# Get values for golfers from tournament round 1
observed_golfers_test = tournament_r1.i_golfer.values

# Get observed scores to use for model
observed_round_score = training_data.Round_Score.values

#Get unique number of golfers - shape will be ok below
num_golfers = len(training_data.i_golfer.drop_duplicates())


#Normal Distribution
with pm.Model() as model:

    mean_golfer_sd  = pm.Uniform('mean_golfer_sd',lower=0, upper=5)
    mean_golfer_mu = pm.Normal('mean_golfer_mu', mu=0, sigma=1)
    # golfer specific parameters
    mean_golfer = pm.Normal('mean_golfer', mu=mean_golfer_mu, sigma=mean_golfer_sd, shape=num_golfers)
    #standard deviation for each golfer - Inverse gamma is prior for standard deviation
    sd_golfer = pm.Uniform('sd_golfer',lower=0, upper=5, shape=num_golfers)     
    # Observed scores to par follow normal distribution
    golfer_to_par = pm.Normal("golfer_to_par", mu=mean_golfer[observed_golfers_shared], sigma=sd_golfer[observed_golfers_shared], observed=observed_round_score)
    # Prior Predictive checks - generate samples without taking data
    prior_checks = pm.sample_prior_predictive(samples=1000, random_seed=1234)


#Set cores to 1
# Tuning samples will be discarded
with model:
    trace = pm.sample(1000,chains=2, tune=1000, cores=1,random_seed=1234)


# Sample Posterior predictive
with model:
    pp_train = pm.sample_posterior_predictive(trace,samples=1000)

# Round scores to the nearest whole number
pp_train_rounded = {'golfer_to_par': pp_train['golfer_to_par'].round(0)}

# Set values for Model as test set for round 1
observed_golfers_shared.set_value(observed_golfers_test)

#Output is decimal number score to par - need to add to par and then round
num_samples = 10000
with model:
    pp_test_set = pm.sample_posterior_predictive(trace,samples=num_samples)
    
# Round scores to the nearest whole number
# Have 100 simulations for each golfer for each tournament
pp_test_rounded = {'golfer_to_par': pp_test_set['golfer_to_par'].round(0)}

       

# Start tournament simulation
# Create empty dataframe
store_winners = pd.DataFrame()

# Run for number of simulations specified below
for i in range(10000):
         
        # Select random draw from simulated scores to get round 1 and round 2 scores
        draw_r1 = np.random.randint(0,num_samples)
        scores_r1 = pd.DataFrame(pp_test_rounded['golfer_to_par'][draw_r1,:])
        scores_r1 = scores_r1.rename(columns={0: "R1"})
        
        draw_r2 = np.random.randint(0,num_samples)
        scores_r2 = pd.DataFrame(pp_test_rounded['golfer_to_par'][draw_r2,:])
        scores_r2 = scores_r2.rename(columns={0: "R2"})
        
        draw_r3 = np.random.randint(0,num_samples)
        scores_r3 = pd.DataFrame(pp_test_rounded['golfer_to_par'][draw_r3,:])
        scores_r3 = scores_r3.rename(columns={0: "R3"})
        
        draw_r4 = np.random.randint(0,num_samples)
        scores_r4 = pd.DataFrame(pp_test_rounded['golfer_to_par'][draw_r4,:])
        scores_r4 = scores_r4.rename(columns={0: "R4"})
        
        # get extra details from tournament
        scores_plus = tournament_r1[['player','tournament name','tournament id','made_cut']].reset_index(drop=True)
        # merge scores back
        scores = pd.merge(scores_plus,scores_r1,left_index=True,right_index=True)
        scores = pd.merge(scores,scores_r2,left_index=True,right_index=True)
        scores = pd.merge(scores,scores_r3,left_index=True,right_index=True)
        scores = pd.merge(scores,scores_r4,left_index=True,right_index=True)
        
        # Add round 1 and roound 2 scores
        scores['total_r1_r2'] = scores['R1'] + scores['R2']
        # Get rank after day 2
        scores['rank_r1_r2'] = scores['total_r1_r2'].rank(method='min')
        # Top 50 make cut and ties for augusta
        missed_cut = scores[scores['rank_r1_r2']> 50]
        # Drop round 3 and round 4 scores from players who didn't make the cut
        missed_cut = missed_cut.drop(columns=['R3', 'R4'])
        
        # Get players who made cut
        made_cut = scores[scores['rank_r1_r2']<= 50]
        # Get final score
        made_cut['final_score'] = made_cut['total_r1_r2'] + made_cut['R3'] + made_cut['R4']
        # Get final rank
        made_cut['rank_final'] = made_cut['final_score'].rank(method='min')
        
        # Select Winner
        winner = made_cut[made_cut['rank_final']==1].reset_index(drop=True)
        winner = winner[['player','final_score','rank_final']]
        
        # If more than 1 player finishes on top score - select winner at random
        if winner.shape[0] > 1:
            rand_row = np.random.randint(0,winner.shape[0])
            winner = winner[rand_row:rand_row+1]
        
        winner['sim_number'] = i
        
        # Append to dataframe
        store_winners = store_winners.append(winner)
        print(i)

# Get count number of times each player won the tournament
player_odds = pd.DataFrame(store_winners['player'].value_counts())
player_odds = player_odds.rename(columns={'player': "Count_Wins"})

# Get percentage chance
# Need to do calibration to check accuracy of percentages
# Highest percentage is 4% which is 25/1 odds
# Lowest is 0.0006 which is 1666/1 - seems too high
player_odds['percent_win'] = player_odds["Count_Wins"]/player_odds["Count_Wins"].sum()
    





