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

# Set values for Model as test set for round 1
observed_golfers_shared.set_value(observed_golfers_test)

#Output is decimal number score to par - need to add to par and then round
num_samples = 10000
with model:
    pp_test_set = pm.sample_posterior_predictive(trace,samples=num_samples)
    
# Round scores to the nearest whole number
# Have 100 simulations for each golfer for each tournament
pp_test_rounded = {'golfer_to_par': pp_test_set['golfer_to_par'].round(0)}

def sim_score(Round):
        draw = np.random.randint(0,num_samples)
        scores = pd.DataFrame(pp_test_rounded['golfer_to_par'][draw,:])
        scores = scores.rename(columns={0: Round})
        return scores

# Return Top X
def top_x(x,df,i):
        top_x = df[df['rank_final']<=x].reset_index(drop=True)
        top_x = top_x[['player','final_score','rank_final']]
        top_x['sim_number'] = i
        return top_x

def get_percent(df,string):
    counts = pd.DataFrame(df['player'].value_counts())
    counts = counts.rename(columns={'player': "Count_"+str(string)})
    num_sims = df['sim_number'].nunique()
    counts['percent_'+str(string)] = counts["Count_"+str(string)]/num_sims
    return counts

    
def sim_tournament(num_simulations):
    
    # Start tournament simulation
    # Create empty dataframe
    store_winners = pd.DataFrame()
    store_top_5 = pd.DataFrame()
    store_top_10 = pd.DataFrame()
    store_top_20 = pd.DataFrame()
    store_made_cut = pd.DataFrame()
    
    # Run for number of simulations specified below
    for i in range(num_simulations):
             
            # Select random draw from simulated scores to get round 1 and round 2 scores
            scores_r1 = sim_score("R1")
            scores_r2 = sim_score("R2")
            scores_r3 = sim_score("R3")
            scores_r4 = sim_score("R4")
            
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
            winner = top_x(1,made_cut,i)
            
            # If more than 1 player finishes on top score - select winner at random (playoff)
            if winner.shape[0] > 1:
                rand_row = np.random.randint(0,winner.shape[0])
                winner = winner[rand_row:rand_row+1]
            
            # Select top 5, top 10, top 20, made_cut
            top_5 = top_x(5,made_cut,i)
            top_10 = top_x(10,made_cut,i)
            top_20 = top_x(20,made_cut,i)
            top_cut = top_x(1000,made_cut,i)
        
            # Append to dataframe
            store_winners = store_winners.append(winner)
            store_top_5 = store_top_5.append(top_5)
            store_top_10 = store_top_10.append(top_10)
            store_top_20 = store_top_20.append(top_20)
            store_made_cut = store_made_cut.append(top_cut)
            
            print(i)
    
     # To get winner just need to count how many times out of 10k simulations he won 
    win_percent =  get_percent(store_winners,"win")
    top_5_percent = get_percent(store_top_5,"5")
    top_10_percent =  get_percent(store_top_10,"10")
    top_20_percent = get_percent(store_top_20,"20")
    cut_percent = get_percent(store_made_cut,"cut")
    
    # Merge results
    results_sim = pd.merge(win_percent,top_5_percent,left_index=True,right_index=True,how='outer')
    results_sim = pd.merge(results_sim,top_10_percent,left_index=True,right_index=True,how='outer')
    results_sim = pd.merge(results_sim,top_20_percent,left_index=True,right_index=True,how='outer')
    results_sim = pd.merge(results_sim,cut_percent,left_index=True,right_index=True,how='outer')

    return results_sim



results_all = sim_tournament(2)



