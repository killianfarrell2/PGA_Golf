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

# Create new column i_golfer
golfers = data.player.unique()
golfers = pd.DataFrame(golfers, columns=["golfer"])
golfers["i"] = golfers.index

# Add i column back to dataframe
data = pd.merge(data, golfers, left_on="player", right_on="golfer", how="left")
data = data.rename(columns={"i": "i_golfer"}).drop("golfer", 1)

# Create new column i_course
courses = data.course.unique()
courses = pd.DataFrame(courses, columns=["course"])
courses["i"] = courses.index

# Add i column back to dataframe
data = pd.merge(data, courses, left_on="course", right_on="course", how="left")
data = data.rename(columns={"i": "i_course"})

# Get count of dates
date_count = pd.DataFrame(data.groupby("date")['hole_par'].count())

#Split into training data with rough 80:20 split
training_data = data[data['date'] <'2020-10-01']
test_data = data[data['date'] >='2020-10-01']


#Set values to be used as x
observed_golfers = training_data.i_golfer.values
observed_golfers_shared = shared(observed_golfers)
# Get values for golfers from test set
observed_golfers_test = test_data.i_golfer.values

observed_round_score = training_data.Round_Score.values
test_set_round_score = test_data.Round_Score.values 
observed_courses = training_data.i_course.values

#Get unique number of golfers - shape will be ok below
num_golfers = len(training_data.i_golfer.drop_duplicates())
num_courses = len(training_data.i_course.drop_duplicates())

#Normal Distribution - gives output of decimal numbers - we need whole numbers
#Poisson Distribution - is discreet but may not be accurate
# Think that Normal Distribution needs to be taken - and then round values from trace
# 51 divergences with intercept - removed and got slight warning
# Poisson and Normal similar at large numbers - mean equals variance for poisson
# This means for Poisson - low scores are a lot more likely
# Negative Binomial may be more appropriate
# Prior is belief before evidence is taken into account
# Intercept is expected mean value of y (golfer to par) when all x = 0
# When intercept gets added in it is less stable for mean golfer

with pm.Model() as model:
    
    #Hyper Priors
    mean_golfer_sd  = pm.HalfCauchy('mean_golfer_sd', beta=1)
    mean_golfer_mu = pm.Normal('mean_golfer_mu', mu=0, sigma=1)
   
    # golfer specific parameters
    mean_golfer = pm.Normal('mean_golfer', mu=mean_golfer_mu, sigma=mean_golfer_sd, shape=num_golfers)
    #sd_golfer = pm.HalfCauchy("sd_golfer", nu=3, sigma=2.5, shape=num_golfers)
    
    #Model Error - deviation of observed value from true value
    eps = pm.HalfCauchy('eps', beta=1)
         
    # Observed scores to par follow normal distribution
    golfer_to_par = pm.Normal("golfer_to_par", mu=mean_golfer[observed_golfers_shared], sigma=eps, observed=observed_round_score)
    #golfer_to_par = pm.Normal("golfer_to_par", mu=mean_golfer[observed_golfers_shared], sigma=sd_golfer[observed_golfers_shared], observed=observed_round_score)
    
   
#Set cores to 1
# Tuning samples will be discarded
with model:
    trace = pm.sample(1000,chains=2, tune=1000, cores=1,random_seed=1234)

# Plot model in graph format 
pm.model_to_graphviz(model)

container = az.from_pymc3(trace=trace)
summary_trace = az.summary(container)

# Create dataframe with trace
df_trace = pm.trace_to_dataframe(trace)

# Check if NUTS sampler converged
# Bayesian fraction of missing information
# Gelman rubin - evaluates difference between multiple markov chains
# Want to see value of less than 1.1 - want to have 1 ideally
# currently 1.013
bfmi = np.max(pm.stats.bfmi(trace))
max_gr = max(np.max(gr_stats) for gr_stats in pm.stats.rhat(trace).values()).values


(
    pm.energyplot(trace, legend=False, figsize=(6, 4)).set_title(
        f"BFMI = {bfmi}\nGelman-Rubin = {max_gr}"
    )
)


#Plot Posterior of specific player - Poisson gives much wider distribution than normal
# Ranges don't look wide enough - shrunk toward average maybe?
# Posterior looks different to summary info - mean is different (-0.53 vs -0.836)
# Plot Posterior for Shane Lowry
pm.plot_posterior(trace['mean_golfer'][100])

# Plot Posterior for Jon Rahm
pm.plot_posterior(trace['mean_golfer'][159])


# Sample Posterior predictive
with model:
    pp_train = pm.sample_posterior_predictive(trace,samples=1000)

#Plot Posterior Distribution
az.plot_ppc(az.from_pymc3(posterior_predictive=pp_train, model=model));
    

# Round scores to the nearest whole number
pp_train_rounded = {'golfer_to_par': pp_train['golfer_to_par'].round(0)}

transposed_rounded = pp_train_rounded['golfer_to_par'].T

# Reset index on training dataset
training_data_reset = training_data.reset_index(drop=True)

# HDI chart that works - reset training index
# Function takes hdi of each column
az.plot_hdi(training_data_reset.index,  pp_train_rounded['golfer_to_par'])

# Get Shane Lowry Values
shane_lowry = training_data_reset[training_data_reset['player']=='Shane Lowry']
shane_lowry_indexs = shane_lowry.index
#Get simulated scores
shane_lowry_rounded = pd.DataFrame(transposed_rounded[shane_lowry_indexs])
# Get actual scores
shane_lowry_actual = shane_lowry.Round_Score.reset_index(drop=True)


# Get Jon Rahm Values
jon_rahm = training_data_reset[training_data_reset['player']=='Jon Rahm']
jon_rahm_indexs = jon_rahm.index
# Get simulated scores
jon_rahm_rounded = pd.DataFrame(transposed_rounded[jon_rahm_indexs])
# Get actual scores
jon_rahm_actual = jon_rahm.Round_Score.reset_index(drop=True)



# Plot Chart for Shane Lowry
_, ax = plt.subplots()
az.plot_hdi(shane_lowry_rounded.index, shane_lowry_rounded.T,hdi_prob=0.94, fill_kwargs={"alpha": 0.8, "label": "Posterior Pred 94% HDI"})
# Get mean of each column of predicted outcomes
ax.plot(shane_lowry_rounded.index, shane_lowry_rounded.mean(axis=1), label="Mean Posterior Pred", alpha=0.8)
ax.plot(shane_lowry_rounded.index, shane_lowry_actual, "x", ms=4, alpha=0.6,color="black", label="Actual Data")
ax.set_xlabel("Observation") 
ax.set_ylabel("Round Score to Par")
ax.set_title("Shane Lowry Observations vs Posterior Predictive")
ax.legend(fontsize=10, frameon=True, framealpha=0.5,loc='upper right',bbox_to_anchor=(1, 1))


# Plot Chart for Jon Rahm
_, ax = plt.subplots()
az.plot_hdi(jon_rahm_rounded.index, jon_rahm_rounded.T,hdi_prob=0.94, fill_kwargs={"alpha": 0.8, "label": "Posterior Pred 94% HDI"})
# Get mean of each column of predicted outcomes
ax.plot(jon_rahm_rounded.index, jon_rahm_rounded.mean(axis=1), label="Mean Posterior Pred", alpha=0.8)
ax.plot(jon_rahm_rounded.index, jon_rahm_actual, "x", ms=4, alpha=0.6,color="black", label="Actual Data")
ax.set_xlabel("Observation") 
ax.set_ylabel("Round Score to Par")
ax.set_title("Jon Rahm Observations vs Posterior Predictive")
ax.legend(fontsize=10, frameon=True, framealpha=0.5,loc='upper right',bbox_to_anchor=(1, 1))


# Check to see if model can reproduce patterns observed in real data
az.plot_ppc(az.from_pymc3(posterior_predictive=pp_train_rounded, model=model));



# Set values for Model as test set
# Have 6687 values
observed_golfers_shared.set_value(observed_golfers_test)

#Output is decimal number score to par - need to add to par and then round
with model:
    pp_test_set = pm.sample_posterior_predictive(trace,samples=100)
    

# Round scores to the nearest whole number
pp_test_rounded = {'golfer_to_par': pp_test_set['golfer_to_par'].round(0)}

# Could be lower as used shared
az.plot_ppc(az.from_pymc3(posterior_predictive=pp_test_rounded, model=model));

import matplotlib as plt

golfer_0 = pp_test_rounded['golfer_to_par'][:, 0]

plt.hist(golfer_0, bins=50, color='tab:blue')


# Next step Simulate Tournaments

list_of_tournaments = test_data['tournament name'].unique()

masters_scores = test_data[test_data['tournament name']=='Masters Tournament']['Round_Score']
masters_golfers = test_data[test_data['tournament name']=='Masters Tournament']['i_golfer']

# Draw from Posterior - Can't use Test Set Posterior as outcome could change

# Only Predictor is golfer
# Every player should have at least 2 rounds

# Increase number of draws and then take an average and mean for each - 100 samples not that much
transpose = pp_test_set['golfer_to_par'].T
transpose_r = pp_test_rounded['golfer_to_par'].T

