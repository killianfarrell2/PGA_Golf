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
    
    #Global model parameters - all golfers have same sd
    #sd_global = pm.HalfStudentT("sd_global", nu=3, sigma=2.5)
    #mean_global = pm.Normal('mean_global', mu=0, sigma=3)
    #intercept = pm.Normal('intercept', mu=0, sigma=3)

    # golfer specific parameters
    mean_golfer = pm.Normal('mean_golfer', mu=0, sigma=3, shape=num_golfers)
    sd_golfer = pm.HalfStudentT("sd_golfer", nu=3, sigma=2.5, shape=num_golfers)
         
    # Observed scores to par follow normal distribution
    golfer_to_par = pm.Normal("golfer_to_par", mu=mean_golfer[observed_golfers_shared], sigma=sd_golfer[observed_golfers_shared], observed=observed_round_score)
    
   
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
pm.plot_posterior(trace['mean_golfer'][0])

# Sample Posterior predictive
with model:
    pp_train = pm.sample_posterior_predictive(trace,samples=100)
    
pp_train["golfer_to_par"].shape


# Round scores to the nearest whole number
pp_train_rounded = {'golfer_to_par': pp_train['golfer_to_par'].round(0)}


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

# hpd
_, ax = plt.subplots(figsize=(12, 6))
ax.plot(
    [observed_golfers_test, observed_golfers_test],
    az.hpd(pp_test_rounded['golfer_to_par']).T,
    lw=6,
    color="#00204C",
    alpha=0.8,
)
# actual outcomes:
ax.scatter(
    x=observed_golfers_test,
    y=test_set_round_score,
    marker="x",
    color="#A69C75",
    alpha=0.8,
    label="Observed outcomes",
)



# Simulate Tournaments

list_of_tournaments = test_data['tournament name'].unique()

masters_scores = test_data[test_data['tournament name']=='Masters Tournament']['Round_Score']
masters_golfers = test_data[test_data['tournament name']=='Masters Tournament']['i_golfer']

# Draw from Posterior - Can't use Test Set Posterior as outcome could change

# Only Predictor is golfer
# Every player should have at least 2 rounds



observed_tournament
test_data['tournament name']

# Increase number of draws and then take an average and mean for each - 100 samples not that much
transpose = pp_test_set['golfer_to_par'].T
transpose_r = pp_test_rounded['golfer_to_par'].T

