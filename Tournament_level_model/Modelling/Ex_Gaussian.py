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

#Ex Gaussian Distribution
#nu=8,mu=-6,sd=4
# nu=1/lambda (lambda is rate and nu is mean)
# nu is time between events and lambda is rate of events
# 1/nu related to alpha and beta from gamma distribution
# Gmma(Shape param=a+n, Rate param=Beta+T)
# Gamma Distribution - wait time until future events
# Exponential predicts time to first event
# Gamma is wait time until kth event (alpha)
# Increasing alpha increase time, increasing beta decreases

with pm.Model() as model:    
    #Hyper Priors - need to use conjugate priors
    #mean_golfer_sd  = pm.HalfNormal('mean_golfer_sd',sigma=2)
    #mean_golfer_mu = pm.Normal('mean_golfer_mu', mu=-6, sigma=1)
    # golfer specific parameters
    mean_golfer = pm.Normal('mean_golfer', mu=-6, sigma=3, shape=num_golfers)
    #standard deviation for each golfer - Inverse gamma is prior for standard deviation
    sd_golfer = pm.HalfNormal('sd_golfer',sigma=3)     
    #Exponential parameter
    #nu_golfer = pm.Gamma('nu_golfer',alpha=1, beta=1,shape=num_golfers)
    nu_golfer = pm.Uniform('nu_golfer',lower=0,upper=10,shape=num_golfers)
    # Mean = nu + mu (this should be larger than mode of -1)
    golfer_to_par = pm.ExGaussian("golfer_to_par", nu=nu_golfer[observed_golfers_shared], mu=mean_golfer[observed_golfers_shared], sigma=sd_golfer, observed=observed_round_score)
    # Prior Predictive checks - generate samples without taking data
    prior_checks = pm.sample_prior_predictive(samples=1000, random_seed=1234)

# Round scores to the nearest whole number
prior_check_rounded = {'golfer_to_par': prior_checks['golfer_to_par'].round(0)}
#Put scores into dataframe
prior_scores_rounded = pd.DataFrame(prior_check_rounded['golfer_to_par'].T)
# Transpose scores
t_prior_scores_rounded = prior_scores_rounded.T
# Get actual scores
actual_scores = training_data.Round_Score.reset_index(drop=True)

# Get index for prior scores rounded
prior_indices = prior_scores_rounded.index

# Get mean of prior scores rounded
prior_mean = prior_scores_rounded.mean(axis=1)

# Plot Chart for all golfers for Prior Predictive
_, ax = plt.subplots()
az.plot_hdi(prior_indices , t_prior_scores_rounded, hdi_prob=0.94, fill_kwargs={"alpha": 0.8, "label": "Prior Pred 94% HDI"})
# Get mean of each column of predicted outcomes
ax.plot(prior_indices , prior_mean, label="Mean Prior Pred", alpha=0.8)
ax.plot(prior_indices , actual_scores, "x", ms=4, alpha=0.6,color="black", label="Actual Data")
ax.set_xlabel("Observation") 
ax.set_ylabel("Round Score to Par")
ax.set_title("Observations vs Prior Predictive")
ax.legend(fontsize=10, frameon=True, framealpha=0.5,loc='upper right',bbox_to_anchor=(1, 1))

# Plot Chart with Score to Par for all golfers as histogram with probabilities of each score
# Probability on y-axis with score to par on x-axis

#Set cores to 1
# Tuning samples will be discarded
with model:
    trace = pm.sample(1000,chains=2, tune=1000, cores=1,random_seed=1234)

container = az.from_pymc3(trace=trace)
summary_trace = az.summary(container)

# Plot model in graph format 
pm.model_to_graphviz(model)

# Create dataframe with trace
df_trace = pm.trace_to_dataframe(trace)

# Check if NUTS sampler converged
# Bayesian fraction of missing information
# Gelman rubin - evaluates difference between multiple markov chains
# Want to see value of less than 1.1 - want to have 1 ideally
# BFMI 0.96
# GR: 1.012
bfmi = np.max(pm.stats.bfmi(trace))
max_gr = max(np.max(gr_stats) for gr_stats in pm.stats.rhat(trace).values()).values

# Energy plot
pm.energyplot(trace, legend=False, figsize=(6, 4)).set_title(f"BFMI = {bfmi}\nGelman-Rubin = {max_gr}")

# Sample Posterior predictive
with model:
    pp_train = pm.sample_posterior_predictive(trace,samples=1000)

#Plot Posterior Distribution
az.plot_ppc(az.from_pymc3(posterior_predictive=pp_train, model=model));
    
# Round scores to the nearest whole number
pp_train_rounded = {'golfer_to_par': pp_train['golfer_to_par'].round(0)}

transposed_rounded = pp_train_rounded['golfer_to_par'].T
scores_rounded = pd.DataFrame(transposed_rounded)

# Reset index on training dataset
training_data_reset = training_data.reset_index(drop=True)

# HDI chart that works - reset training index
# Function takes hdi of each column
az.plot_hdi(training_data_reset.index,  pp_train_rounded['golfer_to_par'])

actual_scores = training_data_reset.Round_Score.reset_index(drop=True)

scores_hdi = az.hdi(pp_train_rounded['golfer_to_par'],hdi_prob=0.94)
scores_hdi_99999 = az.hdi(pp_train_rounded['golfer_to_par'],hdi_prob=0.99999)

# Calculate percentage of golfers obs outside 94%
hdi_actual = pd.merge(pd.DataFrame(actual_scores),pd.DataFrame(scores_hdi),left_index=True,right_index=True)
hdi_actual.loc[(hdi_actual['Round_Score'] >= hdi_actual[0]) & (hdi_actual['Round_Score'] <= hdi_actual[1]) , 'Check_between'] = 1
hdi_actual.loc[(hdi_actual['Round_Score'] < hdi_actual[0]) | (hdi_actual['Round_Score'] > hdi_actual[1]) , 'Check_between'] = 0
print('obs in 94%: ',hdi_actual.Check_between.sum()/hdi_actual.Check_between.count())

# Calculate percentage of golfers obs outside 99.999%
# 10 obs outside with each golfer having nu and golfer having sd
hdi_actual_99999 = pd.merge(pd.DataFrame(actual_scores),pd.DataFrame(scores_hdi_99999),left_index=True,right_index=True)
hdi_actual_99999.loc[(hdi_actual_99999['Round_Score'] >= hdi_actual_99999[0]) & (hdi_actual_99999['Round_Score'] <= hdi_actual_99999[1]) , 'Check_between'] = 1
hdi_actual_99999.loc[(hdi_actual_99999['Round_Score'] < hdi_actual_99999[0]) | (hdi_actual_99999['Round_Score'] > hdi_actual_99999[1]) , 'Check_between'] = 0
print('count obs outside',(hdi_actual_99999.Check_between.count() - hdi_actual_99999.Check_between.sum()))
print('obs in 99.999%: ',hdi_actual_99999.Check_between.sum()/hdi_actual_99999.Check_between.count())

# Plot Chart for all golfers
_, ax = plt.subplots()
az.plot_hdi(scores_rounded.index, scores_rounded.T, hdi_prob=0.99999, fill_kwargs={"alpha": 0.8, "label": "Posterior Pred 99.999% HDI"})
# Get mean of each column of predicted outcomes
ax.plot(scores_rounded.index, scores_rounded.mean(axis=1), label="Mean Posterior Pred", alpha=0.8)
ax.plot(scores_rounded.index, actual_scores, "x", ms=4, alpha=0.6,color="black", label="Actual Data")
ax.set_xlabel("Observation") 
ax.set_ylabel("Round Score to Par")
ax.set_title("Observations vs Posterior Predictive")
ax.legend(fontsize=10, frameon=True, framealpha=0.5,loc='upper right',bbox_to_anchor=(1, 1))


# Set input for model as golfers from test set
observed_golfers_shared.set_value(observed_golfers_test)

#Generate 1000 samples for each observation
with model:
    pp_test_set = pm.sample_posterior_predictive(trace,samples=1000)
    
# Round scores to the nearest whole number
pp_test_rounded = {'golfer_to_par': pp_test_set['golfer_to_par'].round(0)}


#Put scores into dataframe
test_scores_df = pd.DataFrame(pp_test_rounded['golfer_to_par'].T)
# Transpose scores
t_test_scores_df = test_scores_df.T
# Get actual scores
actual_test = test_data.Round_Score.reset_index(drop=True)

# Get index for prior scores rounded
test_indices = test_scores_df.index

# Get mean of prior scores rounded
test_mean = test_scores_df.mean(axis=1)

# Plot Chart for all golfers for Test set
_, ax = plt.subplots()
az.plot_hdi(test_indices , t_test_scores_df, hdi_prob=0.94, fill_kwargs={"alpha": 0.8, "label": "Test set Pred 94% HDI"})
# Get mean of each column of predicted outcomes
ax.plot(test_indices , test_mean, label="Mean Test Pred", alpha=0.8)
ax.plot(test_indices , actual_test, "x", ms=4, alpha=0.6,color="black", label="Actual Data")
ax.set_xlabel("Observation") 
ax.set_ylabel("Round Score to Par")
ax.set_title("Observations vs Test set Prediction")
ax.legend(fontsize=10, frameon=True, framealpha=0.5,loc='upper right',bbox_to_anchor=(1, 1))







