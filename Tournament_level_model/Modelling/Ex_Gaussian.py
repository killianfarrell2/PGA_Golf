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
    sd_golfer = pm.HalfNormal('sd_golfer',sigma=3,shape=num_golfers)     
    #Exponential parameter - 1 for each golfer
    #nu_golfer = pm.Gamma('nu_golfer',alpha=1, beta=1,shape=num_golfers)
    nu_golfer = pm.Uniform('nu_golfer',lower=0,upper=10)
    # Mean = nu + mu (this should be larger than mode of -1)
    golfer_to_par = pm.ExGaussian("golfer_to_par", nu=nu_golfer, mu=mean_golfer[observed_golfers_shared], sigma=sd_golfer[observed_golfers_shared], observed=observed_round_score)
    # Prior Predictive checks - generate samples without taking data
    prior_checks = pm.sample_prior_predictive(samples=1000, random_seed=1234)

# Round scores to the nearest whole number
prior_check_rounded = {'golfer_to_par': prior_checks['golfer_to_par'].round(0)}
#Put scores into dataframe
prior_scores_rounded = pd.DataFrame(prior_check_rounded['golfer_to_par'].T)
# Transpose scores
t_prior_scores_rounded = pd.DataFrame(prior_scores_rounded.T)
# Get actual scores
actual_scores = training_data.Round_Score.reset_index(drop=True)

# Get group by scores
actual_scores_grp = pd.DataFrame(actual_scores.value_counts())
actual_scores_grp['pct'] = actual_scores_grp['Round_Score'] / actual_scores_grp['Round_Score'].sum()

# Add in percentage column for each value
value_counts_pr = t_prior_scores_rounded.apply(pd.Series.value_counts)
value_counts_rows_pr = pd.DataFrame(value_counts_pr.sum(axis=1))
value_counts_rows_pr['pct'] = value_counts_rows_pr[0] / value_counts_rows_pr[0].sum()

# Plot Bar chart (categories) rather than histogram (continous)
# Plot actual  vs Simulated values
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
scores_sim = value_counts_rows_pr.index
percentage_sim = value_counts_rows_pr['pct']
scores_a = actual_scores_grp.index
percentage_a = actual_scores_grp['pct']
ax.bar(scores_a,percentage_a,color='dodgerblue',label='Actual')
ax.bar(scores_sim,percentage_sim,color='none',edgecolor='r', label='Sim')
ax.set_xlabel("Score to Par") 
ax.set_ylabel("Density")
ax.set_title("Actual vs Prior Predictive")
plt.legend()
plt.show()

#Set cores to 1
# Tuning samples will be discarded
# Getting divergences when there is 1 skew parameter for each golfer
with model:
    trace = pm.sample(1000,chains=2, tune=1000, cores=1,random_seed=12345)

container = az.from_pymc3(trace=trace)
summary_trace = az.summary(container)

# Plot model in graph format 
pm.model_to_graphviz(model)

# Create dataframe with trace
df_trace = pm.trace_to_dataframe(trace)

# Check if NUTS sampler converged
# Bayesian fraction of missing information
bfmi = np.max(pm.stats.bfmi(trace))
max_gr = max(np.max(gr_stats) for gr_stats in pm.stats.rhat(trace).values()).values

# Energy plot
pm.energyplot(trace, legend=False, figsize=(6, 4)).set_title(f"BFMI = {bfmi}\nGelman-Rubin = {max_gr}")

# Sample Posterior predictive
with model:
    pp_train = pm.sample_posterior_predictive(trace,samples=1000)
    
# Round scores to the nearest whole number
pp_train_rounded = {'golfer_to_par': pp_train['golfer_to_par'].round(0)}
transposed_rounded = pp_train_rounded['golfer_to_par'].T
scores_rounded = pd.DataFrame(transposed_rounded)

value_counts = scores_rounded.apply(pd.Series.value_counts)
value_counts_rows = pd.DataFrame(value_counts.sum(axis=1))
value_counts_rows['pct'] = value_counts_rows[0] / value_counts_rows[0].sum()


# Plot actual  vs Simulated values
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
scores_sim = value_counts_rows.index
percentage_sim = value_counts_rows['pct']
scores_a = actual_scores_grp.index
percentage_a = actual_scores_grp['pct']
ax.bar(scores_a,percentage_a,color='dodgerblue',label='Actual')
ax.bar(scores_sim,percentage_sim,color='none',edgecolor='r', label='Sim')
ax.set_xlabel("Score to Par") 
ax.set_ylabel("Density")
ax.set_title("Actual vs Posterior Predictive")
plt.legend()
plt.show()


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
t_test_scores_df = pd.DataFrame(test_scores_df.T)

value_counts_test = t_test_scores_df.apply(pd.Series.value_counts)
value_counts_rows_test = pd.DataFrame(value_counts_test.sum(axis=1))
value_counts_rows_test['pct'] = value_counts_rows_test[0] / value_counts_rows_test[0].sum()

# Get actual scores
actual_test = test_data.Round_Score.reset_index(drop=True)
# Get group by scores
actual_test_grp = pd.DataFrame(actual_test.value_counts())
actual_test_grp['pct'] = actual_test_grp['Round_Score'] / actual_test_grp['Round_Score'].sum()


# Plot test  vs Simulated values
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
scores_sim = value_counts_rows_test.index
percentage_sim = value_counts_rows_test['pct']
scores_a = actual_test_grp.index
percentage_a = actual_test_grp['pct']
ax.bar(scores_a,percentage_a,color='dodgerblue',label='Actual')
ax.bar(scores_sim,percentage_sim,color='none',edgecolor='r', label='Sim')
ax.set_xlabel("Score to Par") 
ax.set_ylabel("Density")
ax.set_title("Test vs Posterior Predictive")
plt.legend()
plt.show()


