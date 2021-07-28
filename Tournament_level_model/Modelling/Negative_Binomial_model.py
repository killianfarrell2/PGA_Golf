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
#Round score to par
observed_round_score = training_data.Round_Score.values
test_set_round_score = test_data.Round_Score.values

#Total Round Score
observed_total_score = training_data.Round_total.values
observed_total_score_test = test_data.Round_total.values

#get observed courses
observed_courses = training_data.i_course.values

# Observed Par
observed_par = training_data.par.values

#Get unique number of golfers - shape will be ok below
num_golfers = len(training_data.i_golfer.drop_duplicates())
num_courses = len(training_data.i_course.drop_duplicates())


with pm.Model() as model:
    
    # GLobal parameters
    sd_golfer = pm.HalfStudentT("sd_golfer", nu=3, sigma=2.5)
    
    # golfer specific parameters
    mean_golfer = pm.Normal('mean_golfer', mu=0, sigma=sd_golfer, shape=num_golfers)
    
    # Get theta to be used for Poisson Distribution
    golfer_theta = observed_par + mean_golfer[observed_golfers_shared]
         
    # Observed scores to follow Negative Binomial distribution
    golfer_round_score = pm.Poisson("golfer_round_score", mu=golfer_theta, observed=observed_total_score)
    
   
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
# currently 1.02 - but graph doesn't look great
#BFMI very low
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


with model:
    pp_train = pm.sample_posterior_predictive(trace,samples=100)


# Check to see if model can reproduce patterns observed in real data
# because distribution is discrete doesn't fit data well in graph
az.plot_ppc(az.from_pymc3(posterior_predictive=pp_train, model=model));



# Set values for Model as test set
# Have 6687 values
observed_golfers_shared.set_value(observed_golfers_test)

#Output is decimal number score to par - need to add to par and then round
with model:
    pp_test_set = pm.sample_posterior_predictive(trace,samples=100)

#Create Dataframe from pp_test_set   
normal_scores = pp_test_set['golfer_to_par']

# Round scores to the nearest whole number
round_scores = normal_scores.round(0)
#Each column is a row in data

# Could be lower as used shared
az.plot_ppc(az.from_pymc3(posterior_predictive=pp_test_set, model=model));




