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

#Drop column player id - Can't use player ID - get index 9621 is out of bounds error
data = data.drop(columns=['player id'])


#Split into training data with rough 80:20 split
training_data = data[data['date'] <'2020-10-01']
test_data = data[data['date'] >='2020-10-01']

# Create new column i_golfer for training data
golfers = training_data.player.unique()
golfers = pd.DataFrame(golfers, columns=["golfer"])
golfers["i"] = golfers.index

# Get golfers from test set
golfers_test = test_data.player.unique()

# Add i column back to dataframe
training_data = pd.merge(training_data, golfers, left_on="player", right_on="golfer", how="left")
training_data = training_data.rename(columns={"i": "i_golfer"}).drop("golfer", 1)

# Create new column i_course
courses = training_data.course.unique()
courses = pd.DataFrame(courses, columns=["course"])
courses["i"] = courses.index

# Add i column back to dataframe
training_data = pd.merge(training_data, courses, left_on="course", right_on="course", how="left")
training_data = training_data.rename(columns={"i": "i_course"})



# Get Round 1 entries for each tournament
test_r1 = test_data[test_data['Round']==1]


#Set values to be used as x
observed_golfers = training_data.i_golfer.values
# Set as shared so that can swap in golfers for test set
observed_golfers_shared = shared(observed_golfers)
# Get values for golfers from test set Round 1
observed_golfers_test = test_r1.i_golfer.values
# Get observed scores to use for model
observed_round_score = training_data.Round_Score.values
# Get ovserved course that could be used for model
observed_courses = training_data.i_course.values

#Get unique number of golfers - shape will be ok below
num_golfers = len(training_data.i_golfer.drop_duplicates())
num_courses = len(training_data.i_course.drop_duplicates())

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


# Set values for Model as test set
# Have 6687 values
observed_golfers_shared.set_value(observed_golfers_test)

#Output is decimal number score to par - need to add to par and then round
with model:
    pp_test_set = pm.sample_posterior_predictive(trace,samples=100)
    
# Round scores to the nearest whole number
pp_test_rounded = {'golfer_to_par': pp_test_set['golfer_to_par'].round(0)}
