import pandas as pd
import numpy as np
import pystan
import arviz as az
import matplotlib.pyplot as plt

#Import Tournament data
data_location = 'C:\\KF_Repo\\PGA_Golf\\Tournament_level_model\\Data_manipulation\\model_data.csv'
data = pd.read_csv(data_location)

# Get count of rounds
g_count = pd.DataFrame(data.groupby("player")['hole_par'].count())
g_count = g_count.rename(columns={"hole_par": "Count_rounds"})

# Join count column
data = pd.merge(data, g_count, left_on="player", right_index=True, how="left")


#Drop column player id - Can't use player ID - get index 9621 is out of bounds error
data = data.drop(columns=['player id'])

subset_players = ['Hideki Matsuyama',]

# Filter out 10 golfers
data = data[data["player"].isin(subset_players)]

# Set training data
training_data = data


# Create new column i_golfer for training data
golfers = training_data.player.unique()
golfers = pd.DataFrame(golfers, columns=["golfer"])
# Increase index by 1 so that stan can use it
golfers["i"] = golfers.index + 1

# Create new column i_course for training data
courses = training_data.course.unique()
courses = pd.DataFrame(courses, columns=["course"])
# Increase index by 1 so that stan can use it
courses["i"] = courses.index + 1


# Add i column for golfers back to dataframe
training_data = pd.merge(training_data, golfers, left_on="player", right_on="golfer", how="left")
training_data = training_data.rename(columns={"i": "i_golfer"}).drop("golfer", 1)

# Add i column for courses back to dataframe
training_data = pd.merge(training_data, courses, left_on="course", right_on="course", how="left")
training_data = training_data.rename(columns={"i": "i_course"}).drop("course", 1)



# Select a tournament from round 1 - Augusta
#tournament_r1 = test_r1[test_r1['tournament id']==401219478]

#Set values to be used as x
observed_golfers = training_data.i_golfer.values

# Get observed scores to use for model
# Change to total score
observed_round_score = training_data.Round_total.values

#Set values to be used as x
observed_courses = training_data.i_course.values


#Get unique number of golfers - shape will be ok below
num_golfers = len(training_data.i_golfer.drop_duplicates())

#Get unique number of golfers - shape will be ok below
num_courses = len(training_data.i_course.drop_duplicates())


model_code = """
data {
  int<lower=0> N; 
  int golfer[N];
  vector[N] y;
  int K;
} 
parameters {
  vector[K] a;
  real mu_a;
  real<lower=0.00001> sigma_a;
  real<lower=0.00001> sigma_y;
} 
transformed parameters {
  vector[N] y_hat;
  for (i in 1:N)
    y_hat[i] = a[golfer[i]];
}
model {
  // Set priors
  mu_a ~ normal(0, 10);
  // variability of golfer means
  sigma_a ~ normal(0, 10);
  // residual error of observations
  sigma_y ~ normal(0, 10);
  
  // Coefficient for each golfer
  a ~ normal (mu_a, sigma_a);
  
  // Likelihood
  y ~ normal(y_hat, sigma_y);
  
}"""


model_data = {'N': len(observed_round_score),
               'golfer': observed_golfers,
               'y': observed_round_score,
               'K':num_golfers}


# Create Model - this will help with recompilation issues
stan_model = pystan.StanModel(model_code=model_code)

# Call sampling function with data as argument
fit = stan_model.sampling(data=model_data, iter=2000, chains=4, seed=1,warmup=2000,control=dict(adapt_delta=0.99,max_treedepth=10))

# check divergences
pystan.check_hmc_diagnostics(fit)

# Put Posterior draws into a dictionary
params = fit.extract()

# Put predictions into a dataframe
y_pred = params['y_pred']

# Create summary dictionary
summary_dict = fit.summary()

# Convert to dictionary
# rhat should be < 1.1
trace_summary = pd.DataFrame(summary_dict['summary'], 
                  columns=summary_dict['summary_colnames'], 
                  index=summary_dict['summary_rownames'])


# Mean is 72 and std dev is 3 for prediction


# How many times does true value fall within 95% (2.5-97.5) interval


i=0
for actual in observed_round_score_test:
    simulated = y_pred[:,i]
    count = (simulated == actual).sum()
    print(count)
    i = i+1


# Should mean prediction be compared to actual?






