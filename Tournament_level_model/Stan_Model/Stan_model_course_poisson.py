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

#Filter out golfers with less than 28 rounds played
data = data[data["Count_rounds"]>=28]

#Drop column player id - Can't use player ID - get index 9621 is out of bounds error
data = data.drop(columns=['player id'])

subset_players = ['Sungjae Im','Brian Stuard',
'Adam Schenk','Brian Harman',
'Joel Dahmen','Hideki Matsuyama',
'J.T. Poston','Chez Reavie',
'Sebastian Munoz','Billy Horschel']

# Filter out 10 golfers
data = data[data["player"].isin(subset_players)]


#Split into training data with rough 80:20 split
# Select subset of data for training diagnostics
training_data = data[data['date'] <'2020-10-01']
test_data = data[data['date'] >='2020-10-01']

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


# Get golfers from test set
golfers_test = pd.DataFrame(test_data.player.unique())
# Rename column
golfers_test = golfers_test.rename(columns={0: "golfer"})

# Get courses from test set
courses_test = pd.DataFrame(test_data.course.unique())
# Rename column
courses_test = courses_test.rename(columns={0: "course"})


# Drop golfers that are already in golfers
cond = golfers_test["golfer"].isin(golfers['golfer'])
golfers_test.drop(golfers_test[cond].index, inplace = True)

# Drop courses that are already in courses
cond = courses_test["course"].isin(courses['course'])
courses_test.drop(courses_test[cond].index, inplace = True)


# Add new golfers to golfers dataframe
golfers = golfers.append(golfers_test, ignore_index=True)
golfers["i"] = golfers.index + 1

# Add new courses to courses dataframe
courses = courses.append(courses_test, ignore_index=True)
courses["i"] = courses.index + 1


# Add i column back to dataframe
test_data = pd.merge(test_data, golfers, left_on="player", right_on="golfer", how="left")
test_data = test_data.rename(columns={"i": "i_golfer"}).drop("golfer", 1)


# Add i column back to dataframe
test_data = pd.merge(test_data, courses, left_on="course", right_on="course", how="left")
test_data = test_data.rename(columns={"i": "i_course"}).drop("course", 1)


# Get Round 1 entries for each tournament
test_r1 = test_data[test_data['Round']==1]
# Select a tournament from round 1 - Augusta
tournament_r1 = test_r1[test_r1['tournament id']==401219478]

#Set values to be used as x
observed_golfers = training_data.i_golfer.values
# Get values for golfers from tournament round 1
observed_golfers_test = tournament_r1.i_golfer.values

# Get observed scores to use for model
# Changed this to overall score rather than to par
observed_round_score = training_data.Round_total.values
# Get observed scores to use for model
observed_round_score_test = tournament_r1.Round_total.values

#Set values to be used as x
observed_courses = training_data.i_course.values
# Get values for course from tournament round 1
observed_courses_test = tournament_r1.i_course.values


#Get unique number of golfers - shape will be ok below
num_golfers = len(training_data.i_golfer.drop_duplicates())
num_golfers_test = len(tournament_r1.i_golfer.drop_duplicates())

#Get unique number of golfers - shape will be ok below
num_courses = len(training_data.i_course.drop_duplicates())
num_courses_test = len(tournament_r1.i_course.drop_duplicates())

# Adding intercept term - model did not run
partial_pooling = """
data {
  int<lower=0> N; //number of rows in training set
  int golfer[N];
  int course[N];
  int y[N];
  int K; //num golfers
  int J; //num courses
  int L; //num rows in pred set
  int golfer_pred[L];
  int course_pred[L];
  
} 
parameters {
// golfer parameters
  vector[K] a_player;
  real mu_a;
  real<lower=0.00001> sigma_a;
  
// course parameters - no mean and sigma - keep each course separate
  vector[J] b_course;
 
} 
transformed parameters {
  vector[N] y_hat;
  vector[L] y_hat_pred;
  
for (i in 1:N)
    y_hat[i] = exp(a_player[golfer[i]] + b_course[course[i]]);

for (i in 1:L)
    y_hat_pred[i] = exp(a_player[golfer_pred[i]] + b_course[course_pred[i]]);
    }

model {
  // Set priors
  mu_a ~ normal(0, 100);
  // variability of player means
  sigma_a ~ normal(0, 100);
  

  // Coefficient for each golfer
  a_player ~ normal (mu_a, sigma_a);
 
  // Likelihood
  y ~ poisson(y_hat);
}

generated quantities {
  int y_pred[L];
  for (i in 1:L)
  y_pred[i] = poisson_rng(y_hat_pred[i]);
}

"""

partial_pool_data = {'N': len(observed_round_score),
               'golfer': observed_golfers,
               'y': (observed_round_score),
               'course': observed_courses,
               'K':num_golfers,
               'J':num_courses,
               'L':len(observed_golfers_test),
               'golfer_pred':observed_golfers_test,
               'course_pred':observed_courses_test }


# Create Model - this will help with recompilation issues
stan_model = pystan.StanModel(model_code=partial_pooling)


# Call sampling function with data as argument
fit = stan_model.sampling(data=partial_pool_data, iter=2000, chains=4, seed=1,warmup=1000,control=dict(adapt_delta=0.99,max_treedepth=15))

# check divergences
pystan.check_hmc_diagnostics(fit)


# Put Posterior draws into a dictionary
params = fit.extract()

# Create summary dictionary
summary_dict = fit.summary()

# Convert to dictionary
# rhat should be < 1.1
trace_summary = pd.DataFrame(summary_dict['summary'], 
                  columns=summary_dict['summary_colnames'], 
                  index=summary_dict['summary_rownames'])


# Standard deviation for predicted scores was 9
# Mean prediction was 72 
# sqrt 72 is 8.48
# 2.5% score was 55
# 97.5% was 91
# Standard deviation is too large (-18 to +18) for 95% of scores
# Poisson Model not appropriate with current set up
# Normal distribution can be used with continuity correction



