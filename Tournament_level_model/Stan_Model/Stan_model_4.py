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

# Create new column i_golfer for all data
golfers = data.player.unique()
golfers = pd.DataFrame(golfers, columns=["golfer"])
# Increase index by 1 so that stan can use it
golfers["i"] = golfers.index + 1

# Create new column i_course for all data
courses = data.course.unique()
courses = pd.DataFrame(courses, columns=["course"])
# Increase index by 1 so that stan can use it
courses["i"] = courses.index + 1


#Split into training data with rough 80:20 split
# Select subset of data for training diagnostics
training_data = data[data['date'] <'2020-10-01']
test_data = data[data['date'] >='2020-10-01']


# Add i column for golfers back to dataframe
training_data = pd.merge(training_data, golfers, left_on="player", right_on="golfer", how="left")
training_data = training_data.rename(columns={"i": "i_golfer"}).drop("golfer", 1)

# Add i column for courses back to dataframe
training_data = pd.merge(training_data, courses, left_on="course", right_on="course", how="left")
training_data = training_data.rename(columns={"i": "i_course"}).drop("course", 1)

# Add i column for golfers back to dataframe
test_data = pd.merge(test_data, golfers, left_on="player", right_on="golfer", how="left")
test_data = test_data.rename(columns={"i": "i_golfer"}).drop("golfer", 1)

# Add i column for courses back to dataframe
test_data = pd.merge(test_data, courses, left_on="course", right_on="course", how="left")
test_data = test_data.rename(columns={"i": "i_course"}).drop("course", 1)

#Set values to be used as x
observed_golfers = training_data.i_golfer.values
# Get values for golfers from tournament round 1
observed_golfers_test = test_data.i_golfer.values

# Get observed scores to use for model
observed_round_score = training_data.Round_Score.values
# Get observed scores to use for model
observed_round_score_test = test_data.Round_Score.values

#Set values to be used as x
observed_courses = training_data.i_course.values
# Get values for course from tournament round 1
observed_courses_test = test_data.i_course.values


#Get unique number of golfers - shape will be ok below
num_golfers = len(training_data.i_golfer.drop_duplicates())
num_golfers_test = len(test_data.i_golfer.drop_duplicates())

#Get unique number of golfers - shape will be ok below
num_courses = len(training_data.i_course.drop_duplicates())
num_courses_test = len(test_data.i_course.drop_duplicates())

# Course affect not hierarchical (no link between courses)
# Player affect is hierarchical

model_code = """
data {
  int<lower=0> N; //number of rows in training set
  int golfer[N];
  int K; //num golfers
  vector[N] y;
  int L; //num rows in pred set
  int golfer_pred[L];
} 
parameters {  
  
// golfer parameters
  vector[K] a_golfer;
  real mu_a;
  real<lower=0.00001> sigma_a;

// residual error in likelihood
  real<lower=0.00001> sigma_y;
} 

transformed parameters {
  vector[N] y_hat;
  vector[L] y_hat_pred;

for (i in 1:N)
    y_hat[i] = a_golfer[golfer[i]];

for (i in 1:L)
    y_hat_pred[i] = a_golfer[golfer_pred[i]];
    
}

model {
       
      // Set priors
      mu_a ~ normal(0, 1);
      // variability of golfer means
      // inferential uncertainty
      sigma_a ~ normal(0, 1);  
      
      // variability in score (without prior was 2.8) - still close with below sd = 1
      // when model has large errors could use student t instead
      // predictive uncertainty
      sigma_y ~ normal(0, 1); 
      
      // Coefficient for each golfer
      a_golfer ~ normal (mu_a, sigma_a);
       
    // Likelihood
    y ~ normal(y_hat, sigma_y);
}

generated quantities {
 // add in round function
  vector[L] y_pred;
  for (i in 1:L)
  y_pred[i] = round(normal_rng(y_hat_pred[i], sigma_y));
}


"""


model_data = {'N': len(observed_round_score),
               'golfer': observed_golfers,
               'y': observed_round_score,
               'K':num_golfers,
               'L':len(observed_golfers_test),
               'golfer_pred':observed_golfers_test }


# Create Model - this will help with recompilation issues
stan_model = pystan.StanModel(model_code=model_code)

# Call sampling function with data as argument
fit = stan_model.sampling(data=model_data, iter=2000, chains=4, seed=1,warmup=1000,control=dict(adapt_delta=0.99,max_treedepth=10))

# Put Posterior draws into a dictionary
trace = fit.extract()

# Put predictions into a dataframe
y_pred = trace['y_pred']

# Create summary dictionary
summary_dict = fit.summary()

# Change row names to golfers and course

# create empty array
row_names = np.empty(0)

# Set row names
orig_row_names = summary_dict['summary_rownames']

for string in orig_row_names:
    if "a_golfer[" in string:    
        number = int(string.replace("a_golfer[", '').replace(']', ''))
        value = golfers[golfers.i.isin([number])]['golfer']
    else:
        value = string
    row_names = np.append(row_names,value)

# Convert to dictionary
# rhat should be < 1.1
trace_summary = pd.DataFrame(summary_dict['summary'], 
                  columns=summary_dict['summary_colnames'], 
                  index=row_names)


# Create dataframe of summarised predictions

pred_rows = [row for row in trace_summary.index if 'y_pred[' in row]
predictions = trace_summary.loc[pred_rows , : ]



# How many times does true value fall within 95% (2.5-97.5) interval

store_results = pd.DataFrame()

i=0
for actual in observed_round_score_test:
    simulated = y_pred[:,i]
    lower = predictions['2.5%'][i].astype(int)
    upper = predictions['97.5%'][i].astype(int)
    in_range = [(actual in range(lower,upper))]
    df = pd.DataFrame(in_range)
    store_results = store_results.append(df)
    i = i+1



# Create dataframe with results
# 389 true
# 17 false
# 17/406 = 4.1%
# Actual falls within 96% of predictions

store_results.value_counts()


store_results_2 = pd.DataFrame()

i=0
for actual in observed_round_score_test:
    simulated = y_pred[:,i]
    lower = predictions['25%'][i].astype(int)
    upper = predictions['75%'][i].astype(int)
    in_range = [(actual in range(lower,upper))]
    df = pd.DataFrame(in_range)
    store_results_2 = store_results_2.append(df)
    i = i+1


# Get results for 50% Range
# 210 misses range
# 196 within range
# Falls within 50% range 48% of the time
store_results_2.value_counts()




