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


subset_players = ['Sungjae Im','Brian Stuard',
'Adam Schenk','Brian Harman',
'Joel Dahmen','Hideki Matsuyama',
'J.T. Poston','Chez Reavie',
'Sebastian Munoz','Billy Horschel']

# Filter out 10 golfers
data = data[data["player"].isin(subset_players)]


#Split into training data with rough 80:20 split
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


#Set values to be used as x
observed_golfers = training_data.i_golfer.values

# Get observed scores to use for model
observed_round_score = training_data.Round_Score.values

#Set values to be used as x
observed_courses = training_data.i_course.values

#Get unique number of golfers - shape will be ok below
num_golfers = len(training_data.i_golfer.drop_duplicates())

#Get unique number of golfers - shape will be ok below
num_courses = len(training_data.i_course.drop_duplicates())

num_rows_pred = 


partial_pooling = """
data {
  //N number of rows
  int<lower=0> N; 
  int golfer[N];
  int course[N];
  vector[N] y;
  // number of golfers
  int K;
  // number of courses
  int L;
  // number of rows in predictor set
  int M;
  int golfer_pred[M];
  int course_pred[M];
} 
parameters {

  vector[K] a;
  vector[L] b;
  real mu_a;
  // real mu_b;
  real<lower=0.00001> sigma_a;
  // real<lower=0.00001> sigma_b;
  real<lower=0.00001> sigma_y;
} 
transformed parameters {
  vector[N] y_hat;
  for (i in 1:N)
    y_hat[i] = a[golfer[i]] + b[course[i]];

  for (i in 1:M)
    y_hat_pred[i] = a[golfer_pred[i]] + b[course_pred[i]];
}


model {
  // Set priors - with no priors a doesn't make sense (=5)
  // Mean for all golfers
  mu_a ~ normal(0, 3);
  // variance among golfers
  sigma_a ~ normal(0, 3);
  // residual error in observations
  sigma_y ~ normal(0, 3);
  
  // Coefficient for each golfer - centred around group mean mu a
  a ~ normal (mu_a, sigma_a);
  
  // Set a prior for b 
  b ~ normal (0, 3);
  
  // Likelihood
  y ~ normal(y_hat, sigma_y);
  
}

generated quantities {
 // add in round function
  vector[M] y_pred;
  for (i in 1:M)
  y_pred[i] = round(normal_rng(y_hat_pred[i], sigma_y));
}



"""

partial_pool_data = {'N': len(observed_round_score),
               'golfer': observed_golfers,
               'course': observed_courses,
               'y': observed_round_score,
               'K':num_golfers,
               'L':num_courses,
               'M': len(observed_golfers_test),
               'golfer_pred':observed_golfers_test,
               'course_pred':observed_courses_test }



# Create Model - this will help with recompilation issues
stan_model = pystan.StanModel(model_code=partial_pooling)

# Call sampling function with data as argument
# Depending on seed - b (course affect) will take different value
fit = stan_model.sampling(data=partial_pool_data, iter=2000, chains=4, seed=100,warmup=1000)

# Put Posterior draws into a dictionary
trace = fit.extract()

# Create summary dictionary
summary_dict = fit.summary()


# Change row names to golfers and course

# create empty array
row_names = np.empty(0)

# Set row names
orig_row_names = summary_dict['summary_rownames']

for string in orig_row_names:
    if "a[" in string:    
        number = int(string.replace("a[", '').replace(']', ''))
        value = golfers[golfers.i.isin([number])]['golfer']
    
    elif "b[" in string:    
        number = int(string.replace("b[", '').replace(']', ''))
        value = courses[courses.i.isin([number])]['course']
    else:
        value = string
    row_names = np.append(row_names,value)



# Convert to dictionary
trace_summary = pd.DataFrame(summary_dict['summary'], 
                  columns=summary_dict['summary_colnames'], 
                  index=row_names)

