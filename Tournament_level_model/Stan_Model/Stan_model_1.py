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

#Split into training data with rough 80:20 split
training_data = data[data['date'] <'2020-10-01']
test_data = data[data['date'] >='2020-10-01']

# Create new column i_golfer for training data
golfers = training_data.player.unique()
golfers = pd.DataFrame(golfers, columns=["golfer"])
# Increase index by 1 so that stan can use it
golfers["i"] = golfers.index + 1

# Add i column back to dataframe
training_data = pd.merge(training_data, golfers, left_on="player", right_on="golfer", how="left")
training_data = training_data.rename(columns={"i": "i_golfer"}).drop("golfer", 1)

# Get golfers from test set
golfers_test = pd.DataFrame(test_data.player.unique())
# Rename column
golfers_test = golfers_test.rename(columns={0: "golfer"})

# Drop golfers that are already in golfers
cond = golfers_test["golfer"].isin(golfers['golfer'])
golfers_test.drop(golfers_test[cond].index, inplace = True)


# Add new golfers to golfers dataframe
golfers = golfers.append(golfers_test, ignore_index=True)
golfers["i"] = golfers.index + 1

# Add i column back to dataframe
test_data = pd.merge(test_data, golfers, left_on="player", right_on="golfer", how="left")
test_data = test_data.rename(columns={"i": "i_golfer"}).drop("golfer", 1)


# Get Round 1 entries for each tournament
test_r1 = test_data[test_data['Round']==1]
# Select a tournament from round 1 - Augusta
tournament_r1 = test_r1[test_r1['tournament id']==401219478]

#Set values to be used as x
observed_golfers = training_data.i_golfer.values
# Get values for golfers from tournament round 1
observed_golfers_test = tournament_r1.i_golfer.values

# Get observed scores to use for model
observed_round_score = training_data.Round_Score.values

#Get unique number of golfers - shape will be ok below
num_golfers = len(training_data.i_golfer.drop_duplicates())


# Put data in dictionary format for stan
my_data = {'N':len(observed_round_score),'n_golfers':num_golfers,'y':observed_round_score,'X':observed_golfers}

my_code = """
data {
      int N; // number of data points
      int<lower=0> n_golfers; //number of golfers
      real y[N];// data values for round scores (has to be real instead of int)
      int<lower=0> X[N]; //golfer codes data values
}

parameters {
    
        //hyper parameters
        real mu_golfer;
        real <lower=0> sd_golfer;
        real model_error;
        
        // Parameters in Likelihood
        real intercept; // intercept term in model (base round score)
        vector[n_golfers] golfer; //Coefficient for mean of each golfer
       
}

transformed parameters {
  vector[N] model_mean; //mean when taking into account base score

  model_mean = intercept + golfer[X];


}

model {
       //hyper priors - set a wide group
       mu_golfer ~ normal(0,10);
       sd_golfer ~ normal(0,10);
       intercept ~ normal(0,10);
       model_error  ~ normal(0,10);

       //priors
       golfer ~ normal(mu_golfer, sd_golfer);
       
       //likelihood
       y ~ normal(model_mean, model_error);
     
}

"""



#generated quantities {
#    vector [n_golfers] round_score;
#    for (n in 1:n_golfers) 
#    score[n] = normal_Ipdf(intercept + occ[n],);      
#}


# Create Model - this will help with recompilation issues
stan_model = pystan.StanModel(model_code=my_code)

# Call sampling function with data as argument
fit = stan_model.sampling(data=my_data, iter=2000, chains=4, seed=1,warmup=1000)

# Put Posterior draws into a dictionary
params = fit.extract()


# Get summary statistics for parameters
print(fit)


detailed_summary = fit.summary()
