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

#Set date to datetime
data['date'] = pd.to_datetime(data['date'])

# Add in Date of Round
data['round_date'] = data['date'] - pd.to_timedelta(4 - data['Round'], unit='d')

# Get average score for each tournament round
avg_score = pd.DataFrame(data.groupby(["tournament id",'round_date'])['Round_Score'].mean())

# Set index as columns
avg_score.reset_index(level=0, inplace=True)
avg_score.reset_index(level=0, inplace=True)
# Rename column for avg score
avg_score = avg_score.rename(columns={"Round_Score": "Avg_Score"})

# Add average score
data = pd.merge(data, avg_score, left_on=["tournament id",'round_date'], right_on=["tournament id",'round_date'], how="left")
# Get performance vs avg of field
data['Stg_V_Avg'] =  data["Avg_Score"] - data["Round_Score"]



# Create subset of players
subset_players = ['Austin Cook']

# Filter out golfer
# Comment out below to keep data as is
new_data = data[data["player"].isin(subset_players)]

# Create subset of tournament
subset_tourn = 	[401243004]

# Filter out for tournament
new_data_2 = data[data["tournament id"].isin(subset_tourn)]



# Get group of data

grouped_data = pd.DataFrame(data.groupby('Round_Score').size())
# Add percentage column
grouped_data['pct'] = grouped_data[0] / grouped_data[0].sum()

# If mean is -2, sd = 3, bad round +2 or worse, good round -6 or worse

# Get probabilities for more than 1 standard dev from mean
# Bad round 14%, good round 8% of rounds
bad_round = grouped_data[grouped_data.index>=5]['pct'].sum()
print(bad_round)
good_round = grouped_data[grouped_data.index<=-8]['pct'].sum()
print(good_round)

# Plot Bar Chart - Histogram is misleading with bins

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.bar(grouped_data.index,grouped_data[0])
plt.show()

# Get mean and standard deviation of data
mean_score = data.Round_Score.mean()
print(mean_score)
sd_score = data.Round_Score.std()
print(sd_score)


# Create regression model with no predictors
import statsmodels.api as sm

import statsmodels.stats.api as sms
# Set outcome as Round Score
y = data.Round_Score

# Set predictors as a constant term
X = np.ones(len(y))

# Create Model
reg = sm.OLS(y, X)

# Get results
res = reg.fit()

# Print Summary
print(res.summary())
# Intercept term is the same as the mean
# Skew of 0.331

# Extract stats
name = ["Jarque-Bera", "Chi^2 two-tail prob.", "Skew", "Kurtosis"]
test_results = sms.jarque_bera(res.resid)
stats_skew = test_results[2]



# Get coefficient for mean of Normal Distribution
mu = res.params

# Get residuals (Difference between actual and predicted)
# All have the same prediction which is mean (-1.5)
residuals = res.resid


# Square root of scale is standard error of regression
# Also called residual standard deviation/ residual standard error
# This is the average distance points fall from regression line
res_se = res.scale**.5

# Another way of calculating it
# Sum up all residuals squared - divide by df and square root
res_se_2 = np.sqrt(np.sum(res.resid**2)/res.df_resid)


# Create fake data for each observation
# Round each observation
fake_data = pd.DataFrame(np.round(np.random.normal(mu, res_se, len(y)),0))

# Get group of fake
grouped_data_fake = pd.DataFrame(fake_data.groupby(0).size())


# Plot Bar Chart of fake data

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.bar(grouped_data_fake.index,grouped_data_fake[0])
plt.show()


# Reset rounds
num_sims = 1000
bad_round_array = np.empty(0, dtype=object)
good_round_array = np.empty(0, dtype=object)


# Run Simulation n times
for n in range(num_sims):
    fake_data = pd.DataFrame(np.round(np.random.normal(mu, res_se, len(y)),0))

    # Get group of fake
    grouped_data_fake = pd.DataFrame(fake_data.groupby(0).size())
    grouped_data_fake['pct'] = grouped_data_fake[0] / grouped_data_fake[0].sum()
    
    # Calculate percentage of good and bad round
    bad_round_fake = grouped_data_fake[grouped_data_fake.index>=5]['pct'].sum()
    good_round_fake = grouped_data_fake[grouped_data_fake.index<=-8]['pct'].sum()
    
    # Append to array
    bad_round_array = np.append(bad_round_array,bad_round_fake)
    good_round_array = np.append(good_round_array,good_round_fake)
    
# Print actual bad round percentage and intervals   
print('bad round lower:' ,np.percentile(bad_round_array, 2.5))
print('bad round upper:' ,np.percentile(bad_round_array, 97.5))
print('bad round actual:',bad_round)

# Print actual good round percentage and intervals
print('good round lower:' ,np.percentile(good_round_array, 2.5))
print('good round upper:' ,np.percentile(good_round_array, 97.5))
print('good round actual:',good_round)



# Bad scores for all golfers are higher than confidence interval
# Good scores happen less often than in normal distribution


from scipy.stats import skewnorm
# Generate Skewed Normal Distribution
    

# Create fake data for each observation
# Round each observation
fake_data_skew = pd.DataFrame(np.round( skewnorm.rvs(a=stats_skew,loc=mu,scale=res_se,size=len(y)) ,0))

# Get group of fake
grouped_data_fake_skew = pd.DataFrame(fake_data_skew.groupby(0).size())


# Plot Bar Chart of fake data

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.bar(grouped_data_fake.index,grouped_data_fake[0])
plt.show()



# Check good and bad round for skewed data

# Reset rounds
num_sims = 1000
bad_round_array = np.empty(0, dtype=object)
good_round_array = np.empty(0, dtype=object)

set_skew = 0.15

# Run Simulation n times
for n in range(num_sims):
    fake_data = pd.DataFrame(np.round(skewnorm.rvs(a=set_skew,loc=mu,scale=res_se,size=len(y)) ,0))

    # Get group of fake
    grouped_data_fake = pd.DataFrame(fake_data.groupby(0).size())
    grouped_data_fake['pct'] = grouped_data_fake[0] / grouped_data_fake[0].sum()
    
    # Calculate percentage of good and bad round
    bad_round_fake = grouped_data_fake[grouped_data_fake.index>=5]['pct'].sum()
    good_round_fake = grouped_data_fake[grouped_data_fake.index<=-8]['pct'].sum()
    
    # Append to array
    bad_round_array = np.append(bad_round_array,bad_round_fake)
    good_round_array = np.append(good_round_array,good_round_fake)
    
# Print actual bad round percentage and intervals   
print('bad round lower:' ,np.percentile(bad_round_array, 2.5))
print('bad round upper:' ,np.percentile(bad_round_array, 97.5))
print('bad round actual:',bad_round)

# Print actual good round percentage and intervals
print('good round lower:' ,np.percentile(good_round_array, 2.5))
print('good round upper:' ,np.percentile(good_round_array, 97.5))
print('good round actual:',good_round)


# Bad rounds too likely
# Good rounds not likely enough
# reduce skew








