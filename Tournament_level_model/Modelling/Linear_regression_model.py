import pandas as pd
import matplotlib.pyplot as plt

import numpy as np

import statsmodels.api as sm


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


# Create new column i_golfer
golfers = data.player.unique()
golfers = pd.DataFrame(golfers, columns=["golfer"])
golfers["i"] = golfers.index

# Add i column back to dataframe
data = pd.merge(data, golfers, left_on="player", right_on="golfer", how="left")
data = data.rename(columns={"i": "i_golfer"}).drop("golfer", 1)

#Split into training data with rough 80:20 split
training_data = data[data['date'] <'2020-10-01']
test_data = data[data['date'] >='2020-10-01']




import statsmodels.formula.api as smf

f_rev = 'Round_Score~C(i_golfer)'
model_rev = smf.ols(formula=f_rev, data=training_data).fit()
model_rev.summary()

#Predicted values are summarry and then adding golfer coefficient
# For golfer 0 just intercept is used

X_test_orig = test_data['i_golfer']
#Add constant
X_test = test_data['i_golfer']
X_test = sm.add_constant(X_test)
y_true = test_data['Round_Score'].values


ypred = model_rev.predict(X_test)
#Round prediction to give predicted score
ypred_round = ypred.round(0)

# Metrics
from sklearn import metrics


# Print results of MAE
print(metrics.mean_absolute_error(y_true, ypred_round))

# Print results of MSE
print(metrics.mean_squared_error(y_true, ypred_round))

# Print results of RMSE
print(np.sqrt(metrics.mean_squared_error(y_true, ypred_round)))

# Poisson Regression Model slightly better for RMSE and MSE

2.5374607447285777
10.3331837894422
3.2145269931114595







