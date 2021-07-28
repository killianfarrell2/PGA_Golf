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
# Shift data by 18
data['Round_Score_shift'] = data['Round_Score'] + 18

data['Round_Score_shift'].mean()
17.24735700359722
data['Round_Score_shift'].std()*data['Round_Score_shift'].std()
10.040361602362125

# Under dispersion
print(10.04/17.24) 
0.5823665893271461

#Split into training data with rough 80:20 split
training_data = data[data['date'] <'2020-10-01']
test_data = data[data['date'] >='2020-10-01']


import statsmodels.formula.api as smf

from patsy import dmatrices

f_nb = 'Round_Score_shift~C(i_golfer)'

# Alpha parameter can be used to specify how much mean varies from variance
response, predictors = dmatrices(f_nb, training_data, return_type='dataframe')
model_nb = sm.GLM(response, predictors,family=sm.families.NegativeBinomial(alpha=1)).fit()
print(model_nb.summary())


#Add constant
X_test = test_data['i_golfer']
X_test = sm.add_constant(X_test)

#Get prediction - this is expected mean value
ypred = model_nb.predict(X_test)

y_true = test_data['Round_Score_shift'].values

# Metrics
from sklearn import metrics

# Print results of MAE
print(metrics.mean_absolute_error(y_true, ypred))

# Print results of MSE
print(metrics.mean_squared_error(y_true, ypred))

# Print results of RMSE
print(np.sqrt(metrics.mean_squared_error(y_true, ypred)))



#Poisson stats
2.5523287911598462
10.298705102508523
3.209159563267075






