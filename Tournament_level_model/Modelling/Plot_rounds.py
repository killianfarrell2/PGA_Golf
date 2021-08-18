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

# Select columns needed
subset = training_data[['player','i_golfer','tournament id','date','course','Round','Round_Score']]
# Divide into rounds
r1= subset[subset['Round']==1][['player','i_golfer','tournament id','date','course','Round_Score']]
r2= subset[subset['Round']==2][['i_golfer','tournament id','Round_Score']]
r3= subset[subset['Round']==3][['i_golfer','tournament id','Round_Score']]
r4= subset[subset['Round']==4][['i_golfer','tournament id','Round_Score']]

# Rename columns
r1 = r1.rename(columns={'Round_Score': "R1"})
r2 = r2.rename(columns={'Round_Score': "R2"})
r3 = r3.rename(columns={'Round_Score': "R3"})
r4 = r4.rename(columns={'Round_Score': "R4"})

# Combine rounds
combine_rounds = pd.merge(r1, r2, on=['i_golfer','tournament id'],how="left")
combine_rounds = pd.merge(combine_rounds, r3, on=['i_golfer','tournament id'],how="left")
combine_rounds = pd.merge(combine_rounds, r4, on=['i_golfer','tournament id'],how="left")


# Filter for rounds where we have both R1 and R2
filter_rounds = combine_rounds[combine_rounds['R1'].notnull()]
filter_rounds = filter_rounds[filter_rounds['R2'].notnull()]


# Correlation coefficient of 0.2
print('Corr coeff ',filter_rounds['R1'].corr(filter_rounds['R2']))


# Scatter plot matplotlib
plt.scatter(filter_rounds.R1, filter_rounds.R2,label=f'R1-R2 Correlation = {np.round(np.corrcoef(filter_rounds.R1,filter_rounds.R2)[0,1], 2)}')
plt.title('R1 vs R2 Scores')
plt.xlabel('R1')
plt.ylabel('R2')
plt.legend()
plt.show()


import seaborn as sns
plt.rcParams.update({'figure.figsize':(10,8), 'figure.dpi':100})
sns.lmplot(x='R1', y='R2', data=filter_rounds)
plt.title("Scatter Plot with Linear fit")
plt.show()




