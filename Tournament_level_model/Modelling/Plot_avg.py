import pandas as pd
import matplotlib.pyplot as plt
import theano.tensor as tt
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sc
from scipy.stats import norm
import statistics

#Import Tournament data
data_location = 'D:\\KF_Repo\\PGA_Golf\\Tournament_level_model\\Data_manipulation\\model_data.csv'
data = pd.read_csv(data_location)

#Set date to datetime
data['date'] = pd.to_datetime(data['date'])

# Add in Date of Round
data['round_date'] = data['date'] - pd.to_timedelta(4 - data['Round'], unit='d')

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
data['St_V_Avg'] =  data["Round_Score"] - data["Avg_Score"] 


mu_graph = data['St_V_Avg'].mean()
print(mu_graph)
median_graph = data['St_V_Avg'].median()
print(median_graph)
std_graph = data['St_V_Avg'].std()
print(std_graph)
min_score = data['St_V_Avg'].min()
print(min_score)
max_score = data['St_V_Avg'].max()
print(max_score)
skew =data['St_V_Avg'].skew()
print(skew)
kurtosis =data['St_V_Avg'].kurtosis()
print(kurtosis)
length = len(data['Round_Score'])
print(length)


x_axis = np.arange(-15, 15, 0.1)

# the histogram of the data vs Normal Distribution

plt.hist(data['St_V_Avg'].values, 20, density=True, facecolor='b', alpha=0.75)
plt.plot(x_axis, norm.pdf(x_axis, mu_graph, std_graph))
plt.xlabel('Score vs Avg')
plt.ylabel('Probability')
plt.title('Histogram of Scores vs avg')
plt.text(0, 0.15, r'$\mu=0,\ \sigma=2.84,skew=0.30$')
plt.xlim(-15, 15)
plt.ylim(0, 0.20)
plt.grid(True)
plt.show()





