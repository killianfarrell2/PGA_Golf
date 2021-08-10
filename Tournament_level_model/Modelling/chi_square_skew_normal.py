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

#Gaussian Inference - Round scores to par are normally distributed
az.plot_kde(data['Round_Score'].values, rug=True)
plt.yticks([0], alpha=0);

import matplotlib.pyplot as plt

mu_graph = data['Round_Score'].mean()
print(mu_graph)
mode_graph = data['Round_Score'].mode()
print(mode_graph)
median_graph = data['Round_Score'].median()
print(median_graph)

std_graph = data['Round_Score'].std()
print(std_graph)
min_score = data['Round_Score'].min()
print(min_score)
max_score =data['Round_Score'].max()
print(max_score)
skew =data['Round_Score'].skew()
print(skew)
kurtosis =data['Round_Score'].kurtosis()
print(kurtosis)
length = len(data['Round_Score'])
print(length)

# Generate random scores with Skewed Normal Distribution
import matplotlib.pyplot as plt
import scipy.stats as sc
from scipy.stats import norm
import statistics

random_scores = sc.skewnorm.rvs(a=skew, loc=mu_graph, scale=std_graph, size=length)
# Round random scores to nearest integer
random_scores_rd = pd.DataFrame(random_scores.round(0))

# Observed values
freq_1 = pd.DataFrame(data['Round_Score'].value_counts())
freq_1 = freq_1.reset_index()

# Expected values
freq_exp = pd.DataFrame(random_scores_rd.value_counts())
freq_exp = freq_exp.rename(columns={0: "expected"})
freq_exp = freq_exp.reset_index()
freq_exp = freq_exp.rename(columns={0: "index"})

# Join actual and expected
join = pd.merge(freq_1, freq_exp, left_on='index', right_on='index', how="left")

# Update nan to 0
join['expected'] = join['expected'].fillna(0)

#Get number of categories
dof = len(join['expected'])

# Chi squared test
# test the null that categorical data has given frequencies
chi_1 = sc.chisquare(join['Round_Score'],ddof =0,  f_exp=join['expected'],axis=0)

#P value - less than 5% - can conclude data not drawn from same distribution
# High p values indicate goodness of fit
chi_1_pv = chi_1[1]


#Kurtosis is 0.38 - used to describe extreme values in 1 tail vs the other
# measures outliers
# High Kurtosis indicates heavy tails
# Low Kurtosis means few outliers
# Kurtosis < 3 - distribution is shorter tails are thinner

