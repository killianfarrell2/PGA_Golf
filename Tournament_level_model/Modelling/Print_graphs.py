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

# Generate random scores with Normal Distribution
random_scores = np.random.normal(loc=mu_graph, scale=std_graph, size=length)
# Round random scores to nearest integer
random_scores_rd = pd.DataFrame(random_scores.round(0))

# K-S test

import matplotlib.pyplot as plt
import scipy.stats as sc
from scipy.stats import norm
import statistics


# K-S test comparing both samples
ks1 = sc.ks_2samp(random_scores_rd, data['Round_Score'], alternative='two-sided', mode='auto')

# Max distance from distributions - want closer to 0
ks1_dstat = ks1[0]
# P value - reject null that drawn from same distribution if less than 5%
# My p value is less than 5% so reject null that they are the same
ks1_pvalue = ks1[1]

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


# Tail risk - positive skew (0.35) - mean is greater than mode - right tail longer
# More scores left of distribution, but more extremes on the right
# more likely to get a player to shoot +16 than  -16

#Kurtosis is 0.38 - used to describe extreme values in 1 tail vs the other
# measures outliers
# High Kurtosis indicates heavy tails
# Low Kurtosis means few outliers
# Kurtosis < 3 - distribution is shorter tails are thinner

    
# Plot between -30 and 30 with
# 0.1 steps.
x_axis = np.arange(-15, 15, 0.1)
    

    
plt.plot(x_axis, norm.pdf(x_axis, mu_graph, std_graph))
plt.show()

# the histogram of the data vs Normal Distribution

plt.hist(data['Round_Score'].values, 20, density=True, facecolor='b', alpha=0.75)
plt.plot(x_axis, norm.pdf(x_axis, mu_graph, std_graph))
plt.xlabel('Score to Par')
plt.ylabel('Probability')
plt.title('Histogram of golfer round scores to par')
plt.text(0, 0.15, r'$\mu=-0.68,\ \sigma=3.2,skew=0.35$')
plt.xlim(-15, 15)
plt.ylim(0, 0.20)
plt.grid(True)
plt.show()
    

import statsmodels.api as sm
from scipy.stats import norm
import pylab

#Theoretical Quantiles on X-axis - mean = 0 sd = 1
# Ordered value for random variable on the y axis
# Mas score is 16 over, Min Score is -12
# This indicates that there is fat tails

my_data = data['Round_Score']
sm.qqplot(my_data, line='45')
pylab.show()

# K-S test with Normal Distribution
from scipy.stats import kstest, norm
my_data = data['Round_Score']
ks_statistic, p_value = kstest(my_data, 'norm')
print(ks_statistic, p_value)

# If Value of K-s = 0 then we assume it is Normal
# Our K-S is 0.39 and p value is 0.0 - needs to be greater than 0.05

#Get mean and standard deviations of all round scores to par
round_score_mean = np.mean(data['Round_Score'].values)
round_score_std = np.std(data['Round_Score'].values)
print('mean score:',round_score_mean)
print('std score:',round_score_std)

#Select Jon Rahm
jon_rahm = data[data['player']=='Jon Rahm']
rahm_mean = np.mean(jon_rahm['Round_Score'].values)
rahm_std = np.std(jon_rahm['Round_Score'].values)
print('mean score:',rahm_mean)
print('std score:',rahm_std)


plt.hist(jon_rahm['Round_Score'].values, 10, density=True, facecolor='b', alpha=0.75)
plt.plot(x_axis, norm.pdf(x_axis, rahm_mean, rahm_std))
plt.xlabel('Score to Par')
plt.ylabel('Probability')
plt.title('Jon Rahm round scores to par')
plt.text(0, 0.15, r'$\mu=-2.2,\ \sigma=3.02$')
plt.xlim(-15, 15)
plt.ylim(0, 0.2)
plt.grid(True)
plt.show()


#Select Shane Lowry
shane_lowry = data[data['player']=='Shane Lowry']
lowry_mean = np.mean(shane_lowry['Round_Score'].values)
lowry_std = np.std(shane_lowry['Round_Score'].values)
lowry_skew = shane_lowry['Round_Score'].skew()
lowry_kurt = shane_lowry['Round_Score'].kurtosis()
print('mean score:',lowry_mean)
print('std score:',lowry_std)
print('skew:',lowry_skew )
print('kurtosis:',lowry_kurt)




plt.hist(shane_lowry['Round_Score'].values, 10, density=True, facecolor='b', alpha=0.75)
plt.plot(x_axis, norm.pdf(x_axis, lowry_mean, lowry_std))
plt.xlabel('Score to Par')
plt.ylabel('Probability')
plt.title('Shane Lowry round scores to par')
plt.text(0, 0.15, r'$\mu=-0.5,\ \sigma=2.94$')
plt.xlim(-15, 15)
plt.ylim(0, 0.2)
plt.grid(True)
plt.show()








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


#Set values to be used as x
observed_golfers = training_data.i_golfer.values
observed_golfers_shared = shared(observed_golfers)

observed_round_score = training_data.Round_Score.values


observed_courses = training_data.i_course.values

#Get unique number of golfers - shape will be ok below
num_golfers = len(training_data.i_golfer.drop_duplicates())
num_courses = len(training_data.i_course.drop_duplicates())

#Normal Distribution - gives output of decimal numbers - we need whole numbers
#Poisson Distribution - is discreet but may not be accurate
# Think that Normal Distribution needs to be taken - and then round values from trace
# 51 divergences with intercept - removed and got slight warning
# Poisson and Normal similar at large numbers - mean equals variance for poisson
# This means for Poisson - low scores are a lot more likely
# Negative Binomial may be more appropriate
# Prior is belief before evidence is taken into account
# Intercept is expected mean value of y (golfer to par) when all x = 0
# When intercept gets added in it is less stable for mean golfer

with pm.Model() as model:
    
    #Global model parameters - all golfers have same sd
    #sd_global = pm.HalfStudentT("sd_global", nu=3, sigma=2.5)
    #mean_global = pm.Normal('mean_global', mu=0, sigma=3)
    #intercept = pm.Normal('intercept', mu=0, sigma=3)

    # golfer specific parameters
    mean_golfer = pm.Normal('mean_golfer', mu=0, sigma=3, shape=num_golfers)
    sd_golfer = pm.HalfStudentT("sd_golfer", nu=3, sigma=2.5, shape=num_golfers)
         
    # Observed scores to par follow normal distribution
    golfer_to_par = pm.Normal("golfer_to_par", mu=mean_golfer[observed_golfers_shared], sigma=sd_golfer[observed_golfers_shared], observed=observed_round_score)
    
   

#Set cores to 1
# Tuning samples will be discarded
with model:
    trace = pm.sample(1000,chains=2, tune=1000, cores=1,random_seed=1234)

# Plot model in graph format 
pm.model_to_graphviz(model)

container = az.from_pymc3(trace=trace)
summary_trace = az.summary(container)

# Create dataframe with trace
df_trace = pm.trace_to_dataframe(trace)

# Check if NUTS sampler converged
# Bayesian fraction of missing information
# Gelman rubin - evaluates difference between multiple markov chains
# Want to see value of less than 1.1 - want to have 1 ideally
# currently 1.013
bfmi = np.max(pm.stats.bfmi(trace))
max_gr = max(np.max(gr_stats) for gr_stats in pm.stats.rhat(trace).values()).values


(
    pm.energyplot(trace, legend=False, figsize=(6, 4)).set_title(
        f"BFMI = {bfmi}\nGelman-Rubin = {max_gr}"
    )
)


#Plot Posterior of specific player - Poisson gives much wider distribution than normal
# Ranges don't look wide enough - shrunk toward average maybe?
# Posterior looks different to summary info - mean is different (-0.53 vs -0.836)
pm.plot_posterior(trace['mean_golfer'][0])


# Update data reference.
pm.set_data({"data": x_test}, model=model)


#Output is decimal number score to par - need to add to par and then round
with model:
    pp_trace = pm.sample_posterior_predictive(trace,samples=100)
    


    





