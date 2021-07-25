import pandas as pd
#Downgraded Arviz to 0.11.
import pymc3 as pm
import arviz as az
import matplotlib.pyplot as plt
import theano.tensor as tt
import numpy as np

#Import Tournament data
data_location = 'D:\\KF_Repo\\PGA_Golf\\Tournament_level_model\\Data_manipulation\\model_data.csv'
union = pd.read_csv(data_location)

#Gaussian Inference - Round scores to par are normally distributed
az.plot_kde(union['Round_Score'].values, rug=True)
plt.yticks([0], alpha=0);


g_count = pd.DataFrame(union.groupby("player")['hole_par'].count())
g_count = g_count.rename(columns={"hole_par": "Count_rounds"})

# Join count column
union = pd.merge(union, g_count, left_on="player", right_index=True, how="left")

#Filter out golfers with less than 28 rounds played
union = union[union["Count_rounds"]>=28]


#Drop column player id - Can't use golfer ID - get index 9621 is out of bounds error
union = union.drop(columns=['player id'])

# Create new column i_golfer
golfers = union.player.unique()
golfers = pd.DataFrame(golfers, columns=["golfer"])
golfers["i"] = golfers.index

# Add i column back to dataframe
union = pd.merge(union, golfers, left_on="player", right_on="golfer", how="left")
union = union.rename(columns={"i": "i_golfer"}).drop("golfer", 1)

# Create new column i_course
courses = union.course.unique()
courses = pd.DataFrame(courses, columns=["course"])
courses["i"] = courses.index

# Add i column back to dataframe
union = pd.merge(union, courses, left_on="course", right_on="course", how="left")
union = union.rename(columns={"i": "i_course"})

#Set values
observed_golfers = union.i_golfer.values
observed_golf_round = union.Round.values
observed_round_score = union.Round_Score.values
observed_round_total = union.Round_total.values
observed_round_par = union.par.values
observed_courses = union.i_course.values

#Get unique number of golfers
num_golfers = len(union.i_golfer.drop_duplicates())
num_courses = len(union.i_course.drop_duplicates())

#Normal Distribution - gives output of decimal numbers - we need whole numbers
#Poisson Distribution - is discreet but may not be accurate
# Think that Normal Distribution needs to be taken - and then round values from trace
# 51 divergences with intercept - removed and got slight warning
# Poisson and Normal similar at large numbers - mean equals variance for poisson
# This means for Poisson - low scores are a lot more likely
# Negative Binomial may be more appropriate

with pm.Model() as model:
    
    #Global model parameters - all golfers have same sd
    #sd_global = pm.HalfStudentT("sd_global", nu=3, sigma=2.5)
    #mean_global = pm.Normal('mean_global', mu=0, sigma=3)

    # golfer specific parameters
    mean_golfer = pm.Normal('mean_golfer', mu=0, sigma=3, shape=num_golfers)
    sd_golfer = pm.HalfStudentT("sd_golfer", nu=3, sigma=2.5, shape=num_golfers)
        
    # Observed scores to par follow normal distribution
    golfer_to_par = pm.Normal("golfer_to_par", mu=mean_golfer[observed_golfers], sigma=sd_golfer[observed_golfers], observed=observed_round_score)
    
   
    
#Set cores to 1
with model:
    trace = pm.sample(1000, tune=1000, cores=1)

container = az.from_pymc3(trace=trace)
summary_container = az.summary(container)

# Create dataframe with trace
df_trace = pm.trace_to_dataframe(trace)


#Plot Posterior of specific player - Poisson gives much wider distribution than normal
# Ranges don't look wide enough - shrunk toward average maybe?
# Posterior looks different to summary info - mean is different (-0.53 vs -0.836)
pm.plot_posterior(trace['mean_golfer'][0])


#Output is decimal number score to par
with model:
    pp_trace = pm.sample_posterior_predictive(trace,samples=100)


