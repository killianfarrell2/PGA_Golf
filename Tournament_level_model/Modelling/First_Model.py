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


#Set values
observed_golfers = union.i_golfer.values
observed_golf_round = union.Round.values
observed_round_score = union.Round_total.values
observed_round_par = union.round_par.values
observed_courses = union.i_course.values

#Get unique number of golfers
num_golfers = len(union.i_golfer.drop_duplicates())
num_courses = len(union.i_course.drop_duplicates())

#Normal Distribution - gives output of decimal numbers - we need whole numbers
#Poisson Distribution - is discreet but may not be accurate

#+ course_star[observed_courses] - made model unstable
# addming in mu golfer brought divergences - seems to be less of a range too
# golfers mean are shrunk toward mu_golfer
# Testing against test set would be the best approach
with pm.Model() as model:
    # global model parameters - all golfers have same sd_att now
    sd_golfer = pm.HalfStudentT("sd_golfer", nu=3, sigma=2.5)
    #mu can be different to 0 now depending on the data
    mu_golfer = pm.Normal("mu_golfer", mu=0, sigma=5)
    #sd_course = pm.HalfStudentT("sd_course", nu=3, sigma=2.5)
    # golfer specific
    golfer_star = pm.Normal("golfer_star", mu=mu_golfer, sigma=sd_golfer, shape=num_golfers)
    
    #Course specific
    #course_star = pm.Normal("course_star", mu=0, sigma=sd_course, shape=num_courses)
    golfer_theta = observed_round_par + golfer_star[observed_golfers] 
    
    # likelihood of observed data
    golfer_score = pm.Poisson("golfer_score", mu=golfer_theta, observed=observed_round_score)

#Set cores to 1
with model:
    trace = pm.sample(1000, tune=1000, cores=1)

# Create dataframe with trace
df_trace = pm.trace_to_dataframe(trace)


#Plot Posterior of specific player - Poisson gives much wider distribution than normal
pm.plot_posterior(trace['golfer_star'][0])

#Get summary statistics for each golfer


hpd = pd.DataFrame(pm.stats.hpd(trace["golfer_star"]))
hpd = hpd.rename(columns={0: "2.5%",1:"97.5%"})

mid = pd.DataFrame(np.quantile(trace["golfer_star"], 0.5, axis=0))
mid = mid.rename(columns={0: "50%"})

g_count = pd.DataFrame(union.groupby("i_golfer")['hole_par'].count())
g_count = g_count.rename(columns={"hole_par": "Count_rounds"})


#Combine golfers
golfer_stats = pd.merge(golfers, hpd, left_on="i", right_index=True, how="left")
golfer_stats = pd.merge(golfer_stats, mid , left_on="i", right_index=True, how="left")
golfer_stats = pd.merge(golfer_stats, g_count , left_on="i", right_index=True, how="left")


#Right now output is decimal number - but we need round numbers
with model:
    pp_trace = pm.sample_posterior_predictive(trace,samples=50)


