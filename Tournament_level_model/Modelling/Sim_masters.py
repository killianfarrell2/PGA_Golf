import pandas as pd
import numpy as np
import pystan
import arviz as az
import matplotlib.pyplot as plt
import random

#Import Tournament data
data_location = 'C:\\KF_Repo\\PGA_Golf\\sample_masters.csv'
data = pd.read_csv(data_location)


# Generate Random Numbers
num_sims = 10000
std_dev = 1.7

#Create new dataframe for trace values with 10k simulations
trace = pd.DataFrame()

for i in data.index: 
    # Set mu as mean value
    mu = data['Mean'][i]
    # Get golfer
    golfer = data['Golfer'][i]
    #Create column in dataframe for each golfer
    trace[golfer] = pd.DataFrame(np.round(np.random.normal(mu, std_dev, num_sims),0))



# Sim Tournament

winner = np.empty(0, dtype=object)
made_cut = np.empty(0, dtype=object)
placed = np.empty(0, dtype=object)

num_sims = 10000

for i in range(num_sims):
    # get entries from trace to simulate scores for each round
    r1 = trace.iloc[[random.randint(0,9999)]]
    r2 = trace.iloc[[random.randint(0,9999)]]
    r3 = trace.iloc[[random.randint(0,9999)]]
    r4 = trace.iloc[[random.randint(0,9999)]]
    
    # Combine round 1 and round 2
    r1_r2_combined = pd.concat([r1, r2])
    # Combine round 1,2,3,4
    r1_r4_combined = pd.concat([r1, r2,r3,r4])
    
    # Sum up round 1 and round 2 and sort by strokes gained
    sum_r1_r2 = r1_r2_combined.sum(axis=0).sort_values(ascending=False)
    
    # Only take top 50 as they make cut
    top_50 = sum_r1_r2[0:50]
    
    # Sum up round 1 to 4
    sum_r1_r4 = r1_r4_combined.sum(axis=0).sort_values(ascending=False)
    
    # Get final standings by only taking those who made the cut
    final_standings = sum_r1_r4[sum_r1_r4.index.isin(top_50.index)]
 
    # Set winner for each simulation
    winner = np.append(winner,final_standings.index[0])
    
    # Set plaed for each simulation
    placed = np.append(placed,final_standings.index[0:8])
    
    # Set made cut for each simulation
    made_cut = np.append(made_cut,final_standings.index)



unique_w, counts_w = np.unique(winner, return_counts=True)
unique_mc, counts_mc = np.unique(made_cut, return_counts=True)
unique_p, counts_p = np.unique(placed, return_counts=True)



# Reshape arrays
unique_w = unique_w.reshape(len(unique_w),1)
counts_w = counts_w.reshape(len(counts_w),1)
unique_mc = unique_mc.reshape(len(unique_mc),1)
counts_mc = counts_mc.reshape(len(counts_mc),1)
unique_p = unique_p.reshape(len(unique_p),1)
counts_p = counts_p.reshape(len(counts_p),1)


# Combine Arrays
combine_final_w = pd.DataFrame(np.concatenate((unique_w,counts_w),axis=1))

# Add Percent column
combine_final_w['percent'] = combine_final_w[1] / combine_final_w[1].sum()

# Add odds column
combine_final_w['odds'] = (1 / combine_final_w['percent'] )


# Combine Arrays for made cut
combine_final_mc = pd.DataFrame(np.concatenate((unique_mc,counts_mc),axis=1))

# Add Percent column
combine_final_mc['percent'] = combine_final_mc[1] / num_sims

# Add odds column
combine_final_mc['odds'] = (1 / combine_final_mc['percent'] )



# Combine Arrays for those who placed
combine_final_p = pd.DataFrame(np.concatenate((unique_p,counts_p),axis=1))

# Add Percent column
combine_final_p['percent'] = combine_final_p[1] / num_sims

# Add odds column
combine_final_p['odds'] = (1 / combine_final_p['percent'] )



#Save file as csv
combine_final_p.to_csv('C:\\KF_Repo\\PGA_Golf\\Tournament_level_model\\Masters_outputs\\placed_data.csv', index=False)

#Save file as csv
combine_final_mc.to_csv('C:\\KF_Repo\\PGA_Golf\\Tournament_level_model\\Masters_outputs\\missed_cut_data.csv', index=False)

#Save file as csv
combine_final_w.to_csv('C:\\KF_Repo\\PGA_Golf\\Tournament_level_model\\Masters_outputs\\winner_data.csv', index=False)













