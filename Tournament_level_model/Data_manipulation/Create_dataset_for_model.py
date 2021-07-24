import pandas as pd

#Import Tournament data
tournament_location = 'D:\\KF_Repo\\PGA_Golf\\Tournament_level_model\\Data_manipulation\\PGA_subset.csv'
tournament_data = pd.read_csv(tournament_location)

# Import Hole Level data
hole_level_location = 'D:\\KF_Repo\\PGA_Golf\\Tournament_level_model\\Data_manipulation\\PGA_subset_hole_level.csv'
hole_level_data = pd.read_csv(hole_level_location)

#Bad Initial Energy when using just 1 tournament - so combining all
tournament_data['round_par'] = tournament_data['hole_par']/tournament_data['n_rounds']
tournament_data['Final_score'] = tournament_data['strokes'] - tournament_data['hole_par']

# Divide data up into Rounds
R1 = hole_level_data[hole_level_data['round']==1]
R2 = hole_level_data[hole_level_data['round']==2]
R3 = hole_level_data[hole_level_data['round']==3]
R4 = hole_level_data[hole_level_data['round']==4]

#Break out into rounds
g1 = pd.DataFrame(R1.groupby(['tournament id','player id'])['Strokes Above Par'].sum())
g1 = g1.rename(columns={'Strokes Above Par': 'Round_Score'})
g1['Round'] = 1

g2 = pd.DataFrame(R2.groupby(['tournament id','player id'])['Strokes Above Par'].sum())
g2 = g2.rename(columns={'Strokes Above Par': 'Round_Score'})
g2['Round'] = 2

g3 = pd.DataFrame(R3.groupby(['tournament id','player id'])['Strokes Above Par'].sum())
g3 = g3.rename(columns={'Strokes Above Par': 'Round_Score'})
g3['Round'] = 3

g4 = pd.DataFrame(R4.groupby(['tournament id','player id'])['Strokes Above Par'].sum())
g4 = g4.rename(columns={'Strokes Above Par': 'Round_Score'})
g4['Round'] = 4

 
#Merge individual rounds
merge_1 = pd.merge(tournament_data, g1, left_on=['tournament id','player id'], right_index=True,how='inner')
merge_2 = pd.merge(tournament_data, g2, left_on=['tournament id','player id'], right_index=True,how='inner')
merge_3 = pd.merge(tournament_data, g3, left_on=['tournament id','player id'], right_index=True,how='inner')
merge_4 = pd.merge(tournament_data, g4, left_on=['tournament id','player id'], right_index=True,how='inner')


#Union Dataframes
union = pd.concat([merge_1, merge_2, merge_3,merge_4])
#Get total round score
union['Round_total'] = union['round_par'] + union['Round_Score']

#Drop column player id - Can't use golfer ID - get index 9621 is out of bounds error
union = union.drop(columns=['player id'])

# Create new column i_golfer
golfers = union.player.unique()
golfers = pd.DataFrame(golfers, columns=["golfer"])
golfers["i"] = golfers.index

# Add i column back to dataframe
union = pd.merge(union, golfers, left_on="player", right_on="golfer", how="left")
union = union.rename(columns={"i": "i_golfer"}).drop("golfer", 1)

# create a group using groupby
g_obs = union.groupby("player")['hole_par'].count()


# Create new column i_course
courses = union.course.unique()
courses = pd.DataFrame(courses, columns=["course"])
courses["i"] = courses.index

# Add i column back to dataframe
union = pd.merge(union, courses, left_on="course", right_on="course", how="left")
union = union.rename(columns={"i": "i_course"})

  
#Save file as csv
union.to_csv('D:\\KF_Repo\\PGA_Golf\\Tournament_level_model\\Data_manipulation\\model_data.csv', index=False)
