import pandas as pd

#Import Tournament data
tournament_location = 'D:\\KF_Repo\\PGA_Golf\\Tournament_level_model\\Data_manipulation\\PGA_subset.csv'
tournament_data = pd.read_csv(tournament_location)

tournament_data['Final_score'] = tournament_data['strokes'] - tournament_data['hole_par']

# Import Hole Level data
# all entries in data have completed 18 holes
hole_level_location = 'D:\\KF_Repo\\PGA_Golf\\Tournament_level_model\\Data_manipulation\\PGA_subset_hole_level.csv'
hole_level_data = pd.read_csv(hole_level_location)


#Break out into rounds
def groupby_hole(Round):
    
    select_round = hole_level_data[hole_level_data['round']==Round]
    groupby_df = pd.DataFrame(select_round.groupby(['tournament id','player id'])['Strokes Above Par','par'].sum())
    groupby_df = groupby_df.rename(columns={'Strokes Above Par': 'Round_Score'})
    grouby_count = pd.DataFrame(select_round.groupby(['tournament id','player id'])['Strokes Above Par'].count())
    grouby_count = grouby_count.rename(columns={'Strokes Above Par': 'Count'})
    groupby_df = pd.merge(groupby_df,  grouby_count, left_index=True, right_index=True,how='inner')
    groupby_df['Round'] = Round
    return groupby_df

g1 = groupby_hole(1)
g2 = groupby_hole(2)
g3 = groupby_hole(3)
g4 = groupby_hole(4)

#Merge individual rounds
merge_1 = pd.merge(tournament_data, g1, left_on=['tournament id','player id'], right_index=True,how='inner')
merge_2 = pd.merge(tournament_data, g2, left_on=['tournament id','player id'], right_index=True,how='inner')
merge_3 = pd.merge(tournament_data, g3, left_on=['tournament id','player id'], right_index=True,how='inner')
merge_4 = pd.merge(tournament_data, g4, left_on=['tournament id','player id'], right_index=True,how='inner')


#Union Dataframes
union = pd.concat([merge_1, merge_2, merge_3,merge_4])
#Get total round score
union['Round_total'] = union['par'] + union['Round_Score']

#Save file as csv
union.to_csv('D:\\KF_Repo\\PGA_Golf\\Tournament_level_model\\Data_manipulation\\model_data.csv', index=False)