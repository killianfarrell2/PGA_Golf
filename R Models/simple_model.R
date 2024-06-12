# Import libraries
library(Metrics)
# Bayesian package (sample function)
library(arm)
library(moments)

# Import data
data_location <- 'C:/KF_Repo/PGA_Golf/Tournament_level_model/Data_manipulation/model_data.csv'
mydata <- read.table(data_location, header=TRUE, sep=",")
# Select subset of columns
mydata2 <- mydata[,c('player','course','date','Round','Round_Score','Round_total')]

# set column as date type
mydata2$date <- as.Date(mydata2$date, format= "%Y-%m-%d")

# Add in a column for date of round
mydata2['date_round'] <- mydata2['date']+ (mydata2['Round'] - 4)

# get list of unique tournament rounds
unique_tr <- data.frame(unique(mydata2[,c('date_round','course')]))
# order by round and course name
unique_tr <- unique_tr[order(unique_tr$date_round,unique_tr$course),]
# Create column as tournament round id
unique_tr$tr_id <- seq.int(nrow(unique_tr))
# Join 2 dataframes to get tournament round id
mydata2 <- merge(mydata2, unique_tr, by = c('date_round','course'))


# get list of unique players and put into a dataframe
uniqueplayers <- data.frame(unique(mydata2[,c('player')]))

# player list (16 total)
my_players <- list('Bryson DeChambeau', 'Hideki Matsuyama',
                   'Jordan Spieth','Matthew Fitzpatrick',
                   'Rory McIlroy','Shane Lowry',
                   'Viktor Hovland','Will Zalatoris',
                   'Collin Morikawa','Cameron Smith',
                   'Dustin Johnson','Justin Thomas',
                   'Seamus Power','Tony Finau',
                   'Xander Schauffele','Tiger Woods')

# Subset from list of selected players
mydata2 <- subset(mydata2, mydata2 = player %in%  my_players)

player <- c('Bryson DeChambeau', 'Hideki Matsuyama',
            'Jordan Spieth','Matthew Fitzpatrick',
            'Rory McIlroy','Shane Lowry',
            'Viktor Hovland','Will Zalatoris',
            'Collin Morikawa','Cameron Smith',
            'Dustin Johnson','Justin Thomas',
            'Seamus Power','Tony Finau',
            'Xander Schauffele','Tiger Woods')
# Used 2020 data on average drive - no data for Zalatoris or Woods
avg_dist <- c(322.1,304.4,301.6,294.7,314.0,299.4,299.3,311.8,
              297.3,299.8,311.0, 304.2, 304.5, 309.8,305.4, 285)

# Combine columns
Drive_dist <- data.frame(player, avg_dist)

# Add column to dataset
mydata2 <- merge(mydata2,Drive_dist,by="player")

# get list of dates
date_list <- aggregate(mydata2$date, by=list(mydata2$date), FUN=length)

# Split into training and test
subset_train <- subset(mydata2, date< "2021-01-01")
subset_test <- subset(mydata2, date>= "2021-01-01")

# get unique courses
uniquecourses <- data.frame(unique(subset_train[,c('course')]))


# varying intercept model with no predictors
# allows the intercept to vary by player
# residual is 3.11 which is same as above
# std dev for player is 0.47 - group level variation for player (diff than estimate)
M0 <- lmer (Round_Score ~ 1 + (1 | player),data = subset_train) 
display (M0)

# coefficient for each player
coef(M0)

# Fixed effects (this is the coefficient estimate for intercept)
# Model averaging over players
fixef(M0)
#(Intercept)
#-1.414404
# Std error
se.fixef(M0)
#0.146549 (smaller than intercept of lm of 0.3)

# Estimate of average player is -1.41

# get average of subset train - this is -1.44 (close to intercept)
# different number of observations for players will skew this
mean(subset_train$Round_Score)

# Random effect (how much intercept is shifted up or down for players)
# Bryson is -0.12
# his coefficient is -1.53
# Random effect is difference between intercept and for player
ranef(M0)
# Std error (differ according to sample size)
# Will zalatoris has 11 observations and 0.42 standard error
# Tony Finau has 126 obs and lowest se at 0.23
se.ranef(M0)

# Just select Bryson De Chambeau
subset_BDC <- subset(subset_train,player %in%  'Bryson DeChambeau')

# Create model with an intercept only
fit.BDC <- lm (Round_Score ~ 1, data = subset_BDC)

# Estimate for intercept is the average score
summary(fit.BDC)


# Select Bryson and Morikawa
subset_BDC_2 <- subset(subset_train,player %in%  list('Bryson DeChambeau','Collin Morikawa'))

# Create model with an intercept only
fit.BDC_2 <- lm (Round_Score ~ player -1, data = subset_BDC_2)

# Estimate for intercept is the average score
summary(fit.BDC_2)



# Select Bryson and Morikawa
subset_BDC_2 <- subset(subset_train,player %in%  list('Bryson DeChambeau','Collin Morikawa'))

# Create model with an intercept only
fit.all <- lm (Round_Score ~ player -1, data = subset_train)

# Estimate for intercept is the average score
summary(fit.all)



# Create model completely pooled
fit.np <- lm (Round_Score ~ 1, data = subset_train)

# Estimate for intercept is the average score
# residual standard deviation is 3.141 - same as standard dev of data
summary(fit.np)











