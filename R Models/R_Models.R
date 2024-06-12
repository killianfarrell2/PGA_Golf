# Import libraries
library(Metrics)
# Bayesian package (sample function)
library(arm)
library(moments)

# Import data
data_location <- 'C:/KF_Repo/PGA_Golf/Tournament_level_model/Data_manipulation/model_data.csv'
mydata <- read.table(data_location, header=TRUE, sep=",")
# Select subset of columns
subset <- mydata[,c('player','course','date','Round','Round_Score','Round_total')]

# set column as date type
subset$date <- as.Date(subset$date, format= "%Y-%m-%d")

# Add in a column for date of round
subset['date_round'] <- subset['date']+ (subset['Round'] - 4)

# get list of unique tournament rounds
unique_tr <- data.frame(unique(subset[,c('date_round','course')]))
# order by round and course name
unique_tr <- unique_tr[order(unique_tr$date_round,unique_tr$course),]
# Create column as tournament round id
unique_tr$tr_id <- seq.int(nrow(unique_tr))
# Join 2 dataframes to get tournament round id
subset <- merge(subset, unique_tr, by = c('date_round','course'))


# get list of unique players and put into a dataframe
uniqueplayers <- data.frame(unique(subset[,c('player')]))


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
subset <- subset(subset, subset = player %in%  my_players)

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
subset <- merge(subset,Drive_dist,by="player")

# get list of dates
date_list <- aggregate(subset$date, by=list(subset$date), FUN=length)

# Split into training and test
subset_train <- subset(subset, date< "2021-01-01")
subset_test <- subset(subset, date>= "2021-01-01")

# get unique courses
uniquecourses <- data.frame(unique(subset_train[,c('course')]))

# Step 1A: Fit classical regressions using lm() and glm()
# Model 1 (Player with intercept)
# Bryson de chambeau is the reference point at 0
fit.1 <- lm (Round_Score ~ player, data = subset_train)
summary(fit.1)
# Get residuals from model
res_fit.1 <- resid(fit.1 )

# plot fitted vs residual
plot(fitted(fit.1), res_fit.1)
abline(0,0)

# Remove intercept term
# Each player has their own intercept
# R squared has now also increased (due to issue with R)
fit.1_other <- lm (formula = Round_Score ~ player - 1, data = subset_train)
summary(fit.1_other)

# Display model coefficients (if 2 std +- range crosses 0 not significant)
# Residual standard error is 3.109 (standard dev of residuals)


#Model 1c - add in course (residual standard error drops to 2.924)
fit.1c <- lm (Round_Score ~ player + course, data = subset_train)
summary(fit.1c)

# Model 1 c without intercept (same residual standard error of 2.924)
# course still only has 53 when there is 54 (accordia is the reference)
# not possible to predict on new values not already in the training set
fit.1c_other <- lm (formula = Round_Score ~ player + course - 1, data = subset_train)
summary(fit.1c_other)


# GLM Models gaussian
# identity used as no transformation needed
fit.2 <- glm(formula=Round_Score ~ player, family = gaussian(link = "identity"), data=subset_train)
summary(fit.2)
# gives null deviance and residual deviance
# Null deviance is only with intercept term
#(Dispersion parameter for gaussian family taken to be 9.664919)
sqrt(9.664919)
#3.108845

#Null deviance: 14079  on 1427  degrees of freedom
#Residual deviance: 13647  on 1412  degrees of freedom

# Predict on test set
fit_2_test <- predict(fit.2, newdata = subset_test)

# RMSE is 3.268979 on test set (same as lm model)
rmse(subset_test$Round_Score, fit_2_test)

# GLM for course model (1c)
fit.2c <- glm(formula=Round_Score ~ player + course-1, family = gaussian(link = "identity"), data=subset_train)
summary(fit.2c)
sqrt(8.547593)
#2.923627



# Step 1B: Regression Inferences

# Get prediction interval for lm model
# Assumes coefficient estimates are normally distributed
pred.interval <- predict (fit.1, newdata = subset_test, interval="prediction",
                          level=.95)

# Predict on training set (assumes coefficient normally distributed)
pred_train.interval <- predict(fit.1, interval="prediction",level=.95)

# Set confidence interval
pred_train.ci_interval <- predict(fit.1, interval='confidence',level=.95)

# Get player names from data
player_names <- subset_train[,c('player')]

# Combine player names to pred_train.interval
pred_train.interval_2 <- data.frame(player_names, pred_train.interval)
pred_train.ci_interval_2 <- data.frame(player_names, pred_train.ci_interval)

# Prediction interval is based solely on best estimate and residual standard error


# Simualte for Morikawa  (only has 1 value for intercept and std dev)
# normal(draws,mean,se)
pred_player_draw <- (rnorm (1000, (-1.59048 -0.22975),  3.109))

# Create histogram for spieth
hist(pred_player_draw)

# Round scores
pred_player_draw_r <- round(pred_player_draw, digits=0)

# Create histogram for rounded scores
hist(pred_player_draw_r)


# MODEL USING SIM FUNCTION: Same as non informative priors in bayesian

# Model 1 with simulation 1k times
# Get a posterior distribution of sigma and beta
# gives 1k values for each player (coeff)
# gives 1k values for sigma
n.sims <- 1000
fit.1 <- lm (Round_Score ~ player, data = subset_train)
sim.1 <- sim(fit.1, n.sims)


# Display values for sigma (residual - shot difference)
sim.1@sigma

# Display values for players
sim.1@coef

# Coefficients from lm model
#playerCollin Morikawa     -0.22975    0.44793

# select coefficients for Colin Morikawa (similar to simulated)
player.sim_coef <- sim.1@coef[,'playerCollin Morikawa']
# simed values for coefficient very similar to estimate and std error
mean(player.sim_coef)
# -0.2326292
sd(player.sim_coef)
# 0.4534639
#95% interval
quantile(player.sim_coef, c(.025,.975))
#2.5%      97.5% 
#-1.1199429  0.5780021


#FAKE DATA SIMULATION
# True values of parameters
a <- 1.4
b <- 2.3
sigma <- 0.9
x <- 1:5 #predictors
n <- length(x)

# fit is different to true values
y <- a + b*x + rnorm (n, 0, sigma)
lm.1 <- lm(y~x)
display(lm.1)

# b is the second coefficient in model
b.hat <- coef (lm.1)[2]
b.se <- se.coef (lm.1)[2]

# Does true beta fall within 68% and 95% intervals
# may work once but how many times will it fall

cover.68 <- abs (b - b.hat) < b.se # this will be TRUE or FALSE
cover.95 <- abs (b - b.hat) < 2*b.se # this will be TRUE or FALSE
cat (paste ("68% coverage: ", cover.68, "\n"))
cat (paste ("95% coverage: ", cover.95, "\n"))


# Check how many times it falls in 1k sims
n.fake <- 1000
cover.68 <- rep (NA, n.fake)
cover.95 <- rep (NA, n.fake)
for (s in 1:n.fake){
  y <- a + b*x + rnorm (n, 0, sigma)
  lm.1 <- lm (y ~ x)
  b.hat <- coef (lm.1)[2]
  b.se <- se.coef (lm.1)[2]
  cover.68[s] <- abs (b - b.hat) < b.se
  cover.95[s] <- abs (b - b.hat) < 2*b.se
}
cat (paste ("68% coverage: ", mean(cover.68), "\n"))
cat (paste ("95% coverage: ", mean(cover.95), "\n"))

# Valus are very close to what is in book
# 68% coverage:  0.595
# 95% coverage:  0.855 

# Only 59.5% of 1 sd intervals contain the true parameter
# Our problem is +-1 and +-2 std error intervals are appropriate for normal
# with such a small sample size inferences should use t distribution
# when you rerun for 100 data points normal is more appropriate

#rerun sumulation using t distribution instead
n.fake <- 1000
cover.68 <- rep (NA, n.fake)
cover.95 <- rep (NA, n.fake)
t.68 <- qt (.84, n-2)
t.95 <- qt (.975, n-2)
for (s in 1:n.fake){
  y <- a + b*x + rnorm (n, 0, sigma)
  m.1 <- lm (y ~ x)
  b.hat <- coef (lm.1)[2]
  b.se <- se.coef (lm.1)[2]
  cover.68[s] <- abs (b - b.hat) < t.68*b.se
  cover.95[s] <- abs (b - b.hat) < t.95*b.se
}
cat (paste ("68% coverage ", mean(cover.68), "\n"))
cat (paste ("95% coverage: ", mean(cover.95), "\n"))

# Interstingly 100% of intervals contain true parameter vs book


# Simulate from fitted model and compare to actual data

# Create 1k simulations of player coefficient and residual error
n.sims <- 1000
fit.1 <- lm (Round_Score ~ player, data = subset_train)
sim.1 <- sim(fit.1, n.sims)

# Step 1: Generate 1k fake datasets of 1,428 observations each
# Get length of Round Scores
n <- length (subset_train[,c('Round_Score')])
# Create array of NAs with scores
y.rep <- array (NA, c(n.sims, n))
# Loop through 10 simulations
for (s in 1:n.sims){
  y.rep[s,] <- round(rnorm (n, sim.1@coef[s], sim.1@sigma[s]),digits=0)
}

# Step 2: Get a test statistic
# E.g Take the minimum and maximum value from data

actual_min <- min(subset_train[,c('Round_Score')])
actual_max <- max(subset_train[,c('Round_Score')])

# Plot Histogram of actual data
hist(subset_train[,c('Round_Score')],breaks = 20)


# Build a contigency table
counts <- table(subset_train[,c('Round_Score')])
# Plot bar chart of scores
barplot(counts, main="Round Distribution",
        xlab="Number of Rounds")
# Actual data scores are centred around -2 with peak


# Plot bar chart of fake scores
# Fake 1 has scores -5 to +2 equally likely
# All distributions have too much variance around centre
# Should be more of a peak
barplot(table( y.rep[15,]), main="Round Distribution fake data",
        xlab="Number of Rounds")

# Should have positive kurtosis (Heavy tailed or light tailed relative to normal)


# Heavy tails tend to have many outliers with many high values




# Test function 1: min value for each simulation (Actual is -11)
test <- function (y){
  min (y)
}
test.rep <- rep (NA, n.sims)
for (s in 1:n.sims){
  test.rep[s] <- test(y.rep[s,])
}

# Plot minimum value (actual min is -11 vs here -12 with rounds of -19)
barplot(table( test.rep), main="Distribution Min score",
        xlab="Number of sims")

# Test function 2: max value for each simulation
test_2 <- function (y){
  max (y)
}
test_2.rep <- rep (NA, n.sims)
for (s in 1:n.sims){
  test_2.rep[s] <- test_2(y.rep[s,])
}

# Plot maximum value (actual max is +10 vs mode of 9 here, up to 14 - max is ok)
barplot(table( test_2.rep), main="Distribution Max score",
        xlab="Number of sims")

# Model is not taking into account skew or kurtosis

# Test function 3: mode for each simulation
test_3 <- function (y){
  uniqv <- unique(y)
  uniqv[which.max(tabulate(match(y, uniqv)))]
}
test_3.rep <- rep (NA, n.sims)
for (s in 1:n.sims){
  test_3.rep[s] <- test_3(y.rep[s,])
}

# Get actual most common score
test_3(subset_train[,c('Round_Score')])

# Actual Mode is -2 - close at -1 and some -3 and 0
barplot(table( test_3.rep), main="Distribution Mode score",
        xlab="Number of sims")

# Test function 4: standard deviation for each simulation
test_4 <- function (y){
  sd (y)
}
test_4.rep <- rep (NA, n.sims)
for (s in 1:n.sims){
  test_4.rep[s] <- test_4(y.rep[s,])
}

# Plot Histogram of standard deviations for fake data
# Standard deviation is centred around 3.15
# Actual is within centre of range
hist(test_4.rep)
lines(rep(test_4(subset_train[,c('Round_Score')]),2), c(0,n))


# Test function 5: kurtosis for each simulation
test_5 <- function (y){
  kurtosis (y)
}
test_5.rep <- rep (NA, n.sims)
for (s in 1:n.sims){
  test_5.rep[s] <- test_5(y.rep[s,])
}


# Plot kurtosis for fake datasets (Centred around 3 with few at 3.3)
# Line is at 3.3 for actual
hist(test_5.rep)
lines(rep(test_5(subset_train[,c('Round_Score')]),2), c(0,n))


# Test function 6: skew for each simulation
test_6 <- function (y){
  skewness (y)
}
test_6.rep <- rep (NA, n.sims)
for (s in 1:n.sims){
  test_6.rep[s] <- test_6(y.rep[s,])
}

# Measure Skew 0.3 Actual
skewness(subset_train[,c('Round_Score')])

# Plot skew for fake datasets (Centred around 0 to 0.5 with none at 0.3)
hist(test_6.rep, xlim=c(-0.2,0.5))
lines(rep(test_6(subset_train[,c('Round_Score')]),2), c(0,n))


# How to add skew and kurtosis
# Assymetric contaminated normal distribution
# symmetric long tailed distribution

# Plot Residuals
# Get residuals
res <- round(resid(fit.1),0)
# Get fitted values
fit_values <- round(fitted(fit.1),0)

# Density plot of residuals
plot(density(resid(fit.1)))
# Density plot rounded residuals
plot(density(res))


#produce residual vs. fitted plot (rounded)
# Only 3 predicted values (-2,-1,0)
plot(fit_values, res)
abline(0,0)

# Plot residuals vs predicted (fitted values all between -2 and 0 (most likely))
# No rounded of residuals or fitted values
plot(fitted(fit.1), resid(fit.1))
abline(0,0)


# Link function
# Maps a non liner relationship to a linear one

# Keep reading - not sure how to proceed
# Linear (Normal) Model not appropriate for prediction
# Missing Skew and Kurtosis - not sure how to fix

# Section 2a of book
# Step 2: Set up multilevel models allowing intercepts and slopes to vary using lmer()

# Additional assumptions beyond classical
# each level of the model corresponds to it's own regression with its own assumptions
# little group level variation - multilevel reduces to classical no group indicator
# when group level vary greatly reduces to classical regression with group indicators
# when number of groups is small - classical regression good

# The 16 coefficients for golfers are given a model - common distribution
# a regression model for the alpha js

# complete pooling ignores variation between golfers
# no pooling overstates variation between golfers (overfits)
# less observations mean golfer pulled to overall average




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
# error is around 0.25 to 0.28 (smaller than lm model 0.4 mostly)
#Number of observations by player
player_count <- aggregate(subset_train$player, by=list(subset_train$player), FUN=length)



# Get 95% interval of fixed effect
# Effect that does not vary by group (average intercept)
fixef(M0)[1] + c(-2,2)*se.fixef(M0)[1]

# 95% Interval for intercept of Bryson De Chambeau
# -2.04 to -1.028
coef(M0)$player[1,1] + c(-2,2)*se.ranef(M0)$player[1]

# 95% interval of deviation from average (Error in the intercept)
# -0.63 to +0.38
as.matrix(ranef(M0)$player)[1] + c(-2,2)*se.ranef(M0)$player[1]

# Nested Models
# Floor Measurement
# County Measurement for uranium

# Course Level Model
# Residual is 2.98
# stdev is 1.08
# Lower Deviance
M1 <- lmer (Round_Score ~ 1 + (1 | course),data = subset_train) 
display (M1)

# Non Nested Models (section 13.5)
# Create lmer model with both player and course
# Residual standard deviation is 2.93 - same as lm models
# Course standard dev is 1.12
# player standard dev is 0.56
# Less variance between players than there is between courses 
M2 <- lmer (Round_Score ~ 1+ (1 | player) + (1 | course),data = subset_train) 
display (M2)

# coefficient for each player and course - both varying intercept
coef(M2)
# get fixed effects of Model (This is -1.58) 
# Think this is average score on average course
fixef(M2)
# Random effect - how each course and player affects course
ranef(M2)
# Save dataframe as csv
write.csv(subset_train,"C:\\Users\\killi\\OneDrive\\Desktop\\subset_train.csv", row.names = FALSE)

# Add in player level variable - like county uranium measure from book
# each yard of driving distance decreases score by -0.03
# Residual is still 2.93
# player variance has dropped to 0.52
M3 <- lmer(Round_Score ~  1+ (1 | player) + (1 | course) + avg_dist,data = subset_train)
display (M3)

# Using tournament round 
# Residual drops to 2.77
# Player variance is 0.46
# tournament round variance is 1.41
M4 <- lmer(Round_Score ~  1+ (1 | player) + (1 | tr_id) + avg_dist,data = subset_train)
display (M4)

# Add in course affect as well
# Residual is 2.78
# Tournament round is a subset of course so may not be appropriate together
M5 <- lmer(Round_Score ~  1+ (1 | player) + (1 | course) + (1 | tr_id) + avg_dist,data = subset_train)
display (M5)






# Step 3: Fit fully bayesian models using BUGS

model {
  for (i in 1:n){
    y[i] ~ dnorm (y.hat[i], tau.y)
    y.hat[i] <- a[county[i]] + b*x[i]
  }
  b ~ dnorm (0, .0001)
  tau.y <- pow(sigma.y, -2)
  sigma.y ~ dunif (0, 100)
  for (j in 1:J){
    a[j] ~ dnorm (mu.a, tau.a)
  }
  mu.a ~ dnorm (0, .0001)
  tau.a <- pow(sigma.a, -2)
  sigma.a ~ dunif (0, 100)
}


radon.data <- list ("n", "J", "y", "county", "x")
radon.inits <- function (){
  list (a=rnorm(J), b=rnorm(1), mu.a=rnorm(1),
        sigma.y=runif(1), sigma.a=runif(1))}
radon.parameters <- c ("a", "b", "mu.a", "sigma.y", "sigma.a")













