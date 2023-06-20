

############################################################################################
#             W5451 : Statistical Learning for Data Science with R and Python              #
#                                                                                          #
#                                                                                          #
#    Project 1: Application of different Multiple Linear Regression models                 #
#               using R programming.                                                       #
#                                                                                          # 
#                   Group No : 6        Submitted by: Trusha Patel   6872844               #    
############################################################################################

library(dplyr)
library(caret)
library(rsq)
library(ggplot2)

##################  Example 1   #################

# Data 

data <- read.csv("Wind.csv", header=TRUE)
head(data)
colnames(data) = c("date","AP","WS","TP","WD")
data <- select(data, -date)

# Split the data into training and testing sets
set.seed(123)
split <- createDataPartition(data$AP, p = 0.8, list = FALSE)
train_data <- data[split,]
test_data <- data[-split,]


# Fit a multiple linear regression model using all variables
model1 <- lm(AP ~ ., data = train_data)
summary(model1)

# Fit a multiple linear regression model using WS and WD variables
model2 <- lm(AP ~ WS + WD, data = train_data)
summary(model2)

# Fit a multiple linear regression model using interaction term
model3 <- lm(AP ~ WS*WD, data = train_data)
summary(model3)


# Fit a multiple linear regression model using Non-linear Transformations of the Predictors.
model4 <- lm(AP ~ WS + I(WS^2), data = train_data)
summary(model4)


# Multiple linear regression using polynomials of the independent variables
model5 <- lm(AP ~ poly(WS, 2)+poly(TP, 2)+poly(WD, 2), data = train_data)
summary(model5)

# adj R squared values of all models 

adjRsq1 <- summary(model1)$adj.r.squared
adjRsq2<- summary(model2)$adj.r.squared
adjRsq3 <- summary(model3)$adj.r.squared
adjRsq4<- summary(model4)$adj.r.squared
adjRsq5<- summary(model5)$adj.r.squared
adj_R <- cbind(adjRsq1,adjRsq2,adjRsq3,adjRsq4, adjRsq5)

## anova
anova(model1 ,model2 , model3, model4, model5)

# Plots
Wind_Speed <- train_data$WS
Active_Power <-train_data$AP
plot(Wind_Speed,Active_Power)

#models 1 and 5
points(Wind_Speed, fitted(model1), col="red",pch=20)
points(Wind_Speed, fitted(model5), col="blue",pch=20)
title("Scatter Plot of Model 1 and 5") 

#models 2, 3 ,and 4
plot(Wind_Speed,Active_Power)
points(Wind_Speed, fitted(model2), col="pink",pch=20)
points(Wind_Speed, fitted(model3), col="yellow",pch=20)
points(Wind_Speed, fitted(model4), col="lightblue",pch=20)
title("Scatter Plot of Model 2,3 and 4") 

#Predict
predictions <- predict(model5, newdata = test_data)

ggplot() +
  geom_point(aes(x = predictions, y = test_data$AP), color = "blue") +
  geom_abline(intercept = 0, slope = 1, color = "red")

##################  Example 2   #################

# Data 
data2 <- read.csv("Electricity.csv", header=TRUE)
data2 <- select(data2, -Y2, -X4)
head(data2)

# Split the data into training and testing sets
set.seed(1234)
split2 <- createDataPartition(data2$Y1, p = 0.8, list = FALSE)
train_data2 <- data2[split2,]
test_data2 <- data2[-split2,]

# Fit a multiple linear regression model using all variables
model11 <- lm(Y1 ~ ., data = train_data2)
summary(model11)

# Fit a multiple linear regression model using important variables
model12 <- lm(Y1 ~ X2 + X3 + X5 + X7, data = train_data2)
summary(model12)

# Fit a multiple linear regression model using interaction term
model13 <- lm(Y1 ~ X2*X5, data = train_data2)
summary(model13)

# Fit a multiple linear regression model using polynomial
model14 <- lm(Y1 ~ poly(X1, 2) + poly(X2, 2) + poly(X3, 2) + poly(X6, 2) + poly(X7, 2) + poly(X8, 2), data = train_data2)
summary(model14)

#adjusted R squared values

adjRsq11 <- summary(model11)$adj.r.squared
adjRsq12<- summary(model12)$adj.r.squared
adjRsq13 <- summary(model13)$adj.r.squared
adjRsq14<- summary(model14)$adj.r.squared

adj_R1 <- cbind(adjRsq11,adjRsq12,adjRsq13,adjRsq14)

#Anova
anova(model11 ,model12 , model13, model14)

#predict
predictions <- predict(model14, newdata = test_data2)

ggplot() +
  geom_point(aes(x = predictions, y = test_data2$Y1), color = "blue") +
  geom_abline(intercept = 0, slope = 1, color = "red")


