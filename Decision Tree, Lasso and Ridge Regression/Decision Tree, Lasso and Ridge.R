
############################################################################################
#             W5451 : Statistical Learning for Data Science with R and Python              #
#                                                                                          #
#                                                                                          #
#    Project 2: Application of Decision trees, Lasso and Ridge Regression on different     #
#               data set using R programming.                                              #
#                                                                                          # 
#                   Group No : 06       Submitted by: Trusha Patel   6872844               #    
############################################################################################

########################## Example 1 : Lasso and Ridge regression ##########################

library(glmnet)

Sportify <- read.csv("song_data.csv")
Sportify <- Sportify[-1]
head(Sportify)

y <- Sportify$song_popularity

set.seed(12)
train.size <- nrow(Sportify) * 0.6
train.ind <- sample(1:nrow(Sportify), train.size)
Sportify.train <- Sportify[train.ind,]
Sportify.test <- Sportify[-train.ind,]
lm.fit <- lm(song_popularity~., data = Sportify.train)
lm.pred  <- predict(lm.fit, Sportify.test)

pop.test <- y[-train.ind]
MSE.lm <- mean((pop.test - lm.pred)^2)
MSE.lm

pop.train <- y[train.ind]
Sportify.train.mat <- model.matrix(song_popularity~., Sportify.train)[,-1]
Sportify.test.mat <- model.matrix(song_popularity~., Sportify.test)[,-1]

grid = seq(0.01, 20, 0.01)

#### Ridge Regression

set.seed(109)
ridge.fit <- cv.glmnet(Sportify.train.mat, pop.train, aplha = 0, lambda = grid)

lambda.opt = ridge.fit$lambda.min
lambda.opt

ridge.pred = predict(ridge.fit, newx = Sportify.test.mat, s = lambda.opt)
MSE.ridge = mean((pop.test - ridge.pred)^2)
MSE.ridge

Sportify.mat = model.matrix(song_popularity~., Sportify)[, -1]
ridge.fit = glmnet(Sportify.mat,y, aplha = 0, lambda = grid)
predict(ridge.fit, s = lambda.opt, type = "coefficients")

###### Lasso

Sportify.mat = model.matrix(song_popularity~., Sportify)[, -1]
lasso.fit = glmnet(Sportify.mat, y, aplha = 1, lambda = grid)

set.seed(33)
lasso.fit <- cv.glmnet(Sportify.train.mat, pop.train, aplha = 1, lambda = grid)

lambda.opt = lasso.fit$lambda.min
lambda.opt

lasso.pred <- predict(lasso.fit, newx = Sportify.test.mat, s = lambda.opt )
MSE.lasso <- mean((pop.test - lasso.pred)^2)
MSE.lasso

Sportify.mat = model.matrix(song_popularity~., Sportify)[,-1]
lasso.fit = glmnet(Sportify.mat, y, aplha = 1, lambda = grid)
predict(lasso.fit, s = lambda.opt, type = "coefficients")


### All results 

MSE.all = data.frame(cbind(MSE.lm, MSE.ridge, MSE.lasso), row.names = "MSE")

colnames(MSE.all) = c("lm", "ridge", "lasso")
MSE.all

########################## Example 2 : Decision Tree ##############################

library(tree)
library(randomForest)
library(stats)

Poki<- read.csv("Pokemondata.csv")
Poki <- na.omit(Poki)
Poki$Legendary <- factor(Poki$Legendary, levels = c(TRUE, FALSE), labels = c("L","NL"))
names(Poki)
head(Poki)

###Decision tree

set.seed(06)
train = sample(1:nrow(Poki), 600)
test = Poki[-train,]
trainn = Poki[train,]

Poki.Tree = tree(Legendary~HP+Attack+Defense+SpAtk+SpDef+Speed, trainn)
summary(Poki.Tree)
Poki.Tree
plot(Poki.Tree)
text(Poki.Tree, pretty=0, cex = 0.8)

Poki.Tree.Pred<- predict(Poki.Tree,test,type = "class")
Legend.test <- Poki$Legendary[-train]
table(Poki.Tree.Pred,Legend.test)
mean(Poki.Tree.Pred == test$Legendary) 
mse <- mean((as.numeric(Poki.Tree.Pred) - as.numeric(Legend.test))^2)

###Tree pruning

set.seed(60)
cv.Poki <- cv.tree(Poki.Tree, FUN = prune.misclass)
names(cv.Poki)
cv.Poki

par(mfrow = c(1, 2))
plot(cv.Poki$size , cv.Poki$dev, type = "b")
plot(cv.Poki$k, cv.Poki$dev, type = "b")

###Plotting the results

prune.Poki <- prune.misclass(Poki.Tree , best = 5)
plot(prune.Poki)
text(prune.Poki, pretty = 0, cex = 0.8)

Prune.Tree.Pred<- predict(prune.Poki,test,type = "class")
table(Prune.Tree.Pred,Legend.test)
mean(Prune.Tree.Pred == test$Legendary)
mse1 <- mean((as.numeric(Prune.Tree.Pred) - as.numeric(Legend.test))^2)

###Bagging

bag.poki = randomForest(Legendary ~ HP + Attack + Defense + SpAtk + SpDef + Speed , data=Poki, subset=train, mtry=5, importance=TRUE)
bag.pred<- predict(bag.poki,test,type = "class")
table(bag.pred,Legend.test)
mean(bag.pred == test$Legendary)
mse2 <- mean((as.numeric(bag.pred) - as.numeric(Legend.test))^2)


###logistic regression model

log.fit <- glm(Legendary ~ HP + Attack + Defense + SpAtk + SpDef + Speed, data = trainn, family = binomial)
summary(log.fit)
log.prob <- predict(log.fit, test, type = "response")
log.pred <- rep(0, length(log.prob))
log.pred[log.prob > 0.5] = 1
table(log.pred, test$Legendary)

(11+182)/(11+4+3+182)   
(100-93.5) 

#############################################################################################################
