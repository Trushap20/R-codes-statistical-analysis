#  Deep Neural Networks and its application using R #


data<- read.csv(file="Housing.csv")
library(dplyr)
library(neuralnet)
library(Metrics)
DataFrame<- na.omit(data)
DataFrame<-select(DataFrame,-c(6,7,8,9,10,12,13))
is.na(DataFrame) <- sapply(DataFrame, is.infinite)

max_data <- apply(DataFrame, 2, max) 
min_data <- apply(DataFrame, 2, min)
data_scaled <- scale(DataFrame,center = min_data, scale = max_data - min_data) 
index = sample(1:nrow(DataFrame),round(0.70*nrow(DataFrame)))
train_data <- as.data.frame(data_scaled[index,])  
test_data <- as.data.frame(data_scaled[-index,])

###NN-model with one hidden layer#####

set.seed(12346)
n = names(DataFrame) 
f = as.formula(paste("price ~", paste(n[!n %in% "price"], collapse = " + ")))
Housing_nn = neuralnet(f, data = train_data, hidden = 5 )
plot(Housing_nn, rep = "best")

predict_housing<- predict(Housing_nn,test_data[,1:5])

predict_housing_start <- predict_housing*(max_data[6]-min_data[6])+min_data[6]

test_start <- as.data.frame((test_data$price)*(max_data[6]-min_data[6])+min_data[6])

MSE.housing_nn <- sum((test_start - predict_housing_start)^2)/nrow(test_start)
MSE.housing_nn 

mse(test_start[[1]], predict_housing_start)

cor.net_data = cor(predict_housing_start,test_start )
cor.net_data 

AAE.housing_nn <- sum(abs(test_start - predict_housing_start))/nrow(test_start)
AAE.housing_nn

mae(test_start[[1]],predict_housing_start)

#####NN-model with two hidden layers####

set.seed(3857)

n = names(DataFrame) 
f = as.formula(paste("price ~", paste(n[!n %in% "price"], collapse = " + ")))
Housing_nn2 = neuralnet(f, data = train_data, hidden = c(5,3))
plot(Housing_nn2, rep = "best")

predict_housing2<- predict(Housing_nn2,test_data[,1:5])

predict_housing_start2 <- predict_housing2*(max_data[6]-min_data[6])+min_data[6]

test_start2 <- as.data.frame((test_data$price)*(max_data[6]-min_data[6])+min_data[6])

MSE.housing_nn2 <- sum((test_start2 - predict_housing_start2)^2)/nrow(test_start2)
MSE.housing_nn2 

mse(test_start2[[1]], predict_housing_start2)

cor.net_data2 = cor(predict_housing_start2,test_start2 )
cor.net_data2

AAE.housing_nn2 <- sum(abs(test_start2 - predict_housing_start2))/nrow(test_start2)
AAE.housing_nn2

mae(test_start2[[1]],predict_housing_start2)


###Multi.linear regression ### 

Regression_Model <- lm(price~., data = train_data)
summary(Regression_Model)
predict_lm <- predict(Regression_Model,test_data)
test_start <- as.data.frame((test_data$price)*(max_data[6]-min_data[6])+min_data[6])

MSE.lm <- sum((predict_lm - test_data$price)^2)/nrow(test_data)
mse(test_start[[1]], predict_lm)

cor.lm = cor(predict_lm,test_data$price)
cor.lm

###overall results

MSE.housing_nn; MSE.housing_nn2; MSE.lm
cor.net_data[[1]]; cor.net_data2[[1]]; cor.lm



####################PART 3############################

install.packages("neuralnet")
install.packages("RSNNS")
install.packages("openxlsx")
install.packages("rnn")

library(neuralnet)
library(Rcpp)
library(rnn)
library(RSNNS)
library(longmemo)
library(fGarch)
library(openxlsx)
library(dplyr)

####Example 1####
## Fitting GARCH model using the traditional AR model.

X0 <- read.csv("2FDX .csv")

X = X0[,5][X0[,5]>0]   

Ret=diff(log(X))
Ret=Ret-mean(Ret)
Y0=Ret^2  
n0=length(Y0)
Year=seq(2016, 2021 + 2/12, length.out = n0)

plot(Year, Y0, type="l", ylab="")
title("Squared FDX returns")

#### Fitting GARCH with the traditional AR model

n.te=250
n.in=n0-n.te
Year.te=Year[(n0-n.te+1):n0]
Y0.tr=Y0[1:n.in]
Y.te0=Y0[(n.in+1):n0]

Ret.tr=Ret[1:(n0-n.te)]
Ret.te=Ret[(n.in+1):n0]
Ret.fc=c(Ret.tr[n.in], Ret.te[1:(n.te-1)])
pmax = qmax = 2
BIC_GARCH = matrix(0, pmax, qmax + 1)
for (i in 1:pmax) {
  for (j in 0:qmax) {
    spec <- rugarch::ugarchspec(variance.model = list(model = "sGARCH", garchOrder = c(i, j)),
                                mean.model = list(armaOrder = c(0, 0), include.mean = FALSE),
                                distribution.model = "std")
    GARCHij = rugarch::ugarchfit(spec = spec, data = c(Ret.tr, Ret.te), out.sample = 250)
    BIC_GARCH[i, j + 1] = rugarch::infocriteria(GARCHij)["Bayes", ]
  }
}
p = which(BIC_GARCH == min(BIC_GARCH), arr.ind = TRUE)[1]
q = which(BIC_GARCH == min(BIC_GARCH), arr.ind = TRUE)[2] - 1
spec <- rugarch::ugarchspec(variance.model = list(model = "sGARCH", garchOrder = c(p, q)),
                            mean.model = list(armaOrder = c(0, 0)),
                            distribution.model = "std")
GARCH = rugarch::ugarchfit(spec = spec, data = c(Ret.tr, Ret.te), out.sample = 250)
GARCH@fit$matcoef


## A FFNN-GARCH(p, q) model

GARCH_fc = rugarch::ugarchforecast(GARCH, n.ahead = 1, n.roll = n.te - 1)
sig2.tr = as.numeric(rugarch::sigma(GARCH))^2
sig2.te = as.numeric(rugarch::sigma(GARCH_fc))^2
matplot(Year.te, cbind(Y.te0, sig2.te), type="ll", xlab="Year", ylab="")
title("Test series & GARCH forecasts")

p=1
q=1
Y0.s=min(Y0[(p+1):n0])
Y0.d=max(Y0[(p+1):n0])-Y0.s
X=matrix(0, (n0-p), p+q) 
Y=(Y0[(p+1):n0]-Y0.s)/Y0.d

for(i in 1:p){
  X[,i]=(Y0[(p+1-i):(n0-i)]- min(Y0[(p+1-i):(n0-i)]))/(max(Y0[(p+1-i):(n0-i)])- 
                                                         min(Y0[(p+1-i):(n0-i)]))
}

resid=c(sig2.tr, sig2.te)
residi=(resid-min(resid))/(max(resid)-min(resid))
for(i in 1:q){
  X[,p+i]=residi[(p+1-i):(n0-i)]
}
### creating training and test set
X.tr=X[1:(n.in-p),]
Y.tr=Y[1:(n.in-p)]
X.te=X[(n.in-p+1):(n0-p),]
Y.te=Y[(n.in-p+1):(n0-p)]

##fit NN models and select the best one(s)
nd=5
ASE0=0.01 
ASE_Cor=matrix(ASE0, 6, nd^2)
for(i in 1:nd){
  for(j in 1:nd){
    seedn=1200+i*10+j
    set.seed(seedn)
    model <- mlp(X.tr, Y.tr,
                 size = c(i, j), learnFuncParams = c(0.1), maxit = 4000,
                 linOut = FALSE)
    Y.tr.nn <- predict(model, X.tr)
    Y.te.nn <- predict(model, X.te)
    ASE.nn.in=mean(((Y.tr-Y.tr.nn)*Y0.d)^2)
    ASE.nn.te=mean((Y.te0-(Y.te.nn*Y0.d+Y0.s))^2)
    AAE.nn.in=mean(abs((Y.tr-Y.tr.nn)*Y0.d))
    AAE.nn.te=mean(abs(Y.te0-(Y.te.nn*Y0.d+Y0.s)))
    CCP.nn.in=cor(Y.tr, Y.tr.nn)
    CCP.nn.te=cor(Y.te, Y.te.nn)
    ASE_Cor[1, (i-1)*p+j]=ASE.nn.in
    ASE_Cor[2, (i-1)*p+j]=ASE.nn.te
    ASE_Cor[3, (i-1)*p+j]=AAE.nn.in
    ASE_Cor[4, (i-1)*p+j]=AAE.nn.te
    ASE_Cor[5, (i-1)*p+j]=CCP.nn.in
    ASE_Cor[6, (i-1)*p+j]=CCP.nn.te
  }
}

mat_pos = c(apply(ASE_Cor[1:4, ], 1, which.min), apply(ASE_Cor[5:6, ], 1, which.max))
crit_val = round(c(apply(ASE_Cor[1:4, ], 1, min), apply(ASE_Cor[5:6, ], 1, max)), 6)
nd_hd1 = ifelse(mat_pos %% nd == 0, mat_pos / nd, trunc(mat_pos / nd + 1))

nd_hd2 = ifelse(mat_pos %% nd== 0, 4, mat_pos %% nd)
results = data.frame(mat_pos, crit_val, nd_hd1, nd_hd2)
rownames(results) = c("ASE_tr", "ASE_te", "AAE_tr", "AAE_te", "CCP_tr", "CCP_te")
results


#### Results of the NN selected by the ASE, AAE and CCP
i=2
j=2
seedn=1200+i*10+j
set.seed(seedn)
model <- mlp(X.tr, Y.tr,
             size = c(i, j), learnFuncParams = c(0.1), maxit = 4000,
             linOut = FALSE)
Y.tr.nn1 <- predict(model, X.tr)
Y.te.nn1 <- predict(model, X.te)
ASE.nn.in=mean(((Y.tr-Y.tr.nn1)*Y0.d)^2)
ASE.nn.te=mean((Y.te0-(Y.te.nn1*Y0.d+Y0.s))^2)
AAE.nn.in=mean(abs((Y.tr-Y.tr.nn1)*Y0.d))
AAE.nn.te=mean(abs(Y.te0-(Y.te.nn1*Y0.d+Y0.s)))
CCP.nn.in=cor(Y.tr, Y.tr.nn1)
CCP.nn.te=cor(Y.te, Y.te.nn1)
results_NN = data.frame(ASE.nn.te, AAE.nn.te, CCP.nn.te)

#### the error criteria
ASE.garch.tr = mean((Y0.tr-sig2.tr)^2)
ASE.garch.te = mean((Y.te0-sig2.te)^2)
AAE.garch.tr = mean(abs(Y0.tr-sig2.tr))
AAE.garch.te = mean(abs(Y.te0-sig2.te))
CCP.garch.tr = cor(Y0.tr, sig2.tr)    
CCP.garch.te = cor(Y.te0, sig2.te)                     
results_GARCH = c(ASE.garch.te, AAE.garch.te, CCP.garch.te)
results_all = rbind(results_NN, results_GARCH)
colnames(results_all) = c("ASE", "AAE", "CCP")
rownames(results_all) = c("NN", "GARCH")
results_all



#####Example 2####

X0= read.csv("2FDX .csv") 

X=X0[,5][X0[,5]>0]
n.out=250 

Ret=diff(log(X)) ## the log return
n=length(Ret) ## the length of the return
Ret.in=Ret[1:(n-n.out)] ## training set (of log return)
Ret=Ret-mean(Ret.in) ## subtract the mean of the training set
Ret.in=Ret[1:(n-n.out)] ## training set (demeaned log return)
Ret.out=Ret[(n-n.out+1):n] ## testing set (demeaned log return)
n.in=length(Ret.in)

Ret.fc=c(Ret.in[n.in], Ret.out[1:(n.out-1)])

#forecast volatility with GARCH(1,1)
CSOG=garchFit(formula=~garch(1,1), data=Ret.in, cond.dis="std", trace=FALSE)


sigma.CSOG=CSOG@sigma.t # obtain the conditional volatility
a0=as.numeric(CSOG@fit$coef[2])
a1=as.numeric(CSOG@fit$coef[3])
b1=as.numeric(CSOG@fit$coef[4])
df=as.numeric(CSOG@fit$coef[5])
cor(Ret.in**2, sigma.CSOG**2) 

sigma.CSOG.fc=sqrt(a0+a1*Ret.fc[1]**2+b1*sigma.CSOG[n.in]**2)
for(i in 2:n.out){
  sigma.fc.i=sqrt(a0+a1*Ret.fc[i]**2+b1*sigma.CSOG.fc[i-1]**2)
  sigma.CSOG.fc=c(sigma.CSOG.fc, sigma.fc.i)
}
cor(Ret.out**2, sigma.CSOG.fc**2)

#An NN-ARCH(p) model

p=4
Xnam <- paste0("V", 2:(p+1))
fmla <- as.formula(paste("V1~", paste(Xnam, collapse= "+")))

X=matrix(0, (n-p), (p+1))

X[, 1]=(Ret[(p+1):n]^2-min(Ret[(p+1):n]^2))/(max(Ret[(p+1):n]^2)-
                                               min(Ret[(p+1):n]^2))

for(i in 1:p){
  X[,(i+1)]=(Ret[(p+1-i):(n-i)]^2- min(Ret[(p+1-i):(n-i)]^2))/(max(Ret[(p+1-i):(n-i)]^2)-
                                                                 min(Ret[(p+1-i):(n-i)]^2))
}
X=as.data.frame(X)

X.tr=X[1:(n.in-p),] # training set
X.te=X[(n.in-p+1):(n-p),] # test set


##Modeling volatility with function neuralnet(), p = 5


set.seed(12345)
hid=c(1) # one hidden layer with one hidden neuron
Model <- neuralnet(fmla, data=X.tr, hidden = hid)
plot(Model)

results.in <-  neuralnet::compute(Model, X.tr[,2:(p+1)]) # fit the model
predicted.in <- results.in$net.result 
Cor.in=cor(predicted.in, X.tr[,1]) 
Cor.in

results.out <-  neuralnet::compute(Model, X.te[,2:(p+1)])
predicted.out <- results.out$net.result
Cor.out=cor(predicted.out, X.te[,1]) 
Cor.out

##Plot the results from GARCH and NN-ARCH model

c.re=mean(Ret[(p+1):n.in]^2/predicted.in)
par(mfcol=c(4,2))
par(mar=c(2,2,2,2))
plot.ts(abs(Ret.out))
title("Test absolute returns")
plot.ts(sigma.CSOG.fc)
title("Rolling volatility forecasts by GARCH")
plot.ts(Ret.out/sigma.CSOG.fc)
title("Standardized test returns by GARCH")
acf((Ret.out/sigma.CSOG.fc)^2, main="")
title("acf of sq. st. test returns by GARCH")
plot.ts((X.te[,1]*c.re)**0.5)
title("Recaled absolute test returns")
plot.ts((predicted.out*c.re)**0.5)
title("Rolling volatility forecasts by NN-ARCH")
plot.ts(Ret.out/(predicted.out*c.re)**0.5)

title("Standardized test returns by NN-ARCH")
acf(Ret.out^2/(predicted.out*c.re), main="")
title("acf of sq. st. test returns by NN-ARCH")

