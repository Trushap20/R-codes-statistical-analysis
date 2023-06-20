############################################################################################
#             W5453 : Advanced Time Series Analysis and Forecasting                        #
#                                                                                          #
#    Project 1: Study on the relationship between Japan Unemployment rate and GDP:         #
#                         VAR model Application in R.                                      #
#                                                                                          # 
#                   Group No : 24        Submitted by: Trusha Patel                        #    
############################################################################################


install.packages("deseats_0.0.0.10.zip", repos = NULL, type = "win.binary")
install.packages("deseats_0.0.0.10.tar.gz", repos = NULL, type = "source")
install.packages("deseats_0.0.0.10_no_compilation.tar.gz", repos = NULL, type = "source")

#### 

library(MTS)


#### Data

GDP=read.csv("gdp.csv", header=TRUE)
colnames(GDP)<-c("Date","Gdp")
GDP.log=log(GDP[,2])
n0=length(GDP.log)


Unem=read.csv("unem.csv", header=TRUE)
colnames(Unem)<-c("Date","Unem")
UNEM.m=Unem[,2]
n.m=length(UNEM.m)

gdp <- ts(GDP.log, start = c(1994, 1), freq = 4)
plot(gdp,type = "l",main="Log-real-GDP",xlab="Time", ylab="Log-Real GDP")
une <- ts(Unem$Unem, start = c(1994, 1), freq = 12)
plot(une, type = "l",main="Monthly Unemployment Rate",xlab="Time", ylab="Unemployment rate")

UNEM=1:n0
for(i in 1:n0){UNEM[i]=mean(UNEM.m[((i-1)*3+1):(i*3)])}

n=n0-11

GDP.log=GDP.log[1:n]
UNEM=UNEM[1:n]

GDP.D = diff(GDP.log)*100
UNEM.D=diff(UNEM)
z=cbind(GDP.D, UNEM.D) 
n=length(z[,1])   

### Correlation Plots

plot(UNEM.D,GDP.D)
plot(GDP.D,UNEM.D)

#### Cross correlation matrix

cor(cbind(GDP.D, UNEM.D))

#### Fit the VAR model

m1=VAR(z,0, output=F)
m1=VAR(z,1)
m2=VAR(z,2)
m3=VAR(z,3)
m4=VAR(z,4)
m5=VAR(z,5)
m6=VAR(z,6)
m7=VAR(z,7)
m8=VAR(z,8)
m9=VAR(z,9)
m10=VAR(z,10)

p.max=10
AIC=0:p.max
BIC=AIC
HQ=AIC
for(p in 0:p.max){m1=VAR(z,p, output=F)
AIC[p+1]=m1$aic
BIC[p+1]=m1$bic
HQ[p+1]=m1$hq
}
p.all=0:p.max
matplot(p.all, cbind(AIC, BIC, HQ), type="lll")
title("The three criteria, p=0, 1, ..., 10")
abline(v=1)

library(vars)

model <- VAR(z, p = 1, type = "const", season = NULL, exog = NULL)
summary(model)

### Granger- Causality test

causality(model, cause = NULL, vcov.=NULL, boot=FALSE, boot.runs=100)

### Forecast at 95% confidence interval

predictions <- predict(model, n.ahead = 8, ci = 0.95)
plot(predictions, names = "UNEM.D",main="Forecasting of Unemployment Rate series")
plot(predictions, names = "GDP.D",main="Forecasting of GDP series")

