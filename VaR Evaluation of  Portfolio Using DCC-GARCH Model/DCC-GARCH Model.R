############################################################################################
#             W5453 : Advanced Time Series Analysis and Forecasting                        #
#                                                                                          #
#    Project 2: Value-at-Risk Evaluation of Four Financial Stocks Portfolio Using          #  
#                                 DCC-GARCH Model.                                         #                                   #
#                                                                                          #
#                                                                                          # 
#                   Group No : 24        Submitted by: Trusha Patel                        #    
############################################################################################


library(rugarch)
library(rmgarch)         # for fitting DCC models
library(purrr)           # for functionalized loops
library(zoo)             # for time series handling
library(magrittr)        # for the pipe operator
library(parallel)        # for parallel programming
library(R.utils)         # for dimension reduction from array

Sys.setlocale("LC_TIME", "English")  

datafiles <- c(
  SAP = "SAP.DE.csv",
  SIE = "SIE.DE.csv",
  DTE = "DTE.DE.csv",
  ALV = "ALV.DE.csv"
)

datasets <- datafiles %>%
  map(~ read.csv(.x, header = TRUE))


dates <- datasets$SAP$Date  
dates <- as.Date(dates)

close <- datasets %>%                 
  map_df(~ .x$Close) %>%               
  as.matrix() %>%                      
  zoo(order.by = dates)               

plot(close, xlab = "Year", main = "Daily closing prices for four companies, Jan 2010 - Dec 2022")

rt <- diff(log(close))         

plot(rt, xlab = "Year", main = "Daily log-returns for four companies, Jan 2010 - Dec 2022")

correlation <-cor(rt)

############################## Model specification and fit #######################################

spec_SAP <- ugarchspec(variance.model = list(model = "apARCH", garchOrder = c(1, 1)), mean.model = list(armaOrder = c(0, 0)), distribution.model = "norm")
spec_SIE <- ugarchspec(variance.model = list(model = "sGARCH", garchOrder = c(1, 1)), mean.model = list(armaOrder = c(0, 0)), distribution.model = "norm")
spec_DTE <- ugarchspec(variance.model = list(model = "eGARCH", garchOrder = c(1, 1)), mean.model = list(armaOrder = c(0, 0)), distribution.model = "norm")
spec_ALV <- ugarchspec(variance.model = list(model = "sGARCH", garchOrder = c(1, 1)), mean.model = list(armaOrder = c(0, 0)), distribution.model = "norm")

fit1 <- ugarchfit(spec=spec_SAP, data=rt$SAP)
fit2 <- ugarchfit(spec=spec_SIE, data=rt$SIE)
fit3 <- ugarchfit(spec=spec_DTE, data=rt$DTE)
fit4 <- ugarchfit(spec=spec_ALV, data=rt$ALV)

################################# DCC GARCH ######################################################

mspec <- multispec(list(spec_SAP, spec_SIE, spec_DTE, spec_ALV))

dccspec <- dccspec(uspec = mspec, 
                   dccOrder = c(1, 1),model = "DCC", distribution = "mvnorm")

cl <- makePSOCKcluster(detectCores() - 1) 

multfit <- multifit(mspec, rt, out.sample = 250, cluster = cl)
dccfit <- dccfit(dccspec, data = rt, out.sample = 250,fit = multfit,cluster = cl)

dccfit

#one-step rolling forecasts for out-of-sample data

fc <- dccforecast(dccfit, n.ahead = 1, n.roll = 250 - 1, cluster = cl)

stopCluster(cl)       

dates_test <- tail(dates, 250)     # Out-of-sample dates

sigma_fc <- fc %>%
  sigma() %>%                             
  wrap(map = list(NA, 3)) %>%                          
  t()                                     
colnames(sigma_fc) <- names(rt)    
sigma_fc <- sigma_fc %>%                 
  zoo(order.by = dates_test)

rcor_fc <- rcor(fc, output = "matrix")      # Get conditional correlations 
head(rcor_fc, 2)

##### Reshape to a multivariate time series object

rcor_fc <- as.data.frame(do.call(rbind, rcor_fc)) %>%
  as.matrix() %>%
  zoo(order.by = dates_test)

rcov_fc <- rcov(fc, output = "matrix")      # Get conditional variances
head(rcov_fc)


rcov_fc <- as.data.frame(do.call(rbind, rcov_fc)) %>%
  as.matrix() %>%
  zoo(order.by = dates_test)              


################################### plot ###################################

##### Plot of forecasted conditional standard deviations

plot(sigma_fc, main = "Out-of-sample one-step rolling forecasts of the conditional standard deviations",
     xlab = "Month")

#### Plot of the forecasted conditional correlations

plot(rcor_fc, 
     main = "Out-of-sample one-step rolling forecasts of the conditional correlations",
     xlab = "Month")

#### Plot of the forecasted conditional covariances

plot(rcov_fc, 
     main = "Out-of-sample one-step rolling forecasts of the conditional covariances",
     xlab = "Month")


################################# VaR Forecast  ###########################################

weights <- c(0.25, 0.30, 0.25, 0.20)     

rcov_fc_array <- array(unlist(rcov(fc)), dim = c(4, 4, 250))    
means_fc <- t(wrap(fitted(fc), map = list(NA, 3)))              

##### Get weighted margin of elliptical distribution

marg <- wmargin(distribution = "mvnorm", weights = weights, mean = means_fc, 
                Sigma = rcov_fc_array, shape = rshape(fc), skew = rskew(fc))

#### Get individual distribution name from joint distribution

dm <- switch("mvnorm",
             mvnorm = "norm",
             mvlaplace = "ged",
             mvt = "std")

##### Get the 1% quantile; this is your VaR

portfolio <- apply(tail(rt, 250), MARGIN = 1, FUN = function(x){weighted.mean(x, w = weights)}) %>% zoo(order.by = dates_test)

q01 <- qdist(distribution = dm, p = 0.01, mu = marg[,1], sigma = marg[,2], lambda = -0.5,
             skew = marg[,3], shape = marg[,4]) %>%
  zoo(order.by = dates_test)

par(mfrow=c(2,1))
plot(portfolio, xlab = "Month", ylab = "Portfolio return", main = "Returns of the Portfolio")

plot(q01, xlab = "Month", ylab = "VaR", main = "The 1%-VaR for the Portfolio")

violations <- portfolio < q01
violations <- time(violations)[violations]

matplot(time(q01), cbind(portfolio, q01), type = "hl", xlab = "Month", ylab = "Test return",
        main = "Test portfolio returns with 1%-VaR and violations")
abline(h = 0, col = "black")

points(violations, q01[violations], col = "forestgreen", pch = 13, cex = 1.5)

