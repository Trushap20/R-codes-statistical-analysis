
if (!require(devtools)) {
  install.packages("devtools", repos = "http://cran.us.r-project.org")
}

devtools::source_gist(
  "https://gist.github.com/dschulz13/3d4fdd7a5cc1067520e38c406207fe37", 
  local = globalenv(), verbose = FALSE, quiet = TRUE,
  filename = "P2_3_4_Setup.R")


## We have used CAC 40 index (trading symbol is ^FCHI).
## The time period selected is from January 2016 to December 2021.


#-------------------------------- Part (a) -------------------------------------#

pkgs <- c("fGarch", "zoo", "ggplot2", "ggpubr")
for (pkg in pkgs) {
  if (!require(pkg, character.only = TRUE)) {
    install.packages(pkg, repos = "http://cran.us.r-project.org")
    require(pkg, character.only = TRUE)
  }
}

#Preparing the data set.

library(vroom)

data <- vroom("^FCHI.csv")
data <-na.omit(data)
data[complete.cases(data),]
FCHI <- data$Close
Date <- data$Date
Date <- as.Date(Date, format = "%Y-%m-%d")

#Transforming closing prices into time series.

FCHI <- zoo(FCHI, order.by = Date)

#Getting the number of observations for the closing price and the return series

n_cp <- length(FCHI)   # Number of closing price observations  
n_ret <- n_cp - 1      # Number of observations for the return series

#Computing log-returns.

FCHI.Returns <- diff(log(FCHI))        # The returns as a time series object
FCHI.Ret_v <- coredata(FCHI.Returns)   # The returns as a simple numeric vector
plot(FCHI.Returns)

#-------------------------------- Part (b) -------------------------------------#

bw <- 35

RetDiff <- matrix(0, 3, n_cp - 2 * bw+1)
DV <- c((1:bw) * 0, (1:bw)^0)           # DV divides the data into to a left & a right piece. 
uh <- (-(bw - 1):0) / (bw - 0.5)        # This is the variable in the kernel. 
mu <- 2                                 # The chosen kernel, mu = 2 is suggested.
KWh <- (1 + uh)^mu * (-uh)^(mu - 1)     # Left half weights following Mueller/Wang 1994
KW <- c(KWh, rev(KWh))                  # The whole weights, continuous, if mu >= 2

for(i in (bw + 1):(n_cp - bw+1)) {
  Yi <- FCHI.Ret_v[(i - bw):(i + bw-1)]
  Mi <- lm(Yi ~ DV, weights = KW)             # The "lm" function with only a "DV" is simply used.
  RetDiff[1, i - bw] <- Mi$coefficients[1]    # The estimate for DV = 0
  RetDiff[2, i - bw] <- Mi$coefficients[2]    # The coefficient of DV, for the difference between the cases DV = 1 and DV = 0
}

RetDiff[3, ] <- RetDiff[1, ] + RetDiff[2, ]   # The estimates for DV = 1

# The minimum position of the difference is the estimated beginning of the COVID-19 crisis 

ns <- ((bw + 1):(n_cp - bw))[RetDiff[2, ] == min(RetDiff[2, ])] 
ns 
data[ns, ] 
data[ns + 1, ]

ne <- ((bw + 1):(n_cp - bw))[RetDiff[2, ] == max(RetDiff[2, ])] 
ne 
data[ne, ]
data[ne + 1, ]

#-------------------------------- Part (c) -------------------------------------#

# Fit a GARCH-t(1, 1) to the sub series before the COVID-19 crisis. 


FCHI.Ret1 <- FCHI.Returns[1:(ns-1)]      
FCHI.Ret1.GARCH <- garchFit(~ garch(1, 1), FCHI.Ret1, cond.dist = "std", trace = FALSE) 
FCHI.Ret1.GARCH

# Estimating the degrees of freedom.

df <- FCHI.Ret1.GARCH@fit$par[["shape"]]  
df # df=5.0973

## Get the fitted conditional volatilizes for the fitting period.

Vol.est <- FCHI.Ret1.GARCH@sigma.t

## Predict the conditional volatilizes for the remaining observation time points.

K <- n_cp - ns
GARCH.Pred <- predict(FCHI.Ret1.GARCH, n.ahead = K)
Vol.Pred <- GARCH.Pred$standardDeviation

# Combine the fitted in-sample volatilizes and the forecast.

Vol.all <- c(Vol.est, Vol.Pred)

# Obtain the t-quantiles.

alpha <- 0.001      
qt.L <- qt(alpha, df)
qt.L

# Analyze the extremely negative returns.

Date.Extreme.N <- Date[-1][FCHI.Ret_v < Vol.all * qt.L]  # Find the dates of the extremely negative returns.
Date.Extreme.N                         

round(FCHI.Returns[Date.Extreme.N], digits = 5) 

# Analyze the extremely positive returns.

qt.U <- -qt.L
qt.U
Date.Extreme.P <- Date[-1][FCHI.Ret_v > Vol.all * qt.U]  # Find the positions of the extremely negative returns.
Date.Extreme.P                            

round(FCHI.Returns[Date.Extreme.P], digits = 5)  

#-------------------------------- Part (d) -------------------------------------#

#Display results in plots using ggplot.
date_min <- Date[1]               
date_max <- tail(Date, 1)

# Plot 1: Price series

p1 <- autoplot.zoo(FCHI) +
  ylab("") +
  xlab("Year") +
  ggtitle("CAC 40 price series, January 2016 to December 2021") +
  geom_vline(xintercept = Date[ns + 1], color = 2) +
  geom_vline(xintercept = Date[ne + 1], color = 3) +
  xlim(date_min, date_max) +
  theme_bw() +
  theme(plot.title = element_text(hjust = 0.5))

# Plot 2: Return series

p2 <- autoplot.zoo(FCHI.Returns) +
  ylab("") +
  xlab("Year") +
  ggtitle("CAC 40 returns with 99.9% confidence bounds, January 2016 to December 2021") +
  geom_vline(xintercept = Date[ns + 1], color = 2) +
  geom_vline(xintercept = Date[ne + 1], color = 3) +
  
  geom_line(data = data.frame(x = Date[-1], y = Vol.all * qt.L), aes(x = x, y = y), color = 4) +
  geom_line(data = data.frame(x = Date[-1], y = Vol.all * qt.U), aes(x = x, y = y), color = 4) +
  xlim(date_min, date_max) +
  theme_bw() +
  theme(plot.title = element_text(hjust = 0.5))

#Plot 3: Differences between left- and right-hand side mean returns

p3 <- ggplot(data.frame(x = Date[(bw + 1):(n_cp - bw+1)], y = RetDiff[2, ]), aes(x = x, y = y)) +
  geom_line() +
  ggtitle("Differences of the left- and right-hand side mean returns") +
  geom_vline(xintercept = Date[ns + 1], color = 2) +
  geom_vline(xintercept = Date[ne + 1], color = 3) +
  xlim(date_min, date_max) +
  xlab("Year") +
  ylab("") +
  theme_bw() +
  theme(plot.title = element_text(hjust = 0.5))

#Combine the three ggplot2-plots into a single one

plot_crisis <- ggarrange(p1, p2, p3, ncol = 1, nrow = 3, align = "v")
plot_crisis

#-------------------------------- Part (e) -------------------------------------#

# Comparing the GARCH-t models for the pre- and post-crisis periods.

FCHI.Ret2 <- FCHI.Returns[(ne + 1):(n_cp - 1)]    # The post-crisis returns 
FCHI.Ret2.GARCH <- garchFit(~ garch(1, 1), FCHI.Ret2, cond.dist = "std", trace = FALSE)  # Fit a GARCH-t-model 

# Comparing the two GARCH models to each other.

FCHI.Ret1.GARCH@fit$matcoef

FCHI.Ret2.GARCH@fit$matcoef
#-------------------------------- Part (f) -------------------------------------#
return1  <- RetDiff[1,]
return2  <- RetDiff[3,]
plot(return1,type = "l",col = "red", xlab = "X", ylab = "Y",)

lines(return2, type = "l", col = "blue")

#============================= End of the Script ==============================#

