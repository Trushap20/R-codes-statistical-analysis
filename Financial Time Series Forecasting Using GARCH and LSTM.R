 
#  Financial Time Series Forecasting Using GARCH And LSTM #                                          #



library(neuralnet)
library(Rcpp)
library(RSNNS)
library(fGarch) 
library(keras)
library(tensorflow)
library(ggplot2)
suppressMessages(devtools::source_gist(filename = "helper_funcs.R",
                                       id = "https://gist.github.com/Letmode/8c842b722ab31210ad6df64ef786a8c2"))

############### DATA ############################

data =read.csv("^IXIC.csv")
X= data[,5][data[,5]>0]  
x_num <- as.numeric(X)
X <-na.omit(x_num)
Ret=diff(log(X))
Ret=Ret-mean(Ret)
Y0=Ret^2 
n0=length(Y0)
Year = seq(2015, 2021 + 1, length.out = n0)
par(mfrow=c(2,2))
plot(Year, Y0, type="l", ylab="")
title("Squared IXIC retruns")
n.te=250
n.in=n0-n.te

############### GARCH and FFNN GARCH model ###################

Year.te=Year[(n0-n.te+1):n0]
Y0.tr=Y0[1:n.in]
Y.te0=Y0[(n.in+1):n0]
Ret.tr=Ret[1:(n0-n.te)]
Ret.te=Ret[(n.in+1):n0]
Ret.fc=c(Ret.tr[n.in], Ret.te[1:(n.te-1)])

pmax = qmax = 1
BIC_GARCH = matrix(0, pmax, qmax + 1)
for (i in 1:pmax) {
  for (j in 0:qmax) {
    spec <- rugarch::ugarchspec(variance.model = list(model = "sGARCH", garchOrder = c(i, j)),
                                mean.model = list(armaOrder = c(0, 0)),
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
GARCH_fc = rugarch::ugarchforecast(GARCH, n.ahead = 1, n.roll = n.te - 1)
sig2.tr = as.numeric(rugarch::sigma(GARCH))^2
sig2.te = as.numeric(rugarch::sigma(GARCH_fc))^2

ASE.ga.tr=mean((Y0.tr-sig2.tr)^2)
ASE.ga.tr   

ASE.ga.te=mean((Y.te0-sig2.te)^2)
ASE.ga.te   

AAE.ga.tr=mean(abs(Y0.tr-sig2.tr))
AAE.ga.tr   

AAE.ga.te=mean(abs(Y.te0-sig2.te))
AAE.ga.te   

cor(Y0.tr, sig2.tr)   
cor(Y.te0, sig2.te)  

matplot(Year.te, cbind(Y.te0, sig2.te), type="ll", xlab="Year", ylab="")
title("Test series & GARCH forecasts")

####FFNN-GARCH ####
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

X.tr=X[1:(n.in-p),]
Y.tr=Y[1:(n.in-p)]
X.te=X[(n.in-p+1):(n0-p),]
Y.te=Y[(n.in-p+1):(n0-p)]

##fit NN models and select the best one(s)
nd=8
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

#### best model selected #####
i=2
j=3
seedn=1200+i*10+j
set.seed(seedn)
modelNG <- mlp(X.tr, Y.tr,
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

ASE.NNgarch.tr = mean((Y0.tr-sig2.tr)^2)
ASE.NNgarch.te = mean((Y.te0-sig2.te)^2)
AAE.NNgarch.tr = mean(abs(Y0.tr-sig2.tr))
AAE.NNgarch.te = mean(abs(Y.te0-sig2.te))
CCP.NNgarch.tr = cor(Y0.tr, sig2.tr)    
CCP.NNgarch.te = cor(Y.te0, sig2.te)                     
results_GARCH = c(ASE.NNgarch.te, AAE.NNgarch.te, CCP.NNgarch.te)
results_all = rbind(results_NN, results_GARCH)
colnames(results_all) = c("ASE", "AAE", "CCP")
rownames(results_all) = c("NN", "GARCH")
results_all

############### GARCH LSTM ##################

use_condaenv("r-tensorflow")

set_random_seed(10000)

p = 1
q = 1

nndata = dataprep_nn_garch(Ret, n.in, p.max = p, q.max = q, fixed = TRUE)
nndata$m_garch@fit$matcoef

X_tr = nndata$Xtrain
X_te = nndata$Xtest
Y_tr = nndata$Ytrain
Y_te = nndata$Ytest
Y0_te = nndata$Y0test

dim(X_tr) = c(dim(X_tr), 1) 

pq = p + q

model <- keras_model_sequential() %>%
  layer_lstm(units = 1,
             batch_input_shape = c(1, pq, 1),
             stateful = TRUE,
             return_sequences = TRUE) %>%
  layer_lstm(units = 2,
             stateful = TRUE) %>%
  layer_dense(units = 1, activation = "relu")

model %>%
  compile(loss = 'mse', optimizer = 'adam', metrics = 'mae')

summary(model)

model %>% fit(
  x = X_tr,
  y = Y_tr,
  batch_size = 1,
  epochs = 60,
  verbose = 1,
  shuffle = FALSE
)

X_pred = array(X_te, dim = c(dim(X_te), 1))
pred_out <- model %>% 
  predict(X_pred, batch_size = 1)

model %>% evaluate(X_tr, Y_tr, batch_size = 1)
model %>% evaluate(X_pred, Y_te, batch_size = 1) 

pred_out = pred_out * nndata$Ydiff + nndata$Ymin

ASE_LSTM <-ASE_calc(Y0_te, pred_out)
ASE_GARCH<-ASE_calc(Y0_te, nndata$Ytest_garch)
AAE_LSTM<-AAE_calc(Y0_te, pred_out)
AAE_GARCH<-AAE_calc(Y0_te, nndata$Ytest_garch)
CCP_LSTM<-cor(Y0_te, pred_out)
CCP_GARCH<-cor(Y0_te, nndata$Ytest_garch)

eval_crits1 = data.frame("ASE" = c(ASE_LSTM, ASE_GARCH), "AAE" = c(AAE_LSTM, AAE_GARCH), 
                        "CCP" = c(CCP_LSTM, CCP_GARCH))
rownames(eval_crits1) = c("LSTM", "GARCH")
eval_crits1

Year.og = Year
Year.te = Year[(n.in+ 1):n0]
df_og <- as.data.frame(cbind(Year.og, Y0))
df_fcs <- as.data.frame(cbind(Year.te, pred_out, Y0_te, nndata$Ytest_garch))
colnames(df_fcs) <- c("Year", "pred.te", "real_data", "pred.arma")

plot_data = ggplot(df_og) +
  geom_line(aes(x = Year.og, y = Y0), color = "black") +
  labs(title = "IXIC - squared return series", y = "", x = "Year") +
  theme(legend.text = element_text(size = 10), plot.title = element_text(size = 10),
        axis.title = element_text(size = 10))

plot_fcs = ggplot(df_fcs) +
  geom_line(aes(x = Year, y = real_data, color = "og")) +
  geom_line(aes(x = Year, y = pred.te, color = "lstm")) +
  geom_line(aes(x = Year, y = pred.arma, color = "arma")) +
  labs(title = "LSTM-predictions", y = "") +
  scale_color_manual("", values = c("og" = "black", "lstm" = "red", "arma" = "blue"),
                     labels = c("og" = "Test data", "lstm" = "LSTM-preds", "arma" = "GARCH-preds")) +
  theme(legend.text = element_text(size = 10), plot.title = element_text(size = 10),
        axis.title = element_text(size = 10), legend.position = c(0.85, 0.85))

ggpubr::ggarrange(plot_data, plot_fcs, ncol = 1)
ggsave("IXIC.pdf", height = 8.2, width = 12.2, dpi = 1200)

################ Hybrid LSTM ####################

use_condaenv("r-tensorflow")

set_random_seed(10000)

# conditional variance with GARCH(1, 1)
p = 1
q = 1

nndata = dataprep_nn_garch(Ret, n.in, p.max = p, q.max = q, fixed = TRUE)
nndata$m_garch@fit$matcoef

Y_te = nndata$Ytest
Y0_te = nndata$Y0test 
ht_tr = nndata$Ytrain_garch
ht_te = nndata$Ytest_garch
ht = c(ht_tr, ht_te)
eps_t = Ret / sqrt(ht) 


p = 8 # approximate GARCH(1,1) with ARCH(8)
q = 0
hybriddata = dataprep_nn_garch(eps_t, n.in, p.max = p, q.max = q, fixed = TRUE)
X_tr = hybriddata$Xtrain
X_te = hybriddata$Xtest
Y_tr = hybriddata$Ytrain
Y_te = hybriddata$Ytest


dim(X_tr) = c(dim(X_tr), 1)

pq = p + q

model <- keras_model_sequential() %>%
  layer_lstm(units = 1,
             batch_input_shape = c(1, pq, 1),
             stateful = TRUE,
             return_sequences = TRUE) %>%
  layer_lstm(units = 2,
             stateful = TRUE) %>%
  layer_dense(units = 1, activation = "relu")

model %>%
  compile(loss = 'mse', optimizer = 'adam', metrics = 'mae')

summary(model)

model %>% fit(
  x = X_tr,
  y = Y_tr,
  batch_size = 1,
  epochs = 60,
  verbose = 1,
  shuffle = FALSE
)

X_pred = array(X_te, dim = c(dim(X_te), 1))
pred_out <- model %>% 
  predict(X_pred, batch_size = 1)

pred_out = pred_out * hybriddata$Ydiff + hybriddata$Ymin 
pred_out = pred_out * ht_te 

ASE_HLSTM <-ASE_calc(Y0_te, pred_out)
ASE_GARCH<-ASE_calc(Y0_te, nndata$Ytest_garch)
AAE_HLSTM<-AAE_calc(Y0_te, pred_out)
AAE_GARCH<-AAE_calc(Y0_te, nndata$Ytest_garch)
CCP_HLSTM<-cor(Y0_te, pred_out)
CCP_GARCH<-cor(Y0_te, nndata$Ytest_garch)
eval_crits = data.frame("ASE" = c(ASE_GARCH,ASE_HLSTM), "AAE" = c(AAE_GARCH, AAE_HLSTM), 
                        "CCP" = c(CCP_GARCH, CCP_HLSTM))
rownames(eval_crits) = c("GARCH", "HLSTM")
eval_crits
Year.og = Year
Year.te = Year[(n.in + 1):n0]
df_og <- as.data.frame(cbind(Year.og, Y0))
df_fcs <- as.data.frame(cbind(Year.te, pred_out, Y0_te, nndata$Ytest_garch))
colnames(df_fcs) <- c("Year", "pred.te", "real_data", "pred.arma")

plot_data = ggplot(df_og) + geom_line(aes(x = Year.og, y = Y0), color = "black") + labs(title = "IXIC - squared return series", y = "", x = "Year") +  theme(legend.text = element_text(size = 10), plot.title = element_text(size = 10), axis.title = element_text(size = 10))


plot_fcs = ggplot(df_fcs) + geom_line(aes(x = Year, y = real_data, color = "og")) + geom_line(aes(x = Year, y = pred.te, color = "lstm")) + geom_line(aes(x = Year, y = pred.arma, color = "arma")) + labs(title = "Hybrid GARCH-LSTM-predictions", y = "") + scale_color_manual("", values = c("og" = "black", "lstm" = "red", "arma" = "blue"), labels = c("og" = "Test data", "lstm" = "Hybrid LSTM-preds", "arma" = "GARCH-preds")) + theme(legend.text = element_text(size = 10), plot.title = element_text(size = 10), axis.title = element_text(size = 10), legend.position = c(0.85, 0.85))

ggpubr::ggarrange(plot_data, plot_fcs, ncol = 1)
ggsave("IXIC.pdf", height = 8.2, width = 12.2, dpi = 1200)

################ all models ##############

eval_crit = data.frame("ASE" = c( ASE_GARCH, ASE.nn.te, ASE_LSTM, ASE_HLSTM), "AAE" = c( AAE_GARCH, AAE.nn.te, AAE_LSTM, AAE_HLSTM), "CCP" = c( CCP_GARCH,CCP.nn.te, CCP_LSTM, CCP_HLSTM))

rownames(eval_crit) = c( "GARCH","FFNN-GARCH","LSTM", "HLSTM")
eval_crit

