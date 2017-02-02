setwd("~/4th Year/project/model/beta model")

W = read.csv("../../data/csv/adjacency.csv")
W$X = NULL
W_n = floor(sum(W)/2)

observed = read.csv("../../data/csv/simulated.csv")
observed$X = NULL

expected = read.csv("../../data/csv/expected.csv")
expected$X = NULL
log_expected = c(log(expected))

numRegions = dim(observed)[1]
numRegions = nrow(observed)
nt = dim(observed)[2]

alpha = 0.9

model_data <- list(numRegions = numRegions,       
                   nt = nt,    
                   observed = as.matrix(observed),          
                   log_expected = log_expected$x,              
                   W_n = W_n, 
                   W = as.matrix(W),
                   alpha = alpha)

library(rstan)

model <- stan_model(file = "beta_model.stan")

model_advi <- vb(model, data=model_data)

model_nuts <- sampling(model, data = model_data, chains = 3, iter = 100)

model_data <- extract(model_advi)
result_unsorted <- colMeans(model_data$prop_unusual)
result <- sort(result_unsorted, index.return=TRUE)
