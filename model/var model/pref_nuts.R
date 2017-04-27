setwd("~/4th Year/project/model/var model")
W = read.csv("../../data/csv/adjacency.csv")
W$X = NULL
W_n = floor(sum(W)/2)

observed = read.csv("../../data/csv/simulated_spatial_corr.csv")
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

model <- stan_model(file = "var.stan")

#model_advi <- vb(model, data=model_data)

model_nuts <- sampling(model, data = model_data, chains = 1, iter = 2000)

model_data <- extract(model_nuts)
result_unsorted2 <- colMeans(model_data$indicator)
result2 <- sort(result_unsorted2, index.return=TRUE)
