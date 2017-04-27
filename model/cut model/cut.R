setwd("~/4th Year/project/model/cut model")
general <- read.csv('general.csv')
specific <- read.csv('specific.csv')
observed <- read.csv('../../data/csv/simulated.csv')

observed$X = NULL
specific$X = NULL
general$X = NULL

library(rjags)

numRegions = dim(observed)[1]
nt = dim(observed)[2]

prop_unusual = 15/210

cut_model <- jags.model('cut.jags',
                        data = list('N' = numRegions,
                                    'prop_unusual' = prop_unusual,
                                    'T' = nt,
                                    'general' = general,
                                    'specific' = specific,
                                    'y' = observed),
                        n.chains = 4,
                        n.adapt = 10000)

predict_mcarray <- coda.samples(cut_model,
                                c('z'),
                                n.iter = 10000)

#predict_mcmc <- as.mcmc.list(predict_mcarray)
predict <- as.matrix(predict_mcarray)
predict_av <- colMeans(predict)
predict_sorted <- sort(predict_av, index.return=TRUE)

predict_values <- predict_sorted$x
predict_regions <- predict_sorted$ix

write.csv(predict_regions, 'predicted.csv')
write.csv(predict_values, 'values.csv')

#plot(predict_mcarray[,18,])


