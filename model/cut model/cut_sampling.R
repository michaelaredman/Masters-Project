general_sample_unshaped <- read.csv('general_sample.csv', header=FALSE)
general_sample_unshaped <- general_sample_unshaped$V1
general_sample <- aperm(array(general_sample_unshaped, dim=c(15, 210, 20)))

specific_sample_unshaped <- read.csv('specific_sample.csv', header=FALSE)
specific_sample_unshaped <- specific_sample_unshaped$V1
specific_sample <- aperm(array(specific_sample_unshaped, dim=c(15, 210, 20)))

numSamples <- 20

general_flat <- matrix(0, nrow=210, ncol=numSamples*15)
for(i in 1:210) {
  for(sample in 0:(numSamples-1)) {
    general_flat[i, (1+15*sample):((sample+1)*15)] <- general_sample[sample+1, i, ]
  }
}
specific_flat <- matrix(0, nrow=210, ncol=numSamples*15)
for(i in 1:210) {
  for(sample in 0:(numSamples-1)) {
    specific_flat[i, (1+15*sample):((sample+1)*15)] <- specific_sample[sample+1, i, ]
  }
}
observed_duplicated <- matrix(0, nrow=210, ncol=numSamples*15)
for(i in 0:(numSamples-1)) {
  observed_duplicated[,(i*15+1):((i+1)*15)] <- as.matrix(observed)
}

numRegions = dim(observed)[1]
nt = dim(observed)[2]

prop_unusual = 15/210

library(rjags)

cut_model_sampling <- jags.model('cutNew.jags',
                                  data = list('N' = numRegions,
                                              'prop_unusual' = prop_unusual,
                                              'T' = nt,
                                              'numSamples' = numSamples,
                                              'general_flat' = general_flat,
                                              'specific_flat' = specific_flat,
                                              'y' = observed_duplicated),
                                  n.chains = 4,
                                  n.adapt = 1000)

predict_samples <- coda.samples(cut_model_sampling,
                                c('z'),
                                n.iter = 10000)

predict <- as.matrix(predict_samples)
predict_av <- colMeans(predict)
predict_sorted <- sort(predict_av, index.return=TRUE)

predict_values <- predict_sorted$x
predict_regions <- predict_sorted$ix
