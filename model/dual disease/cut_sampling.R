numSamples <- 50
setwd("~/4th Year/project/model/dual disease")

general.sample.A.unshaped <- read.csv('general_sample_A.csv', header=FALSE)
general.sample.A.unshaped <- general.sample.A.unshaped$V1
general.sample.A <- aperm(array(general.sample.A.unshaped, dim=c(15, 210, numSamples)))

specific.sample.A.unshaped <- read.csv('specific_sample_A.csv', header=FALSE)
specific.sample.A.unshaped <- specific.sample.A.unshaped$V1
specific.sample.A <- aperm(array(specific.sample.A.unshaped, dim=c(15, 210, numSamples)))

general.sample.B.unshaped <- read.csv('general_sample_B.csv', header=FALSE)
general.sample.B.unshaped <- general.sample.B.unshaped$V1
general.sample.B <- aperm(array(general.sample.B.unshaped, dim=c(15, 210, numSamples)))

specific.sample.B.unshaped <- read.csv('specific_sample_B.csv', header=FALSE)
specific.sample.B.unshaped <- specific.sample.B.unshaped$V1
specific.sample.B <- aperm(array(specific.sample.B.unshaped, dim=c(15, 210, numSamples)))

general.A.flat <- matrix(0, nrow=210, ncol=numSamples*15)
for(i in 1:210) {
  for(sample in 0:(numSamples-1)) {
    general.A.flat[i, (1+15*sample):((sample+1)*15)] <- general.sample.A[sample+1, i, ]
  }
}
specific.A.flat <- matrix(0, nrow=210, ncol=numSamples*15)
for(i in 1:210) {
  for(sample in 0:(numSamples-1)) {
    specific.A.flat[i, (1+15*sample):((sample+1)*15)] <- specific.sample.A[sample+1, i, ]
  }
}
general.B.flat <- matrix(0, nrow=210, ncol=numSamples*15)
for(i in 1:210) {
  for(sample in 0:(numSamples-1)) {
    general.B.flat[i, (1+15*sample):((sample+1)*15)] <- general.sample.B[sample+1, i, ]
  }
}
specific.B.flat <- matrix(0, nrow=210, ncol=numSamples*15)
for(i in 1:210) {
  for(sample in 0:(numSamples-1)) {
    specific.B.flat[i, (1+15*sample):((sample+1)*15)] <- specific.sample.B[sample+1, i, ]
  }
}

observed.A <- read.csv("../../data/csv/simulated_A.csv")
observed.A$X <- NULL
observed.A.duplicated <- matrix(0, nrow=210, ncol=numSamples*15)
for(i in 0:(numSamples-1)) {
  observed.A.duplicated[,(i*15+1):((i+1)*15)] <- as.matrix(observed.A)
}

observed.B <- read.csv("../../data/csv/simulated_B.csv")
observed.B$X <- NULL
observed.B.duplicated <- matrix(0, nrow=210, ncol=numSamples*15)
for(i in 0:(numSamples-1)) {
  observed.B.duplicated[,(i*15+1):((i+1)*15)] <- as.matrix(observed.B)
}

numRegions = dim(observed.A)[1]
nt = dim(observed.A)[2]

prop_unusual = 15/210

library(rjags)

cut_model_sampling <- jags.model('cutNew.jags',
                                  data = list('N' = numRegions,
                                              'prop_unusual' = prop_unusual,
                                              'T' = nt,
                                              'numSamples' = numSamples,
                                              'general_flat_A' = general.A.flat,
                                              'specific_flat_A' = specific.A.flat,
                                              'general_flat_B' = general.B.flat,
                                              'specific_flat_B' = specific.B.flat,
                                              'region_A' = observed.A.duplicated,
                                              'region_B' = observed.B.duplicated),
                                  n.chains = 4,
                                  n.adapt = 10000)

predict_samples <- coda.samples(cut_model_sampling,
                                c('z'),
                                n.iter = 10000)

predict <- as.matrix(predict_samples)
predict_av <- colMeans(predict)
predict_sorted <- sort(predict_av, index.return=TRUE)

predict_values <- predict_sorted$x
predict_regions <- predict_sorted$ix
