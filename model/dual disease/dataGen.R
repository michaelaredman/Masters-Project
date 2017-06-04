library(maptools)
library(CARBayesdata)
library(rgdal)
library(spdep)
library(MASS)
library(classInt)
library(RColorBrewer)
setwd("~/4th Year/project/data/shapefiles")
shape.basic <- readOGR('.', 'CCG_BSC Feb2013  (clipcoast 200m)')
shape_deathtowales <- shape.basic[shape.basic@data$CCGname != 'Wales',]
shape <- shape_deathtowales[shape_deathtowales@data$CCGname != 'NHS Isle of Wight CCG',]
load("~/4th Year/project/data/rdata/expected_data.Rda")
asthma <- asthma_expected_i[asthma_expected_i$CCG != '10L',]
expected = as.vector(asthma['E']$E)

unusual_temp <- read.csv('~/4th Year/project/data/csv/prefUnusual.csv', header=FALSE)
unusual <- sort(unusual_temp$V1) + 1 #add one as python had zero index

random_walk <- function(len, sigma) {
  walk <- rep(0, len)
  walk[1] <- rnorm(1, sd=sigma)
  for(i in 2:len) {
    walk[i] <- walk[i-1] + rnorm(1, sd=sigma)
  }
  return(walk)
}

set.seed(21)

rwalk.A <- random_walk(len=15, sigma=0.1)
rwalk.A <- rwalk.A - mean(rwalk.A)
rwalk.B <- random_walk(len=15, sigma=0.1)
rwalk.B <- rwalk.B - mean(rwalk.B)

rwalk.A.unusual <- rwalk.A
rwalk.A.unusual[c(8, 10, 12)] <- rwalk.A.unusual[c(8, 10, 12)] + abs(rnorm(n=3, sd = 0.1))
rwalk.B.unusual <- rwalk.B
rwalk.B.unusual[12:15] <- rwalk.B.unusual[12:15] + abs(rnorm(n=4, sd = 0.1))

plot(rwalk.A, type='l')
lines(rwalk.A.unusual, type='l')
plot(rwalk.B, type='l')
lines(rwalk.B.unusual, type='l')

neib <- poly2nb(shape)
adj <- nb2mat(neib, style="B") 
num.neib <- colSums(adj)
alpha <- 0.9
prec.matrix <- diag(num.neib) - alpha*adj
cov.matrix <- solve(prec.matrix)
spatial.sd.A <- 0.02
spatial.sd.B <- 0.04

CAR.A <- mvrnorm(mu=rep(0, length(num.neib)), Sigma=cov.matrix*spatial.sd.A)
CAR.B <- mvrnorm(mu=rep(0, length(num.neib)), Sigma=cov.matrix*spatial.sd.B)

class <- classIntervals(CAR.A, 9, style="quantile")
plotclr <- brewer.pal(9,"YlOrRd")
colcode <- findColours(class, plotclr)
plot(shape, col=colcode)
class <- classIntervals(CAR.B, 9, style="quantile")
plotclr <- brewer.pal(9,"YlOrRd")
colcode <- findColours(class, plotclr)
plot(shape, col=colcode)

mu.A <- matrix(data=0, nrow=210, ncol=15)
for(i in 1:210) {
  mu.A[i, ] <- mu.A[i, ] + rwalk.A
}
for(i in unusual) {
  mu.A[i, ] <- rwalk.A.unusual
}
for(i in 1:15) {
  mu.A[, i] <- mu.A[, i] + CAR.A + log(expected)
}
rate.matrix.A <- exp(mu.A)

mu.B <- matrix(data=0, nrow=210, ncol=15)
for(i in 1:210) {
  mu.B[i, ] <- mu.B[i, ] + rwalk.B
}
for(i in unusual) {
  mu.B[i, ] <- rwalk.B.unusual
}
for(i in 1:15) {
  mu.B[, i] <- mu.B[, i] + CAR.B + log(expected)
}
rate.matrix.B <- exp(mu.B)

pois <- function(lambda) {rpois(n=1, lambda)}

simulated.A <- apply(rate.matrix.A, MARGIN=c(1, 2), FUN=pois)
simulated.B <- apply(rate.matrix.B, MARGIN=c(1, 2), FUN=pois)
write.csv(simulated.A, '~/4th Year/project/data/csv/simulated_A.csv')
write.csv(simulated.B, '~/4th Year/project/data/csv/simulated_B.csv')


