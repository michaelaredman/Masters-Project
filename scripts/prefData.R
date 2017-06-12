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

numRegions <- length(shape@data$SP_ID)
cols <- rep('white', numRegions)
cols[unusual] <- rep('yellow', length(unusual))

plot(shape, col=cols)

random_walk <- function(len, sigma) {
  walk <- rep(0, len)
  walk[1] <- rnorm(1, sd=sigma)
  for(i in 2:len) {
    walk[i] <- walk[i-1] + rnorm(1, sd=sigma)
  }
  return(walk)
}

set.seed(314)

rwalk <- random_walk(len=15, sigma=0.1)
rwalk <- rwalk - mean(rwalk)

rwalk_unusual <- rwalk
rwalk_unusual[10:15] <- rwalk_unusual[10:15] + abs(rnorm(n=6, sd = 0.3))

plot(rwalk_unusual, type='l', col='red', ylab = 'Temporal trend', xlab = 't')
lines(rwalk, type='l', col='blue')

neib <- poly2nb(shape)
adj <- nb2mat(neib, style="B") 
num.neib <- colSums(adj)
alpha <- 0.9
prec.matrix <- diag(num.neib) - alpha*adj
cov.matrix <- solve(prec.matrix)
spatial.sd <- 0.02

CAR <- mvrnorm(mu=rep(0, length(num.neib)), Sigma=cov.matrix*spatial.sd)

class <- classIntervals(CAR, 9, style="quantile")
display.brewer.pal(name = "YlOrRd", n=9)
plotclr <- brewer.pal(9,"YlOrRd")
colcode <- findColours(class, plotclr)
plot(shape, col=colcode)

mu <- matrix(data=0, nrow=210, ncol=15)
for(i in 1:210) {
  mu[i, ] <- mu[i, ] + rwalk
}
for(i in unusual) {
  mu[i, ] <- rwalk_unusual
}
for(i in 1:15) {
  mu[, i] <- mu[, i] + CAR + log(expected)
}
rate.matrix <- exp(mu)

pois <- function(lambda) {rpois(n=1, lambda)}
simulated <- apply(rate.matrix, MARGIN=c(1, 2), FUN=pois)
write.csv(simulated, '~/4th Year/project/data/csv/simulated_spatial_corr.csv')

for(i in 1:15) {
  class2 <- classIntervals(simulated[,i], 9, style="quantile")
  plotclr2 <- brewer.pal(9,"YlOrRd")
  colcode2 <- findColours(class2, plotclr2)
  plot(shape, col=colcode2)
}



