setwd("~/4th Year/project/model/bugs/")

observed <- read.csv('../../data/csv/simulated.csv')
observed$X = NULL
numRegions = dim(observed)[1]
nt = dim(observed)[2]

adj_list = read.csv("../../data/csv/adj_list.csv")$x


cardinality = read.csv("../../data/csv/card.csv")
cardinality$X = NULL

WINE="/usr/local/bin/wine"
WINEPATH="/usr/local/bin/winepath"
OpenBUGS.pgm="/Users/Mike/.wine/drive_c/Program Files (x86)/OpenBUGS/OpenBUGS323/OpenBUGS.exe"

baystdetect <- function(){
  for (i in 1:N) {
    for (t in 1:T) {
      y[i,t] ~ dpois(m[i,t])
      log(m[i,t]) <- log.mu[i,t] + log(E[i,t])
      log.mu[i,t] <- z[i] * common[i,t] + (1-z[i]) * specific[i,t]
    }
    z[i] ~ dbern(0.95)
  }     
  # Common-trend Model 1
  for (i in 1:N) {
    for (t in 1:T) {
      y1[i,t] ~ dpois(mu1[i,t])
      log(mu1[i,t]) <- log(E[i,t]) + alpha0 + eta[i] + gamma[t]
      temp1[i,t] <- alpha0 + eta[i] + gamma[t]
      common[i,t] <- cut(temp1[i,t])
    }
    eta[i] ~ dnorm(v[i],prec.eta)
  }
  # prior specifications for Model 1
  alpha0 ~ dflat()
  v[1:N] ~ car.normal(adj[],weights[],num[],prec.v)
  gamma[1:T] ~ car.normal(adj.tm[],weights.tm[],num.tm[],prec.gamma)
  prec.v <- pow(sigma.v,-2)
  sigma.v ~ dnorm(0,1)%_%I(0,)
  prec.gamma <- pow(sigma.gamma,-2)
  sigma.gamma ~ dnorm(0,1)%_%I(0,)
  prec.eta <- pow(sigma.eta,-2)
  sigma.eta ~ dnorm(0,1)%_%I(0,)
  # Area-specific Model 2
  for (i in 1:N) {
    for (t in 1:T) {
      y2[i,t] ~ dpois(mu2[i,t])
      log(mu2[i,t]) <- log(E[i,t]) + u[i] + xi[i,t]
      temp2[i,t] <- u[i] + xi[i,t]
      specific[i,t] <- cut(temp2[i,t])
    }
    # area-specific trends
    xi[i,1:T] ~ car.normal(adj.tm[],weights.tm[],num.tm[],prec.xi[i])
    # area-specific intercepts (no smoothing)
    u[i] ~ dnorm(0,0.001)
    # hierarchical modelling of the local temporal variability
    prec.xi[i] <- pow(var.xi[i],-1)
    var.xi[i] <- exp(log.var.xi[i])
    log.var.xi[i] ~ dnorm(mean.log.var.xi,prec.log.var.xi)
    sigma.xi[i] <- pow(var.xi[i],0.5)
  }
  # hyper priors
  mean.log.var.xi ~ dnorm(0,0.001)
  prec.log.var.xi <- pow(var.log.var.xi,-1)
  var.log.var.xi <- pow(sd.log.var.xi,2)
  sd.log.var.xi ~ dnorm(0,prec.sd.log.var.xi)%_%I(0,)
  sd.sd.log.var.xi <- 2.5
  prec.sd.log.var.xi <- pow(sd.sd.log.var.xi,-2)
}

# write the model code out to a file
write.model(baystdetect, "baystdetect.txt")
model.file1 = paste(getwd(),"baystdetect.txt", sep="/")
## and let's take a look:
file.show("baystdetect.txt")

weights <- rep(1, length(adj_list))

num.tm = list()
adj.tm = list()
num.tm[1] = 1; adj.tm[1] = 2
num.tm[2] = 2; adj.tm[2] = 1; adj.tm[3] = 3 
num.tm[3] = 2; adj.tm[4] = 2; adj.tm[5] = 4
num.tm[4] = 2; adj.tm[6] = 3; adj.tm[7] = 5
num.tm[5] = 2; adj.tm[8] = 4; adj.tm[9] = 6
num.tm[6] = 2; adj.tm[10] = 5; adj.tm[11] = 7
num.tm[7] = 2; adj.tm[12] = 6; adj.tm[13] = 8
num.tm[8] = 2; adj.tm[14] = 7; adj.tm[15] = 9
num.tm[9] = 2; adj.tm[16] = 8; adj.tm[17] = 10
num.tm[10] = 2; adj.tm[18] = 9; adj.tm[19] = 11
num.tm[11] = 2; adj.tm[20] = 10; adj.tm[21] = 12
num.tm[12] = 2; adj.tm[22] = 11; adj.tm[23] = 13
num.tm[13] = 2; adj.tm[24] = 12; adj.tm[25] = 14
num.tm[14] = 2; adj.tm[26] = 13; adj.tm[27] = 15
num.tm[15] = 1; adj.tm[28] = 14

weights.tm <- rep(1, length(adj.tm))

#initialization of variables
data <- list(y = as.matrix(observed), N = numRegions, T = nt, adj=c(adj_list), weights=weights, num=c(cardinality), adj.tm=adj.tm, 
             weights.tm=weights.tm, num.tm=num.tm)

bayst_trace <- bugs(data, model.file = model.file1,n.chains = 3, n.iter = 1000, 
                    OpenBUGS.pgm=OpenBUGS.pgm, WINE=WINE, WINEPATH=WINEPATH,useWINE=T)
