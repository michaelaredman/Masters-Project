library(sp)
library(shapefiles)
library(maptools)
library(CARBayesdata)
library(rgdal)
library(spdep)
library(classInt)
library(RColorBrewer)
lipshape <- readOGR('.', 'scot')
data(lipshape)
# Projection is squished
proj4string(lipshape) <- CRS("+proj=merc")
summary(lipshape)
# Let's change it to something better
lipproj <- spTransform(lipshape, CRS("+proj=cea +lon_0=0 +lat_ts=30 +x_0=0 +y_0=0 +datum=WGS84 +ellps=WGS84 +units=m +no_defs"))
plot(lipproj)
plot(lipshape)

# Calculate the adjacency matrix
neib <- poly2nb(lipproj)
adjNorm <- nb2mat(neib) # Note: This is already normalized!
adj <- nb2mat(neib, style="B") # Note: This is already normalized!

# Some extra stuff
coords <- coordinates(lipproj)
plot(neib, coords, col="grey")
plot(neib, coords, col="grey", add=TRUE)
# And some more!
lipdata.df <- as.data.frame(lipdata)
cases <- data.matrix(lipdata.df['observed'])
plotclr <- brewer.pal(8,"YlOrRd")
class <- classIntervals(cases, 8, style="quantile")
colcode <- findColours(class, plotclr)
plot(lipproj, col=colcode)
legend("topleft",legend=leglabs(round(class$brks)), fill=plotclr, cex=0.6, inset=0.05)

# We now prepare the data for stan
lipdata.df <- as.data.frame(lipdata)
cases <- data.matrix(lipdata.df['observed']) # These are our y[i]
predictor <- data.matrix(lipdata.df['pcaff']) # Proportion working in jobs with potential exposure
expected_cases <- data.matrix(lipdata.df['expected']) # The number of cases expected by relevant population alone
X <- model.matrix(~predictor) # This is the design matrix
full_d <- list(n = nrow(X),         # number of observations
               p = ncol(X),         # number of coefficients
               X = X,               # design matrix
               y = c(cases),               # observed number of cases
               log_offset = c(log(expected_cases)), # log(expected) num. cases
               W = adj)  

# We now run stan
niter <- 1000
nchains <- 4
full_fit <- stan('scot.stan', data = full_d, iter = niter, chains = nchains, verbose = FALSE)
print(full_fit, pars = c('beta', 'tau', 'alpha', 'lp__'))

# visualize results
to_plot <- c('beta', 'tau', 'alpha', 'phi[1]', 'phi[2]', 'phi[3]', 'lp__')
traceplot(full_fit, pars = to_plot)
library(ggmcmc)
library(coda)
s <- mcmc.list(lapply(1:ncol(fit), function(x) mcmc(as.array(fit)[,x,])))








