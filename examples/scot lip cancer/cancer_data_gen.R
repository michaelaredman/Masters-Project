setwd("~/4th Year/project/examples/scot lip cancer")
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

# Calculate the adjacency matrix
neib <- poly2nb(lipshape)
adjNorm <- nb2mat(neib) # Note: This is already normalized!
adj <- nb2mat(neib, style="B") # Note: This is already normalized!

# We now prepare the data for stan
data(lipdata)
lipdata.df <- as.data.frame(lipdata)
cases <- data.matrix(lipdata.df['observed']) # These are our y[i]
predictor <- data.matrix(lipdata.df['pcaff']) # Proportion working in jobs with potential exposure
expected_cases <- data.matrix(lipdata.df['expected']) # The number of cases expected by relevant population alone
X <- model.matrix(~predictor) # This is the design matrix
write.csv(X, 'X.csv')
write.csv(expected_cases, 'expected.csv')
write.csv(cases, 'cases.csv')
write.csv(adj, 'adj.csv')
