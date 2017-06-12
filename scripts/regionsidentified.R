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

regions_identified <- c(6, 10, 41, 75, 82, 83, 101, 107, 156, 178, 202)
anom_regions <- c(159, 67, 37, 145)
not_identified <- c(29, 57, 105, 196)

numRegions <- length(shape@data$SP_ID)
cols <- rep('white', numRegions)
cols[regions_identified] <- rep('forestgreen', length(regions_identified))
cols[anom_regions] <- rep('firebrick2', length(anom_regions))
cols[not_identified] <- rep('dodgerblue', length(not_identified))

plot(shape, col=cols)

regions_identified <- c(8, 10, 11)
not_identified <- c(3, 23, 25, 27, 43, 86, 100, 125, 130, 152, 2, 6)
anom_regions <- c(169, 140, 130, 69, 118, 155, 13, 153, 117, 206, 82, 181)

numRegions <- length(shape@data$SP_ID)
cols <- rep('white', numRegions)
cols[regions_identified] <- rep('forestgreen', length(regions_identified))
cols[anom_regions] <- rep('firebrick2', length(anom_regions))
cols[not_identified] <- rep('dodgerblue', length(not_identified))

plot(shape, col=cols)

foobar <- replicate(10000, length(intersect(sample(1:210, 15), 1:15)))
mean(foobar)
mean_foobar <- 15*15/210


