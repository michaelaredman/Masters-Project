setwd("~/4th Year/project/data/csv/")
load("~/4th Year/project/data/rdata/expected_data.Rda")
load("~/4th Year/project/data/rdata/simulated_data.Rda")
load("~/4th Year/project/data/rdata/unusual_areas.Rda")
asthma <- asthma_expected_i[asthma_expected_i$CCG != '10L',]
expected = as.vector(asthma['E']$E)
sim <- Simulated_data[Simulated_data$X != '10L',]
simulated = as.matrix(sim[,c(2,3,4,5,6,7,8,9,10,11,12,13,14,15,16)])
rownames(simulated) <- 1:nrow(simulated)
write.csv(expected, file='expected.csv')
write.csv(simulated, file='simulated.csv')

library(sp)
library(shapefiles)
library(maptools)
library(rgdal)
library(spdep)
setwd("~/4th Year/project/data/shapefiles")
shape <- readOGR('.', 'CCG_BSC Feb2013  (clipcoast 200m)')
shape_deathtowales <- shape[shape@data$CCGname != 'Wales',]
shape_deathtoisleofwight <- shape_deathtowales[shape_deathtowales@data$CCGname != 'NHS Isle of Wight CCG',]

neib <- poly2nb(shape_deathtoisleofwight)
adj <- nb2mat(neib, style="B") 
write.csv(adj, file='~/4th Year/project/data/csv/adjacency.csv')
cardinality = card(neib)
adj_list <- neib[1]
for (i in 2:210) {
  adj_list <- c(adj_list, neib[i], recursive=TRUE)
}
write.csv(adj_list, file='~/4th Year/project/data/csv/adj_list.csv')
write.csv(cardinality, file='~/4th Year/project/data/csv/card.csv')

row.names(asthma) <- 1:210
View(asthma)
unusual <- asthma[is.element(asthma$CCG, unusual_areas$CCG),]
unusual_regions <- row.names(unusual)
write.csv(unusual_regions, file='~/4th Year/project/data/csv/unusual.csv')
