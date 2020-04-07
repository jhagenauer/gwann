# Geographically Weighted Artificial Neural Network

### Install

    library(devtools)
    install_github("jhagenauer/gwann",INSTALL_opts=c("--no-multiarch"))
    
    if (!require("devtools"))
       install.packages("devtools")
    devtools::install_github("jhagenauer/gwann",INSTALL_opts=c("--no-multiarch"))
    
### Example

    library(ggplot2)
    library(devtools)
    library(viridis)
    library(reshape2)
    library(gwann)

    data(toy4)

    dm<-as.matrix( dist(toy4[,c("lon","lat")])  )
    x<-as.matrix(toy4[,c("x1","x2")])
    y<-as.numeric( toy4[,c("y")] )

    r<-gwann(x=x,y=y,dmX=dm,dmP=dm,nrHidden=8,batchSize=50,bandwidth=1.49)

    a<-cbind( t(r$weights), toy4[,c("lon","lat")] )
    m<-melt(a,id.vars=c("lon","lat"))
    ggplot(m,aes(lon,lat,fill=value)) + geom_raster() + facet_wrap(~variable) + scale_fill_viridis()

### References
