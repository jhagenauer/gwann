# Geographically Weighted Artificial Neural Network

### System Requirements

Java JDK 1.2 or higher (for JRI/REngine JDK 1.4 or higher). If it is not already installed, you can get it [here](https://www.oracle.com/java/technologies/javase-downloads.html).

### Install
    if (!require("devtools"))
       install.packages("devtools")
    devtools::install_github("jhagenauer/gwann",INSTALL_opts=c("--no-multiarch"))
    
### Example

    library(ggplot2)
    library(viridis)
    library(reshape2)
    library(gwann)

    data(toy4)
    dm<-as.matrix(dist(toy4[,c("lon","lat")])  )
    x<-as.matrix(toy4[,c("x1","x2")])
    y<-as.numeric(toy4[,c("y")] )

    r<-gwann(x=x,y=y,dm=dm,trainIdx=1:nrow(x),predIdx=1:nrow(x),nrHidden=5,batchSize=100,lr=0.01,adaptive=F,gridSearch=T, minBw=1, maxBw=2, steps=10,threads=8)
    print(paste("RMSE: ",r$rmse))
    print(paste("Iterations: ",r$its))
    print(paste("Bandwidth: ",r$bw))

    # predictions
    s<-cbind( Prediction=diag(r$predictions), toy4[,c("lon","lat")] )
    ggplot(s,aes(lon,lat,fill=Prediction)) + geom_raster() + scale_fill_viridis() + coord_fixed()

    # surfaces
    s<-cbind( t(r$weights[[2]]), toy4[,c("lon","lat")] )
    colnames(s)<-c(paste("Neuron",1:5),"Bias Neuron","lon","lat")
    m<-melt(s,id.vars=c("lon","lat"))
    ggplot(m,aes(lon,lat,fill=value)) + geom_raster() + facet_wrap(~variable) + scale_fill_viridis() + coord_fixed()

### References
