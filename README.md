# Geographically Weighted Artificial Neural Network

### System Requirements

Java JDK 1.2 or higher (for JRI/REngine JDK 1.4 or higher). If it is not already installed, you can get it [here](https://www.oracle.com/java/technologies/javase-downloads.html).

### Install
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

    r<-gwann(x=x,y=y,dmX=dm,dmP=dm,nrHidden=5,bandwidth=1.7,batchSize=50,threads=8)

    # convergence
    m<-do.call(rbind,lapply(1:length(r$evaluations), function(i) cbind(melt(r$evaluations[[i]]),fold=i)))
    ggplot(m) + geom_line(aes(x=Var1,y=value,colour=as.factor(fold))) + labs(x="Evaluations",y="RMSE")

    # weights input to hidden layer
    w1<-cbind( data.frame(r$weights[[1]]),c("x1","x2","Bias") )
    colnames(w1)<-c(paste("Neuron ",1:5),"Bias Neuron","input")
    m<-melt(w1)
    ggplot(m[complete.cases(m),]) + geom_bar(aes(input,value),stat="identity") + facet_wrap(~variable)

    # surfaces
    a<-cbind( t(r$weights[[2]]), toy4[,c("lon","lat")] )
    colnames(a)<-c(paste("Neuron",1:5),"Bias Neuron","lon","lat")
    m<-melt(a,id.vars=c("lon","lat"))
    ggplot(m,aes(lon,lat,fill=value)) + geom_raster() + facet_wrap(~variable) + scale_fill_viridis() + coord_fixed()

### References
