library(ggplot2)
library(viridis)
library(reshape2)
library(gwann)

data(toy4)
dm<-as.matrix(dist(toy4[,c("lon","lat")])  )
x<-as.matrix(toy4[,c("x1","x2")])
y<-as.numeric(toy4[,c("y")] )

r<-gwann(x=x,y=y,dm=dm,trainIdx=1:nrow(x),predIdx=1:nrow(x),nrHidden=5,batchSize=100,threads=8,adaptive=F,bandwidth=1.801,lr=0.01,patience=1000)
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

#################

library(rJava)
.jinit()
.jaddClassPath("inst/java/commons-math3-3.6.1.jar")
.jaddClassPath("inst/java/gt-api-19.1.jar")
.jaddClassPath("inst/java/gt-data-19.1.jar")
.jaddClassPath("inst/java/gt-main-19.1.jar")
.jaddClassPath("inst/java/gwann-0.0.1-SNAPSHOT.jar")
.jaddClassPath("inst/java/jblas-1.2.4.jar")
.jaddClassPath("inst/java/jts-core-1.14.0.jar")
.jaddClassPath("inst/java/jts-io-1.14.0.jar")
.jaddClassPath("inst/java/log4j-1.2.17.jar")


data(toy4)

dm<-as.matrix( dist(toy4[,c("lon","lat")])  )
x<-as.matrix(toy4[,c("x1","x2")])
y<-as.numeric( toy4[,c("y")] )

x<-x
y<-y
nrHidden<-5
bandwidth<-1.8
batchSize<-100
opt="adam"
threads<-8
batchSize<-10
lr<-0.01
linOut<-T
kernel<-"gaussian"
adaptive<-T
iterations<-100
patience<-100

.jmethods("supervised.nnet.gwann.GWANN_RInterface")

r<-.jcall(obj="supervised.nnet.gwann.GWANN_RInterface",method="run",returnSig = "Lsupervised/nnet/gwann/ReturnObject;",
          .jarray(x,dispatch=T),
          y,
          .jarray(dm,dispatch=T),
          as.integer(1:nrow(dm)),
          as.integer(1:nrow(dm)),
          nrHidden,batchSize,opt,lr,linOut,kernel,bandwidth,adaptive,iterations,patience,threads)
