library(ggplot2)
library(viridis)
library(reshape2)
#devtools::install(args=c("--no-multiarch"))
library(gwann)

data(toy4)

x_train<-as.matrix(toy4[,c("x1","x2")])
y_train<-as.numeric(toy4[,c("y")] )
dm<-as.matrix(dist(toy4[,c("lon","lat")])  )

r<-gwann(x_train=x_train,y_train=y_train,w_train=dm,
         x_pred=x_train,y_pred=y_train,w_train_pred=dm,
         nrHidden=5,batchSize=100,lr=0.05,
         adaptive=F,
         #bandwidth=10,
         bwSearch="goldenSection",
         bwMin=min(dm)/4, bwMax=max(dm)/4,
         threads=8,
         permutations = 1000
)
print(paste("RMSE: ",r$rmse))
print(paste("Iterations: ",r$its))
print(paste("Bandwidth: ",r$bw))

# predictions
s<-cbind( Prediction=diag(r$predictions), toy4[,c("lon","lat")] )
ggplot(s,aes(lon,lat,fill=Prediction)) + geom_raster() + scale_fill_viridis() + coord_fixed()

# importance
s<-cbind( x1=diag(r$importance[1,,]), x2=diag(r$importance[2,,]),toy4[,c("lon","lat")] )
m<-melt(s,id.vars=c("lon","lat"))
ggplot(m,aes(lon,lat,fill=value)) + geom_raster() + facet_wrap(~variable) + scale_fill_viridis() + coord_fixed()

# surfaces
s<-cbind( t(r$weights[[2]]), toy4[,c("lon","lat")] )
#colnames(s)<-c(paste("Neuron",1:5),"Bias Neuron","lon","lat")
m<-melt(s,id.vars=c("lon","lat"))
ggplot(m,aes(lon,lat,fill=value)) + geom_raster() + facet_wrap(~variable) + scale_fill_viridis() + coord_fixed()

# cluster
k<-kmeans( t(r$weights[[2]]), centers=5 )
s<-cbind( cluster=as.factor(k$cluster), toy4[,c("lon","lat")] )
ggplot(s,aes(lon,lat,fill=cluster)) + geom_raster() +  coord_fixed()

# test
library(data.table)
m<-as.data.table(m)

s<-m[,list(mean=mean(value),sd=sd(value)),by=list(lat,lon)]
ggplot(s,aes(lon,lat,fill=mean)) + geom_raster() + scale_fill_viridis() + coord_fixed()
ggplot(s,aes(lon,lat,fill=sd)) + geom_raster() + scale_fill_viridis() + coord_fixed()
