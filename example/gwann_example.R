library(ggplot2)
library(viridis)
library(reshape2)
devtools::install(args=c("--no-multiarch"))
library(gwann)

data(toy4)

x_train<-as.matrix(toy4[,c("x1","x2")])
y_train<-as.numeric(toy4[,c("y")] )
dm<-as.matrix(dist(toy4[,c("lon","lat")])  )

r<-gwann::gwann(x_train=x_train,y_train=y_train,w_train=dm,
         x_pred=x_train,y_pred=y_train,w_train_pred=dm,
         nrHidden=5,batchSize=100,lr=0.01,
         adaptive=F,
         bandwidth=1.97,
         #bwSearch="goldenSection",
         bwMin=min(dm)/4, bwMax=max(dm)/4, steps=10,permutations=100,
         threads=1
)
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

# importance x2
s<-cbind( err_diff=diag(r$importance[2,,]), toy4[,c("lon","lat")] )
ggplot(s,aes(lon,lat,fill=err_diff)) + geom_raster() + scale_fill_viridis() + coord_fixed()
