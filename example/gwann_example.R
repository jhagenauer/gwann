library(viridis)
library(reshape2)
devtools::install(args=c("--no-multiarch"))
library(gwann)
library(gwann)

data(toy4)

x<-as.matrix(toy4[,c("x1","x2")])
y<-as.numeric(toy4[,c("y")] )
dm<-as.matrix(dist(toy4[,c("lon","lat")])  )

# cross validation
s_test<-sample(nrow(x),0.3*nrow(x))

r<-gwann(x_train=x[-s_test,],y_train=y[-s_test],w_train=dm[-s_test,-s_test],
         x_pred=x[s_test,],y_pred=y[s_test],w_train_pred=dm[-s_test,s_test],
         nrHidden=5,batchSize=100,lr=0.01,
         adaptive=F,
         bwSearch="goldenSection",
         bwMin=min(dm)/4, bwMax=max(dm)/4, steps=10,permutations=100,
         threads=1
)
p<-diag(r$predictions)
print(paste("Out-of-sample RMSE: ",sqrt(mean((p-y[s_test])^2))))
print(paste("In-sample RMSE: ",r$rmse))
print(paste("Iterations: ",r$its))
print(paste("Bandwidth: ",r$bw))
