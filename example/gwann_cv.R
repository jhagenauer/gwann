library(viridis)
library(gwann)
library(ggplot2)
library(caret)

data(toy4)

x<-as.matrix(toy4[,c("x1","x2")])
y<-as.numeric(toy4[,c("y")] )
dm<-as.matrix(dist(toy4[,c("lon","lat")])  )

e<-expand.grid(bs=c(10,25,50,100),lr=0.1,n=c(5,15,20,25,30,35,45))

folds<-createMultiFolds(toy4$y,k=10,times=1)
df<-data.frame()
for( s_train in folds ) {
  s_test<-(-s_train)
  for( i in 1:nrow(e) ) {
    p<-e[i,]
      r<-gwann(x_train=x[-s_test,],y_train=y[-s_test],w_train=dm[-s_test,-s_test],
               x_pred=x[s_test,],w_pred=dm[-s_test,s_test],
               nrHidden=p$n,batchSize=p$bs,lr=p$lr,
               adaptive=F,cv_patience=999,cv_max_iterations=99999,
               #bwSearch="goldenSection",bwMin=min(dm)/4, bwMax=max(dm)/4,
               bandwidth=2,
               threads=8
      )
      pred<-diag(r$predictions)
      rmse<-sqrt(mean((pred-y[s_test])^2))
      df<-rbind(df,data.frame(bs=p$bs,lr=p$lr,n=p$n,rmse=rmse))
    }
}
aggregate(df,by=list(df$lr,df$n),mean)
