library(viridis)
library(gwann)
library(ggplot2)
library(caret)

data(toy4)

x<-as.matrix(toy4[,c("x1","x2")])
y<-as.numeric(toy4[,c("y")] )
dm<-as.matrix(dist(toy4[,c("lon","lat")])  )

folds<-createMultiFolds(toy4$y,k=10,times=1)
df<-data.frame()
for( s_train in folds ) {
  s_test<-(-s_train)

  for( lr in c(0.05, 0.1))
    for( n in c(5,15,25,35,45) ) {
      r<-gwann(x_train=x[-s_test,],y_train=y[-s_test],w_train=dm[-s_test,-s_test],
               x_pred=x[s_test,],w_pred=dm[-s_test,s_test],
               nrHidden=n,batchSize=50,lr=lr,
               adaptive=F,cv_patience=999,cv_max_iterations=99999,
               #bwSearch="goldenSection",bwMin=min(dm)/4, bwMax=max(dm)/4,
               bandwidth=2,
               threads=8
      )
      p<-diag(r$predictions)
      rmse<-sqrt(mean((p-y[s_test])^2))
      df<-rbind(df,data.frame(lr=lr,n=n,rmse=rmse))
    }
}
aggregate(df,by=list(df$lr,df$n),mean)
