
roxygen2::roxygenise()
detach("package:gwann", unload=TRUE)
remove.packages("gwann")
devtools::install_local(".")

#############################################

library(viridis)
library(gwann)
library(ggplot2)
library(tidyverse)

x<-as.matrix(toy4[,c("x1","x2")])
y<-as.numeric(toy4[,c("y")] )
dm<-as.matrix(dist(toy4[,c("lon","lat")])  )
data(toy4)
s_test<-sample(nrow(x),0.3*nrow(x)) # indices of test samples

# Case 1

params<-list()
for( nr in c( 4))
  for( lr in c(0.1) )
    for( bs in c(50,100) )
      for( kn in c("gaussian") )
        for( opt in c("adam") )
          params[[length(params)+1]]<-data.frame(nr=nr,lr=lr,bs=bs,opt=opt,kn=kn)
params<-sample(params)

d_best<-Inf
for( p in params ) {

set.seed(1)

x_pred1 <- x[s_test,]
w_pred1 <- dm[-s_test, s_test]

r1<-gwann(x_train=x[-s_test,],y_train=y[-s_test],w_train=dm[-s_test,-s_test],
         x_pred=x_pred1,w_pred=w_pred1,
         nrHidden=p$nr,batchSize=p$bs,lr=p$lr,
         optimizer=p$opt,kernel=p$kn,
         adaptive=F,
         #bandwidth=99999999,
         cv_patience=9999,
         threads=15
)
p1<-diag(r1$predictions)
print(p1[1:5])

# Case 2

set.seed(1)

x_pred2 <- x[s_test[1:5],]
w_pred2 <- dm[-s_test, s_test[1:5]]

r2<-gwann(x_train=x[-s_test,],y_train=y[-s_test],w_train=dm[-s_test,-s_test],
         x_pred=x_pred2,w_pred=w_pred2,
         nrHidden=p$nr,batchSize=p$bs,lr=p$lr,
         optimizer=p$opt,kernel=p$kn,
         adaptive=F,
         #bandwidth=99999999,
         cv_patience=9999,
         threads=15
)
p2<-diag(r2$predictions)
print(p2)

d<-(p1[1:5]-p2)^2 %>% mean() %>% sqrt()
if( !is.nan(d) & d<d_best) {
  d_best<-d
  print(d)
  print(p)
}

}

# Case 3, difference due to different initialization and/or bad convergence

s_test3 <- s_test
s_test3[2] <- s_test3[1]

x_pred3 <- x[s_test3,]
w_pred3 <- dm[-s_test, s_test3]

params<-list()
for( nr in c(4))
  for( lr in c(0.1) )
    for( bs in c(50,100) )
      for( kn in c("gaussian") )
        for( opt in c("adam") )
          params[[length(params)+1]]<-data.frame(nr=nr,lr=lr,bs=bs,opt=opt,kn=kn)
params<-sample(params)


best_rmse<-Inf
results<-list()
for( p in params ) {
  set.seed(7)

  r3<-gwann(
    x_train=x[-s_test,],y_train=y[-s_test],w_train=dm[-s_test,-s_test],
    x_pred=x_pred3,w_pred=w_pred3,
    nrHidden=p$nr,batchSize=p$bs,lr=p$lr,
    optimizer=p$opt,
    kernel=p$kn,
    cv_patience=9999,
    threads=15
  )
  di<-diag(r3$predictions)

  d<-abs(di[1]-di[2])
  rmse<-(di-y[s_test3])^2 %>% mean() %>% sqrt()

  results[[length(results)+1]]<-data.frame(diff=d,rmse=rmse,p)

  if( rmse < best_rmse ) {
    print(paste("rmse",rmse))
    print(paste("diff",d))
    print(p)
    best_rmse<-rmse
  }
}
do.call(rbind,results) %>% View()
