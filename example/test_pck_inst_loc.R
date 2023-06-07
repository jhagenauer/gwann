
roxygen2::roxygenise()
detach("package:gwann", unload=TRUE)
remove.packages("gwann")
devtools::install_local(".")

#############################################

library(viridis)
library(gwann)
library(ggplot2)

x<-as.matrix(toy4[,c("x1","x2")])
y<-as.numeric(toy4[,c("y")] )
dm<-as.matrix(dist(toy4[,c("lon","lat")])  )
data(toy4)
s_test<-sample(nrow(x),0.3*nrow(x)) # indices of test samples

# Case 1

set.seed(1)

x_pred1 <- x[s_test,]
w_pred1 <- dm[-s_test, s_test]

r1<-gwann(x_train=x[-s_test,],y_train=y[-s_test],w_train=dm[-s_test,-s_test],
         x_pred=x_pred1,w_pred=w_pred1,
         nrHidden=1,batchSize=100,lr=0.05,
         adaptive=F,
         bandwidth=2.1,
         iterations=10000,
         threads=8
)
p<-diag(r1$predictions)
print(p[1:5])

# Case 2

set.seed(1)

x_pred2 <- x[s_test[1:5],]
w_pred2 <- dm[-s_test, s_test[1:5]]

r2<-gwann(x_train=x[-s_test,],y_train=y[-s_test],w_train=dm[-s_test,-s_test],
         x_pred=x_pred2,w_pred=w_pred2,
         nrHidden=1,batchSize=100,lr=0.05,
         adaptive=F,
         bandwidth=2.1,
         #iterations=10000,
         threads=8
)
p<-diag(r2$predictions)
print(p[1:5])

# Case 3

set.seed(1)

s_test3 <- s_test
s_test3[2] <- s_test3[1]
x_pred3 <- x[s_test3,]
w_pred3 <- dm[-s_test, s_test3]

r3<-gwann(x_train=x[-s_test,],y_train=y[-s_test],w_train=dm[-s_test,-s_test],
         x_pred=x_pred3,w_pred=w_pred3,
          nrHidden=4,batchSize=100,lr=0.1,
          adaptive=F,
          bandwidth=2.1,
          kernel="gaussian",
          #optimizer="sgd",
          cv_patience=1000,
          #iterations=100000,
          threads=8
)
p<-diag(r3$predictions)
print(p[1:5])


