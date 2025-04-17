
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

set.seed(1)

x_pred1 <- x[s_test,]
w_pred1 <- dm[-s_test, s_test]

g<-gwann(x_train=x[-s_test,],y_train=y[-s_test],w_train=dm[-s_test,-s_test],
         x_pred=x_pred1,w_pred=w_pred1,
         nrHidden=4,batchSize=50,lr=0.01,
         optimizer="nesterov",kernel="gaussian",
         adaptive=F,
         bandwidth=99,
         cv_patience=99,
         threads=15
)
p1<-diag(g$predictions)
print(p1[1:5])

p2<-diag(predict_gwann(g,x_pred1))
print(p2[1:5])

#################################################################

n<-nnet(x_train=x[-s_test,],y_train=y[-s_test],
         x_pred=x_pred1,
         nrHidden=4,batchSize=50,lr=0.01,
         optimizer="nesterov",
         cv_patience=99,
         threads=15
)
p1<-n$predictions[,1]
print(p1[1:5])

predict_gwann(n,x_pred1)

