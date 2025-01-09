gwann_nnet <- list(
                library = "gwann",
                label ="Basic aritficial neural network",
                loop = NULL,
                type = "Regression",
                parameters = data.frame( parameter = c("n","bs","lr","norm","opt"),
                                         class = c("numeric","numeric","numeric","logical","character"),
                                         label = "# Neurons","Batch size","Learning rate","Normalize"),

                grid = function(x, y, len = NULL, search = "grid") {
                  if(search == "grid") {
                    out <- expand.grid(n=c(1,2,4,8),bs=c(5,25,50,100),lr=0.1,opt="nesterov",norm=T)

                  } else {
                    out <- data.frame(n=4,bs=10,lr=0.1,norm=T,opt="nesterov")
                  }
                  out
                },

                fit = function(x, y, wts, param, lev, last, weights, classProbs, ...) {
                  xn<-as.matrix(x)
                  gwann::nnet(
                          x_train=xn, # [[D
                          y_train=y, # [D
                          x_pred=xn[1:2,], # [[D
                          norm=param$norm, # Z
                          nrHidden=param$n,batchSize=param$bs, # D
                          optimizer=as.character(param$opt), # S
                          lr=param$lr,
                          linOut=T,
                          threads=15
                  )
                },

                predict = function(modelFit, newdata, preProc = NULL, submodels = NULL) {
                  out <- gwann::predict_gwann(modelFit,as.matrix(newdata))
                  out
                },

                prob = NULL,
                tags = c("Neural Network"),
                varImp = NULL,
                sort = function(x) x
)

if( F ) {

  if( F ) {
    roxygen2::roxygenise()
    detach("package:gwann", unload=TRUE)
    remove.packages("gwann")
    devtools::install_local(".")
  }

  library(gwann)
  library(caret)
  library(tidyverse)
  library(mlbench)

  data(BostonHousing)

  folds<-createMultiFolds(BostonHousing$medv,k=10,times=1)
  tc<-trainControl(index=folds, allowParallel = T, returnData = F, savePredictions = "final")

  #t<-train(form=medv~crim+zn+indus+nox+rm+age+dis,data=BostonHousing,method=gwann_nnet,preProcess=c("center","scale","YeoJohnson"))
  t<-train(x=BostonHousing[,c("crim","zn","indus","nox")],y=BostonHousing$medv,method=gwann_nnet,tuneGrid = expand.grid(n=4,bs=4,lr=0.1,opt="nesterov",norm=T))
}

