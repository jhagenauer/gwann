gwann_nnet <- list(
                library = "gwann",
                loop = NULL,
                type = "Regression",
                parameters = data.frame( parameter = c("n","bs","lr","norm"),
                                         class = c("numeric","numeric","numeric","logical"),
                                         label = "# Neurons","Batch size","Learning rate","Normalize"),

                grid = function(x, y, len = NULL, search = "grid") {
                  if(search == "grid") {
                    out <- expand.grid(n=c(1,2,4,8),bs=c(10,50,100),lr=0.1,norm=T)

                  } else {
                    out <- data.frame(n=4,bs=10,lr=0.1,norm=T,opt="nesterov")
                  }
                  out
                },

                fit = function(x, y, wts, param, lev, last, weights, classProbs, ...) {
                  xn<-as.matrix(x)
                  gwann::nnet(
                          x_train=xn,y_train=y,
                          x_pred=xn[1:2,],
                          norm=param$norm,
                          nrHidden=param$n,batchSize=param$bs,lr=param$lr,
                          optimizer="nesterov",
                          cv_patience=99,
                          threads=15
                  )
                },

                predict = function(modelFit, newdata, preProc = NULL, submodels = NULL) {

                  out <- gwann::predict(modelFit$finalModel,newdata)
                  out
                },

                prob = NULL,
                tags = c("Neural Network"),
                varImp = NULL,
                sort = function(x) x
)

if( F ) {

  library(gwann)
  library(caret)
  library(tidyverse)
  library(mlbench)

  data(BostonHousing)

  pp<-c("center","scale","YeoJohnson")
  folds<-createMultiFolds(BostonHousing$medv,k=10,times=1)
  tc<-trainControl(index=folds, allowParallel = T, returnData = F, savePredictions = "final")

  t<-train(form=medv~crim+zn+indus+chas+nox+rm+age+dis,data=BostonHousing,method=gwann_nnet,preProcess=pp)
}
