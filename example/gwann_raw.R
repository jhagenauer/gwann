library(rJava)
.jinit()
.jaddClassPath("inst/java/commons-math3-3.6.1.jar")
.jaddClassPath("inst/java/gwann-0.0.4-SNAPSHOT.jar")
.jaddClassPath("inst/java/jblas-1.2.4.jar")
.jaddClassPath("inst/java/log4j-api-2.17.1.jar")
.jaddClassPath("inst/java/log4j-core-2.17.1.jar")

.jmethods("supervised.nnet.gwann.GWANN_RInterface")

data(toy4)

dm<-as.matrix( dist(toy4[,c("lon","lat")])  )
x<-as.matrix(toy4[,c("x1","x2")])
y<-as.numeric( toy4[,c("y")] )

x_train<-x
y_train<-y
w_train<-dm

x_pred<-x
y_pred<-y
w_train_pred<-dm

nrHidden<-5
norm<-T
bandwidth<-10
opt="Nesterov"
threads<-8
batchSize<-50
lr<-0.01
linOut<-T
kernel<-"gaussian"
adaptive<-T
iterations<-(-1)
folds<-10
repeats<-1
patience<-1000
bwSearch="goldenSection"
bwMin<-1
bwMax<-4
steps=1
permutations<-0

r<-.jcall(obj="supervised.nnet.gwann.GWANN_RInterface",method="run",returnSig = "Lsupervised/nnet/gwann/Return_R;",

          .jarray(x_train,dispatch=T),
          y_train,
          .jarray(w_train,dispatch=T),

          .jarray(x_pred,dispatch=T),
          y_pred,
          .jarray(w_train_pred,dispatch=T),

          norm,nrHidden,batchSize,opt,lr,linOut,
          kernel,bandwidth,adaptive,
          bwSearch,

          bwMin,bwMax,steps,iterations,patience,folds,repeats,permutations,threads
          )
