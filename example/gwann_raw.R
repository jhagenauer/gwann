library(rJava)
.jinit()
.jaddClassPath("inst/java/commons-math3-3.6.1.jar")
.jaddClassPath("inst/java/gt-api-19.1.jar")
.jaddClassPath("inst/java/gt-data-19.1.jar")
.jaddClassPath("inst/java/gt-main-19.1.jar")
.jaddClassPath("inst/java/gwann-0.0.3-SNAPSHOT.jar")
.jaddClassPath("inst/java/jblas-1.2.4.jar")
.jaddClassPath("inst/java/jts-core-1.14.0.jar")
.jaddClassPath("inst/java/jts-io-1.14.0.jar")
.jaddClassPath("inst/java/log4j-1.2.17.jar")


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
bandwidth<-(-1)
opt="Nesterov"
threads<-8
batchSize<-50
lr<-0.01
linOut<-T
kernel<-"gaussian"
adaptive<-T
iterations<-(-1) #351
folds<-10
repeats<-1
patience<-100
bwSearch="goldenSection"
bwMin<-1
bwMax<-4
steps=1
permutations<-0

.jmethods("supervised.nnet.gwann.GWANN_RInterface")

r<-.jcall(obj="supervised.nnet.gwann.GWANN_RInterface",method="run",returnSig = "Lsupervised/nnet/gwann/ReturnObject;",

          .jarray(x_train,dispatch=T),
          y_train,
          .jarray(w_train,dispatch=T),

          .jarray(x_pred,dispatch=T),
          y_pred,
          .jarray(w_train_pred,dispatch=T),

          norm,nrHidden,batchSize,opt,lr,linOut,
          kernel,bandwidth,adaptive,
          bwSearch,

          bwMin,bwMax,steps,iterations,patience,folds,repeats,permutations,threads)
