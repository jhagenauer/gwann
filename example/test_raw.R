# For internal testing. Do not run!!!

library(rJava)
.jinit()
.jaddClassPath("inst/java/commons-math3-3.6.1.jar")
.jaddClassPath("inst/java/gwann-0.6-SNAPSHOT.jar")
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
w_train_pred<-dm

norm<-T
nrHidden<-4
batchSize<-40
optimizer<-"adam"
lr<-0.1
linOut<-T
kernel<-"gaussian"
bandwidth<-5
adaptive<-T
bwSearch<-"goldenSection"
bwMin<-4
bwMax<-20
steps<-4
iterations<-100
cv_max_iterations<-100
cv_patience<-99
cv_folds<-10
cv_repeats<-1
permutations<-0
threads<-14

if(!exists(".Random.seed")) set.seed(NULL)
seed<-.Random.seed[1]

r<-.jcall(obj="supervised.nnet.gwann.GWANN_RInterface",method="run",returnSig = "Lsupervised/nnet/gwann/Return_R;",

          # [[D
          .jarray(x_train,dispatch=T),
          # [D
          y_train,
          # [[D
          .jarray(w_train,dispatch=T),.jarray(x_pred,dispatch=T),.jarray(w_train_pred,dispatch=T),

          #Z
          norm,

          #D
          nrHidden,batchSize,

          #S
          optimizer,

          #D
          lr,

          #Z
          linOut,

          #S
          kernel,

          #D
          bandwidth,

          #Z
          adaptive,

          #S
          bwSearch,

          #D
          bwMin,bwMax,steps,iterations,cv_max_iterations,cv_patience,cv_folds,cv_repeats,permutations,threads,

          #I
          seed
)

head( diag(r$predictions) )

p<-.jcall(obj="supervised.nnet.gwann.GWANN_RInterface",method="predict",returnSig = "[[D",
          r$gwann,
          .jarray(x_pred,dispatch=T)
)
p2<-sapply(p,.jevalArray)

head( diag(p2) )

