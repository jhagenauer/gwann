#' Build a Geographically Weighted Artificial Neural Network.
#'
#' @param x_train Matrix of training data. Rows are observations, columns are independent variables.
#' @param y_train Vector. Values represent target values for the observations in \code{x_train}.
#' @param w_train Quadratic matrix of distances between the observations of \code{x_train}. The matrix solely used for calculating the adaptive distances.
#' @param x_pred Matrix of prediction data. Rows are observations, columns are independent variables.
#' @param w_pred Matrix of distances between the observations of \code{x_train} (rows) and \code{x_pred} (columns).
#' @param norm Center and scale variables before training? This affects the final model and the CV-procedure.
#' @param nrHidden Number of hidden neurons.
#' @param batchSize Batch size.
#' @param optimizer Optimizer (sgd, momentum, nesterov, adam).
#' @param lr Learning rate.
#' @param kernel Kernel (gaussian, bisquare, boxcar, exponential, tricube).
#' @param bandwidth Bandwidth size. If NA, it is determined using CV.
#' @param adaptive Adaptive instead of fixed bandwidth?
#' @param bwSearch Method for searching an appropriate bandwidth (goldenSection or grid). Ignored if bandwidth is explicitly given.
#' @param bwMin Lower limit for bandwidth search.
#' @param bwMax Upper limit for bandwidth search.
#' @param steps Number of bandwidths to test when doing a grid search/local search. Ignored if bandwidth is explicitly given or golden section search is used.
#' @param iterations Number of training iterations. If NA, it is determined using 10-fold CV. If given, \code{cv_max_iterations} and \code{cv_patience} are ignored.
#' @param cv_max_iterations Maximum number of iterations during CV.
#' @param cv_patience After how many iterations with no improvement should training during CV prematurely stop?
#' @param cv_folds Number of CV folds.
#' @param cv_repeats Number of repeats of CV.
#' @param permutations Number of permutations for calculating feature importance (EXPERIMENTAL and full of bugs. Do not use yet!).
#' @param threads Number of threads to use.
#' @return A list of several elements.
#' The first element \code{predictions} contains the predictions.
#' The second element \code{weights} contains the connection weights of the hidden neurons to the output neurons.
#' The third element \code{bandwidth} is the bandwidth that is used to train the final model.
#' The fourth element \code{iterations} is the number of training iterations for the final model.
#' The fifth element \code{seconds} is the number of seconds it took to build the final model.
#' The sixth element \code{gwann} is the trained GWANN model as Java object.
#' @examples
#' \dontrun{
#' #' data(toy4)
#'
#' dm<-as.matrix( dist(toy4[,c("lon","lat")])  )
#' x<-as.matrix(toy4[,c("x1","x2")])
#' y<-as.numeric( toy4[,c("y")] )
#'
#' r<-gwann(x_train=x,y_train=y,w_train=dm,x_pred=x,w_pred=dm,nrHidden=5,batchSize=100,lr=0.01,adaptive=F,bwSearch="goldenSection",bwMin=min(dm)/4, bwMax=max(dm)/4, steps=10,threads=8)
#' }
#' @references
#' Julian Hagenauer & Marco Helbich (2022) A geographically weighted artificial neural network, International Journal of Geographical Information Science, 36:2, 215-235, DOI: 10.1080/13658816.2021.1871618
#' @export
gwann<-function(x_train,y_train,w_train,x_pred,w_pred,norm=T,
                nrHidden=ncol(x_train)*2,batchSize=50,optimizer="nesterov",lr=0.05,linOut=T,
                kernel="gaussian",bandwidth=NA,adaptive=F,
                bwSearch="goldenSection", bwMin=NA, bwMax=NA, steps=20,adj=T,
                iterations=NA,
                cv_max_iterations=Inf,cv_patience=999,cv_folds=10,cv_repeats=1,
                permutations=0,
                threads=4) {

  # TODO Why not pass NA-values to java?
  if( is.na(bandwidth) )
    bandwidth<-(-1)
  if( is.na(iterations) )
    iterations<-(-1)
  if( is.na(bwMin) )
    bwMin<-(-1)
  if( is.na(bwMax) )
    bwMax<-(-1)

  if( any( apply(x_train,2,sd) == 0 ) ) warning("Zero variance column found in training data!")
  if( nrow(w_train) != ncol(w_train) ) stop("w_train must be quadratic!")
  if( nrow(w_pred) != nrow(x_train) & ncol(w_pred) != nrow(x_pred) ) stop(paste0("w_pred must have ",nrow(x_train), "rows and ",nrow(x_pred), " columns "))
  if( nrow(x_train) != length(y_train) ) stop("Number of rows of x_train does not match length of y_train")
  if( is.na(bandwidth) & !(bwSearch %in% c("goldenSection","grid") ) ) {
    warning("Unknown method for searching bw. Using golden section search.")
    bwSearch<-"goldenSection"
  }

  if(!exists(".Random.seed")) set.seed(NULL)
  seed<-.Random.seed[1]

  r<-rJava::.jcall(obj="supervised.nnet.gwann.GWANN_RInterface",method="run",returnSig = "Lsupervised/nnet/Return_R;",
            .jarray(x_train,dispatch=T),
            y_train,
            .jarray(w_train,dispatch=T),

            .jarray(x_pred,dispatch=T),
            .jarray(w_pred,dispatch=T),

            norm,
            nrHidden,batchSize,
            optimizer,
            lr,
            linOut,
            kernel,bandwidth,
            adaptive,
            bwSearch,
            bwMin,bwMax,steps,iterations,cv_max_iterations,cv_patience,cv_folds,cv_repeats,permutations,threads,
            seed
          )

  return(
    list(
      predictions=r$predictions,
      importance=r$importance,
      weights=r$weights,
      bandwidth=r$bw,
      iterations=r$its,
	    seconds=as.difftime(r$secs,units="secs"),
      ro=r$ro
    )
  )
}
#' Build a basic artifical neural network (experimental).
#'
#' @export
nnet<-function(x_train,y_train,x_pred,
                norm=T,
                nrHidden=ncol(x_train)*2,batchSize=50,optimizer="nesterov",lr=0.05,linOut=T,
                iterations=NA,
                cv_max_iterations=Inf,cv_patience=999,cv_folds=10,cv_repeats=1,
                permutations=0,
                threads=4) {

  if( is.na(iterations) )
    iterations<-(-1)

  if( any( apply(x_train,2,sd) == 0 ) ) warning("Zero variance column found in training data!")
  if( nrow(x_train) != length(y_train) ) stop("Number of rows of x_train does not match length of y_train")

  if(!exists(".Random.seed")) set.seed(NULL)
  seed<-.Random.seed[1]

  r<-rJava::.jcall(obj="supervised.nnet.NNet_RInterface",method="run",returnSig = "Lsupervised/nnet/Return_R;",
            .jarray(x_train,dispatch=T),
            y_train,

            .jarray(x_pred,dispatch=T),

            norm,
            nrHidden,batchSize,
            optimizer,
            lr,
            linOut,
            iterations,cv_max_iterations,cv_patience,cv_folds,cv_repeats,permutations,threads,
            seed
          )

  return(
    list(
      predictions=r$predictions,
      importance=r$importance,
      weights=r$weights,
      bandwidth=NA,
      iterations=r$its,
	    seconds=as.difftime(r$secs,units="secs"),
      ro=r$ro
    )
  )
}
#' Predict values for a trained GWANN/NNet model (experimental).
#'
#' Locations of observations must match with locations of the prediction data when the model was built.
#' @param ro GWANN model as Java object.
#' @param x_pred Matrix of prediction data. Rows are observations, columns are independent variables.
#' @export
predict_gwann<-function(model,x_pred) {
  ro<-model$ro
  if( rJava::`%instanceof%`(ro$nnet,"supervised.nnet.gwann.GWANN") ) {
    #print("GWANN")

    if( nrow(x_pred) != ro$nnet$weights[[2]] %>% ncol()) # assume 2 layers for now
      warning("Number of locations for predictions to not match!")

    r<-rJava::.jcall(obj="supervised.nnet.gwann.GWANN_RInterface",method="predict",returnSig = "[[D",
                     ro,
                     .jarray(x_pred,dispatch=T)
    )
    return(sapply(r,.jevalArray))

  } else {
    #print("NNet")

    r<-rJava::.jcall(obj="supervised.nnet.NNet_RInterface",method="predict",returnSig = "[[D",
                     ro,
                     .jarray(x_pred,dispatch=T)
    )
    return(sapply(r,.jevalArray))
  }
}
