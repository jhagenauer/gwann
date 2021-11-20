#' Build a Geographically Weighted Artificial Neural Network.
#'
#' @param x_train Matrix of training data. Rows are observations, columns are independent variables.
#' @param y_train Vector. Values represent target values for the observations in \code{x_train}.
#' @param w_train Quadratic matrix of distances between the observations of \code{x_train}. The matrix solely used for calculating the adaptive distances.
#' @param x_pred Matrix of prediction data. Rows are observations, columns are independent variables.
#' @param y_pred Vector. Values represent target values for the observations in \code{x_pred}.
#' @param w_train_pred Matrix of distances between the observations of \code{x_train} (rows) and \code{x_pred} (columns).
#' @param norm Center and scale independent variables before training? This affects the final model and the CV-procedure.
#' @param nrHidden Number of hidden neurons.
#' @param batchSize Batch size.
#' @param optimizer Optimizer (sgd, momentum, nesterov).
#' @param lr Learning rate.
#' @param kernel Kernel.
#' @param bandwidth Bandwidth size. If NA, it is determined using CV.
#' @param adaptive Adaptive instead of fixed bandwidth?
#' @param bwSearch Method for searching an appropriate bandwidth (goldenSection, grid, local). Ignored if bandwidth is explicitly given.
#' @param bwMin Lower limit for bandwidth search.
#' @param bwMax Upper limit for bandwidth search.
#' @param steps Number of bandwidths to test when doing a grid search/local search. Ignored if bandwidth is explicitly given or golden section search is used.
#' @param iterations Number of training iterations. If NA, it is determined using 10-fold CV.
#' @param patience After how many iterations with no improvement should training prematurely stop?
#' @param folds Number of cross-validation folds
#' @param repeats Number of repeats of cross-validation procedure
#' @param permutations Number of permutations for calculating feature importance (Experimental)
#' @param threads Number of threads to use.
#' @return A list of five elements.
#' The first element \code{predictions} contains the predictions.
#' The second elemnt \code{weights} contains the connection weights of the hidden neurons to the output neurons.
#' The third element \code{rmse} is the mean RMSE of the CV procedure.
#' The fourth element \code{bandwidth} is the bandwidth that is used to train the final model.
#' The fifth element \code{iterations} is the numer of training iterations for the final model.
#' @examples
#' data(toy4)
#'
#' dm<-as.matrix( dist(toy4[,c("lon","lat")])  )
#' x<-as.matrix(toy4[,c("x1","x2")])
#' y<-as.numeric( toy4[,c("y")] )
#'
#' \dontrun{
#' r<-gwann(x_train=x_train,y_train=y_train,w_train=dm,x_pred=x_train,y_pred=y_train,w_train_pred=dm,nrHidden=5,batchSize=100,lr=0.01,adaptive=F,#bandwidth=10,bwSearch="goldenSection",bwMin=min(dm)/4, bwMax=max(dm)/4, steps=10,threads=8)
#'
#' if( all ( sapply( c("ggplot2","reshape2","viridis"), require, character.only=T ) ) ) {
#'    s<-cbind( Prediction=diag(r$predictions), toy4[,c("lon","lat")] )
#'    ggplot(s,aes(lon,lat,fill=Prediction)) + geom_raster() + scale_fill_viridis() + coord_fixed()
#' }
#' }
#' @references
#' Hagenauer, Julian, and Marco Helbich. "A geographically weighted artificial neural network." International Journal of Geographical Information Science (2021): 1-21.
#' @export
gwann<-function(x_train,y_train,w_train=NA,x_pred,y_pred=NA,w_train_pred,norm=T,
                nrHidden=4,batchSize=10,optimizer="nesterov",lr=0.1,linOut=T,
                kernel="gaussian",bandwidth=NA,adaptive=F,
                bwSearch="goldenSection", bwMin=NA, bwMax=NA, steps=20,
                iterations=NA,patience=100,
                folds=10,repeats=1,permutations=0,
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
  if( all(is.na(y_pred)) )
    y_pred<-as.numeric( rep(NA,nrow(x_pred)) )

  if( nrow(w_train) != ncol(w_train) ) stop("w_train must be quadratic!")
  if( is.na(bandwidth) & !(bwSearch %in% c("goldenSection","grid","local") ) ) {
    warning("Unknown method for searching bw. Using golden section search.")
    bwSearch<-"goldenSection"
  }

  r<-.jcall(obj="supervised.nnet.gwann.GWANN_RInterface",method="run",returnSig = "Lsupervised/nnet/gwann/ReturnObject;",
            .jarray(x_train,dispatch=T),
            y_train,
            .jarray(w_train,dispatch=T),

            .jarray(x_pred,dispatch=T),
            y_pred,
            .jarray(w_train_pred,dispatch=T),

            norm,nrHidden,batchSize,optimizer,lr,linOut,
            kernel,bandwidth,adaptive,
            bwSearch,bwMin,bwMax,steps,
            iterations,patience,
            folds,repeats,
            permutations,
            threads)

  return(
    list(
      predictions=r$predictions,
      importance=r$importance,
      weights=r$weights,
      rmse=r$rmse,
      bw=r$bw,
      its=r$its
    )
  )
}
