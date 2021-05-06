#' Build a Geographically Weighted Artificial Neural Network.
#'
#' @param x Matrix. Rows are observations, columns are independent variables.
#' @param y Vector. Values represent target values for the observations in \code{x}.
#' @param dm Matrix of distances between the observations of \code{x}.
#' @param trainIdx a vector containing the indices of observations of \code{x} used for training.
#' @param predIdx a vector containing the indices of observations of \code{x} used for prediction.
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
#' r<-gwann(x=x,y=y,dm=dm,trainIdx=1:nrow(x),predIdx=1:nrow(x),nrHidden=5,batchSize=50,bandwidth=1.8,iterations=5610,lr=0.01)
#'
#' if( all ( sapply( c("ggplot2","reshape2","viridis"), require, character.only=T ) ) ) {
#'    s<-cbind( Prediction=diag(r$predictions), toy4[,c("lon","lat")] )
#'    ggplot(s,aes(lon,lat,fill=Prediction)) + geom_raster() + scale_fill_viridis() + coord_fixed()
#' }
#' }
#' @references
#' Hagenauer, Julian, and Marco Helbich. "A geographically weighted artificial neural network." International Journal of Geographical Information Science (2021): 1-21.
#' @export
gwann<-function(x,y,dm,trainIdx=1:nrow(dm),predIdx=1:nrow(dm),
                nrHidden=4,batchSize=10,optimizer="nesterov",lr=0.1,linOut=T,
                kernel="gaussian",bandwidth=NA,adaptive=F,
                gwSearch="goldenSection", bwMin=NA, bwMax=NA, steps=20,
                iterations=NA,patience=100,
                folds=10,repeats=1,
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

  if( nrow(dm) != ncol(dm) ) stop("dm must be quadratic!")
  if( length(y) != ncol(dm) ) stop("y must have the same length as dm rows!")
  if( any(is.na(y[trainIdx])) ) stop("trainIdx must not rever to any NAs in y!")
  if( !(bwSearch %in% c("goldenSection","grid","local") ) ) warning("Unknown method for searching bw. Using local search.")

  r<-.jcall(obj="supervised.nnet.gwann.GWANN_RInterface",method="run",returnSig = "Lsupervised/nnet/gwann/ReturnObject;",
            .jarray(x,dispatch=T),
            y,
            .jarray(dm,dispatch=T),
            trainIdx,
            predIdx,
            nrHidden,batchSize,optimizer,lr,linOut,
            kernel,bandwidth,adaptive,
            bwSearch,bwMin,bwMax,steps,
            iterations,patience,
            folds,repeats,
            threads)

  return(
    list(
      predictions=r$predictions,
      weights=r$weights,
      rmse=r$rmse,
      bw=r$bw,
      its=r$its
      )
    )
}
