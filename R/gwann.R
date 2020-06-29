#' Build a Geographically Weighted Artificial Neural Network.
#'
#' @param x Matrix. Rows are observations, columns are independent variables.
#' @param y Vector. Values represent target values for the observations in \code{x}.
#' @param dm Matrix of distances between the observations of \code{x}.
#' @param trainIdx a vector containing the indices of observations of \code{x} used for training.
#' @param predIdx a vector containing the indices of observations of \code{x} used for prediction.
#' @param nrHidden Number of hidden neurons.
#' @param batchSize Batch size.
#' @param optimizer Optimizer.
#' @param lr Learning rate.
#' @param kernel Kernel.
#' @param bandwidth Bandwidth size. If NA, it is determined using 10-fold CV.
#' @param adaptive Adaptive instead of fixed bandwidth?
#' @param iterations Number of training iterations. If NA, is determined using 10-fold CV.
#' @param patience After how many iterations with no improvement should training prematurely stop?
#' @param threads Number of threads to use.
#' @return A list of five elements.
#' The first element \code{predictions} contains the predictions.
#' The second elemnt \code{weights} contains the connection weights of the hidden neurons to the output neurons.
#' The third element \code{rmse} is the mean RMSE of the 10-fold CV procedure.
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
#' Not yet published
#' @export
gwann<-function(x,y,dm,trainIdx=1:nrow(dm),predIdx=1:nrow(dm),nrHidden=4,batchSize=10,optimizer="nesterov",lr=0.1,linOut=T,kernel="gaussian",bandwidth=NA,adaptive=F,iterations=NA,patience=100,threads=4) {
  if( is.na(bandwidth) )
    bandwidth<-(-1)
  if( is.na(iterations) )
    iterations<-(-1)

  if( nrow(dm) != ncol(dm) ) stop("dm must be quadratic!")
  if( length(y) != ncol(dm) ) stop("y must have the same length as dm rows!")
  if( any(is.na(y[trainIdx])) ) stop("trainIdx must not rever to any NAs in y!")


  r<-.jcall(obj="supervised.nnet.gwann.GWANN_RInterface",method="run",returnSig = "Lsupervised/nnet/gwann/ReturnObject;",
            .jarray(x,dispatch=T),
            y,
            .jarray(dm,dispatch=T),
            trainIdx,
            predIdx,
            nrHidden,batchSize,optimizer,lr,linOut,kernel,bandwidth,adaptive,iterations,patience,threads)

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
