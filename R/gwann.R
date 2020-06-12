#' Build a Geographically Weighted Artificial Neural Network.
#'
#' @param x Matrix. Rows are observations, columns are independent variables.
#' @param y Vector. Values represent target values for the observations in \code{x}.
#' @param dmX Matrix of distances between the observations of \code{x}.
#' @param dmP Matrix of distances between the observations of \code{x} (Rows) and some prediction locations (Columns).
#' @param nrHidden Number of hidden neurons.
#' @param batchSize Batch size.
#' @param optimizer Optimizer.
#' @param lr Learning rate.
#' @param kernel Kernel.
#' @param bandwidth Bandwidth size. If NA, it is determined using 10-fold CV.
#' @param adaptive Adaptive instead of fixed bandwidth?
#' @param iterations Number of training iterations. If NA, is determined using 10-fold CV.
#' @param patience After how many iterations with no improvement should training prematurly stop?
#' @param threads Number of threads to use.
#' @return A list with two elements.
#' The first element is the vector \code{predictions} which contains the predictions for the locations defined by \code{dmP}.
#' The second elemnt is the matrix \code{weights} which contains the connection weights of the hidden neurons to the output neurons. The output neurons refer to the positions defined by \code{dmP}.
#' @examples
#' data(toy4)
#'
#' dm<-as.matrix( dist(toy4[,c("lon","lat")])  )
#' x<-as.matrix(toy4[,c("x1","x2")])
#' y<-as.numeric( toy4[,c("y")] )
#'
#' \dontrun{
#' r<-gwann(x=x,y=y,dmX=dm,dmP=dm,nrHidden=8,batchSize=50,bandwidth=1.49)
#'
#' if( all ( sapply( c("ggplot2","reshape2","viridis"), require, character.only=T ) ) ) {
#'    s<-cbind( Prediction=diag(r$predictions), toy4[,c("lon","lat")] )
#'    ggplot(s,aes(lon,lat,fill=Prediction)) + geom_raster() + scale_fill_viridis() + coord_fixed()
#' }
#' }
#' @references
#' Not yet published
#' @export
gwann<-function(x,y,dmX,dmP,nrHidden=4,batchSize=10,optimizer="nesterov",lr=0.1,linOut=T,kernel="gaussian",bandwidth=NA,adaptive=F,iterations=NA,patience=100,threads=4) {
  if( is.na(bandwidth) )
    bandwidth<-(-1)
  if( is.na(iterations) )
    iterations<-(-1)

  if( nrow(dmX) != ncol(dmX) ) stop("dmX must be quadratic!")
  if( nrow(dmX) != nrow(dmP) ) stop("dmX and dmP must have the same number of rows!")

  r<-.jcall(obj="supervised.nnet.gwann.GWANN_RInterface",method="run",returnSig = "Lsupervised/nnet/gwann/ReturnObject;",
            .jarray(x,dispatch=T),
            y,
            .jarray(dmX,dispatch=T),
            .jarray(dmP,dispatch=T),
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
