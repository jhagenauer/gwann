#' Build a Geographically Weighted Artificial Neural Network.
#'
#' @param x Matrix. Rows are observations, columns are independent variables.
#' @param y Vector. Values represent target values for the observations in \code{x}.
#' @param dmX Matrix of distances between the observations of \code{x}.
#' @param dmP Matrix of distances between the observations of \code{x} (Rows) and some prediction locations (Columns).
#' @param nrHidden Number of hidden neurons.
#' @param batchSize Batch size.
#' @param lr Learning rate.
#' @param kernel Kernel.
#' @param bandwidth Bandwidth size. If na, a bandwidth is automatically determined.
#' @param adaptive Adaptive instead of fixed bandwidth?
#' @param maxIts Maximum number of iterations.
#' @param noImp After how many iterations with no improvement should training prematurly stop?
#' @param batchPerIt How many batch presentations per iteration?
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
#'   a<-cbind( t(r$weights), toy4[,c("lon","lat")] )
#'   m<-melt(a,id.vars=c("lon","lat"))
#'   x11()
#'   ggplot(m,aes(lon,lat,fill=value)) + geom_raster() + facet_wrap(~variable) + scale_fill_viridis()
#' }
#' }
#' @references
#' Not yet published
#' @export
gwann<-function(x,y,dmX,dmP,nrHidden=4,batchSize=10,lr=0.1,linOut=T,kernel="gaussian",bandwidth=NA,adaptive=F,maxIts=5000,noImp=100,batchPerIt=10,threads=4) {
  if( is.na(bandwidth) )
    bandwidth<-(-1)

  if( nrow(dmX) != ncol(dmX) ) stop("dmX must be quadratic!")
  if( nrow(dmX) != nrow(dmP) ) stop("dmX and dmP must have the same number of rows!")

  r<-.jcall(obj="supervised.nnet.gwann.GWANN_RInterface",method="run",returnSig = "Lsupervised/nnet/gwann/ReturnObject;",
            .jarray(x,dispatch=T),
            y,
            .jarray(dmX,dispatch=T),
            .jarray(dmP,dispatch=T),
            nrHidden,batchSize,lr,linOut,kernel,bandwidth,adaptive,maxIts,noImp,batchPerIt,threads)

  return(
    list(
      predictions=r$predictions,
      weights=r$weights,
      evaluations=r$evaluations,
      rmse=r$rmse,
      bw=r$bw,
      its=r$its
      )
    )
}
