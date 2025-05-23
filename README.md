# Geographically Weighted Artificial Neural Network

### System Requirements

Java JDK 1.2 or higher (for JRI/REngine JDK 1.4 or higher). If it is not already installed, you can get it [here](https://www.oracle.com/java/technologies/javase-downloads.html).

### Install
    Sys.setenv("R_REMOTES_NO_ERRORS_FROM_WARNINGS" = "true")
    if (!require("devtools"))
       install.packages("devtools",INSTALL_opts="--no-multiarch")
    devtools::install_github("jhagenauer/gwann")
    
### Example
    options(java.parameters="-Xmx8g")
    
    library(viridis)
    library(gwann)
    library(ggplot2)
    
    data(toy4)
    
    x<-as.matrix(toy4[,c("x1","x2")])
    y<-as.numeric(toy4[,c("y")] )
    dm<-as.matrix(dist(toy4[,c("lon","lat")])  )
    idx_pred<-sample(nrow(x),0.3*nrow(x)) # indices of prediction samples
    
    r<-gwann(x_train=x[-idx_pred,],y_train=y[-idx_pred],w_train=dm[-idx_pred,-idx_pred],
         x_pred=x[idx_pred,],w_pred=dm[-idx_pred,idx_pred],
         nrHidden=4,batchSize=50,lr=0.1,optimizer="adam",cv_patience=9999,
         adaptive=F,
         bwSearch="goldenSection",bwMin=min(dm)/4, bwMax=max(dm)/4,
         threads=8
    )
    p<-diag(r$predictions)
    print(paste("RMSE: ",sqrt(mean((p-y[s_test])^2))))
    print(paste("Iterations: ",r$iterations))
    print(paste("Bandwidth: ",r$bandwidth))
    
    # plot predictions
    s<-cbind( Prediction=p, toy4[s_test,c("lon","lat")] )
    ggplot(s,aes(lon,lat,fill=Prediction)) + geom_raster() + scale_fill_viridis() + coord_fixed()

### Note

- If you get `java.lang.OutOfMemoryError: Java heap space` put `options(java.parameters="-Xmx8g")` before loading the package and adjust it to your available memory. To take effect, you most likely have to restart R/RStudio then.
- The learning rate (lr), the batch size (batchSize) and the number of hidden neurons (nrHidden) have a substantial effect on the performance and therefore should be chosen carefully. (The number of iterations as well as the bandwidth are also important but are by default automatically determined by GWANN using cross-validation.)
- Test vey different batch sizes, ranging from 1, 5, 10, to 100
- Transforming the data to make their distributions approximally normal often improves the performance of GWANN.
- 'nesterov' optimizer has shown to be most effective.

### References

Julian Hagenauer & Marco Helbich (2022) A geographically weighted artificial neural network, International Journal of Geographical Information Science, 36:2, 215-235, DOI: 10.1080/13658816.2021.1871618 
