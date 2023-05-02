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
    s_test<-sample(nrow(x),0.3*nrow(x)) # indices of test samples
    
    r<-gwann(x_train=x[-s_test,],y_train=y[-s_test],w_train=dm[-s_test,-s_test],
         x_pred=x[s_test,],w_pred=dm[-s_test,s_test],
         nrHidden=30,batchSize=50,lr=0.1,
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
- Transforming the data to make their distributions approximally normal often improves the performance of GWANN.

### References

Julian Hagenauer & Marco Helbich (2022) A geographically weighted artificial neural network, International Journal of Geographical Information Science, 36:2, 215-235, DOI: 10.1080/13658816.2021.1871618 
