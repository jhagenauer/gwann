# Geographically Weighted Artificial Neural Network

### System Requirements

Java JDK 1.2 or higher (for JRI/REngine JDK 1.4 or higher). If it is not already installed, you can get it [here](https://www.oracle.com/java/technologies/javase-downloads.html).

### Install
    Sys.setenv("R_REMOTES_NO_ERRORS_FROM_WARNINGS" = "true")
    if (!require("devtools"))
       install.packages("devtools",INSTALL_opts="--no-multiarch")
    devtools::install_github("jhagenauer/gwann")
    
### Example

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
    nrHidden=5,batchSize=100,lr=0.01,
    adaptive=F,
    bwSearch="goldenSection",
    bwMin=min(dm)/4, bwMax=max(dm)/4, steps=10,permutations=100,
    threads=1
    )
    p<-diag(r$predictions)
    print(paste("Out-of-sample RMSE: ",sqrt(mean((p-y[s_test])^2))))
    print(paste("In-sample RMSE: ",r$rmse))
    print(paste("Iterations: ",r$its))
    print(paste("Bandwidth: ",r$bw))
    
    # plot predictions
    s<-cbind( Prediction=p, toy4[s_test,c("lon","lat")] )
    ggplot(s,aes(lon,lat,fill=Prediction)) + geom_raster() + scale_fill_viridis() + coord_fixed()

### References

Hagenauer, Julian, and Marco Helbich. "A geographically weighted artificial neural network." International Journal of Geographical Information Science (2021): 1-21.
