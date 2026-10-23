package supervised.nnet.gwann;

import java.util.ArrayList;
import java.util.List;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.jblas.DoubleMatrix;

import supervised.nnet.NNet;
import supervised.nnet.NNetUtils;
import supervised.nnet.activation.Function;
import utils.GWUtils.GWKernel;

public class GWANN extends NNet {
	
	private static Logger log = LogManager.getLogger(GWANN.class);

	public static boolean f = true;
		
	public GWANN(Function[][] l, double[][][] weights, double[] eta, Optimizer m, double lambda ) {
		super(l,weights,eta,m,lambda); 
	}
	
	double bw_min, bw_max;
	
	double[][][] vWeights = null, mWeights = null;
	double[] vBW = null, mBW = null;
	
	DoubleMatrix W;
	GWKernel kernel;
	public double bw[];
		
	public GWANN(Function[][] ll, double[][][] weights, double[] eta, Optimizer m, double lambda, DoubleMatrix w, GWKernel kernel, double bw_min, double bw_max ) {
		super(ll,weights,eta,m,lambda);
		this.W = w;
		this.kernel = kernel;
		
		this.bw_min = bw_min;
		this.bw_max = bw_max;
		
		bw = new double[ll[ll.length-2].length];
		for( int i = 0; i < bw.length; i++ ) 
			bw[i] =  bw_max;	
		//bw = new double[] {5.156213328754902, 2.7602200556014242, 21.602274079470504};
			
		this.vWeights = new double[weights.length][][];
		this.mWeights = new double[weights.length][][];
		for (int i = 0; i < weights.length; i++) {
			this.vWeights[i] = new double[weights[i].length][];
			this.mWeights[i] = new double[weights[i].length][];

			for (int j = 0; j < weights[i].length; j++) {
				this.vWeights[i][j] = new double[weights[i][j].length];
				this.mWeights[i][j] = new double[weights[i][j].length];
			}
		}
	    
        mBW = new double[bw.length];
        vBW = new double[bw.length];
	}
	

	private double[][][] getErrorGradient_no_hidden( double[] x, double[] desired, double[][] sw, int h ) {
	    double[][] out = presentInt( x, weights )[0];
	    double[][][] error_grad = new double[1][layer[0].length][layer[1].length];
	     
	    for (int i = 0; i < layer[1].length; i++) 
	           error_grad[0][h][i] = out[0][h] * (out[1][i] - desired[i]) * sw[h][i];

	    return error_grad;
	}
		
	private double[][][] getErrorGradient_bf(double[] x, double[] desired, double[][] sw, int targetFeature) {
	    double[][] out = presentInt(x, weights)[0];
	    int numLocations = layer[1].length;
	    int numFeatures = layer[0].length;

	    double[][][] error_grad = new double[1][numFeatures][numLocations];

	    for (int i = 0; i < numLocations; i++) {
	        // 1. Calculate the 'Partial Residual' for this specific feature
	        // Start with the ground truth for that location
	        double partialResidual = desired[i]; 
	        
	        // Subtract the influence of all OTHER features
	        for (int h = 0; h < numFeatures; h++) 
	            if (h != targetFeature) 
	            	partialResidual -= weights[0][h][i] * out[0][h];
	            
	        // 2. Compute gradient: (Predicted_Component - Partial_Residual) * Input * Spatial_Weight
	        // This isolates how much feature 'targetFeature' needs to move to close the gap
	        double currentComponent = weights[0][targetFeature][i] * out[0][targetFeature];
	        double error = currentComponent - partialResidual;

	        error_grad[0][targetFeature][i] = out[0][targetFeature] * error * sw[targetFeature][i];
	    }

	    return error_grad;
	}
			
	double[][][] getErrorGradient( double[] x, double[] desired, double[] sw ) {
		int ll = layer.length - 1; // index of last layer (neuron)
		double[][] delta = new double[layer.length][];
		double[][][] error_grad = new double[layer.length-1][][];		
		for( int l = ll; l > 0; l-- ) {
			delta[l-1] = new double[layer[l].length];
			error_grad[l-1] = new double[layer[l - 1].length][layer[l].length];
		}
		
		double[][] out = presentInt( x, weights )[0];
		for (int l = ll; l > 0; l--) {	
			
			for (int i = 0; i < layer[l].length; i++) { 	
				
				double error_signal = 0;				
				if( l == ll ) {
					error_signal = (out[l][i] - desired[i]) * sw[i];
				} else {
					for (int j = 0; j < weights[l][i].length; j++)
						if( !layer[l+1][j].isConstant() ) 
							error_signal += delta[l][j] * weights[l][i][j];		
				}
				delta[l-1][i] = layer[l][i].fDevFOut(out[l][i]) * error_signal;
				
				for( int h = 0; h < layer[l-1].length; h++ ) 
					error_grad[l-1][h][i] = out[l-1][h] * delta[l-1][i];				
			}				
		}				
		return error_grad;
	}
			
	double[][][] getErrorGradient( double[] x, double[] desired, double[][] sw ) {
		int ll = layer.length - 1; // index of last layer
		double[][] delta = new double[ll][];
		double[][][] error_grad = new double[ll][][];		
		for( int l = ll; l > 0; l-- ) {
			delta[l-1] = new double[layer[l].length];
			error_grad[l-1] = new double[layer[l - 1].length][layer[l].length];
		}
				
		double[][] out = presentInt( x, weights )[0]; 	
		for (int l = ll; l > 0; l--) {
			for (int i = 0; i < layer[l].length; i++) {

				double error_signal = 0.0;
				if (l == ll) {
					error_signal = (out[l][i] - desired[i]);
				} else {
					for (int j = 0; j < weights[l][i].length; j++) {
						if( !layer[l+1][j].isConstant() )
							error_signal += delta[l][j] * weights[l][i][j];
					}
				}
				delta[l - 1][i] = layer[l][i].fDevFOut(out[l][i]) * error_signal;

				for (int h = 0; h < layer[l - 1].length; h++)
					error_grad[l-1][h][i] = out[l-1][h] * delta[l-1][i] * ( l==ll ? sw[h][i] : 1.0);	
			}
		}
		return error_grad;
	}
	
	double[][][] getErrorGradient_new( double[] x, double[] desired, double[][] sw ) {
		int ll = layer.length - 1; // index of last layer
		double[][] delta = new double[ll][];
		double[][][] error_grad = new double[ll][][];		
		
		// 1. Initialize delta and error_grad arrays
		for( int l = ll; l > 0; l-- ) {
			delta[l-1] = new double[layer[l].length];
			error_grad[l-1] = new double[layer[l - 1].length][layer[l].length];
		}
				
		// 2. Forward pass
		double[][] out = presentInt( x, weights )[0]; 	
		
		// 3. Backward pass
		for (int l = ll; l > 0; l--) {
			for (int i = 0; i < layer[l].length; i++) {

				double error_signal = 0.0;
				
				// Calculate error signal coming into the current node
				if (l == ll) {
					// Output layer error
					error_signal = (out[l][i] - desired[i]);
				} else {
					// Hidden layer error: pull delta from the next layer
					for (int j = 0; j < weights[l][i].length; j++) {
						if( !layer[l+1][j].isConstant() ) {
							
							double effective_weight = weights[l][i][j];
							
							// CHAIN RULE FIX: If propagating error from the final layer, 
							// the connection's spatial weight must be factored in.
							if (l == ll - 1) {
								effective_weight *= sw[i][j]; 
							}							
							error_signal += delta[l][j] * effective_weight;
						}
					}
				}
				
				// Compute delta for current node
				delta[l - 1][i] = layer[l][i].fDevFOut(out[l][i]) * error_signal;

				// Compute gradient for weights feeding into current node
				for (int h = 0; h < layer[l - 1].length; h++) {
					
					// Apply spatial weight only to the final layer's connections
					double spatial_modifier = (l == ll) ? sw[h][i] : 1.0;					
					error_grad[l-1][h][i] = out[l-1][h] * delta[l-1][i] * spatial_modifier;
				}
			}
		}
		
		return error_grad;
	}

	private double[][][] getErrorGradient_( double[] x, double[] desired, double[][] sw ) {
	    double[][] out = presentInt( x, weights )[0];
	    double[][][] error_grad = new double[1][layer[0].length][layer[1].length];
	     
	    for (int i = 0; i < layer[1].length; i++) 
	    	 for (int h = 0; h < layer[0].length; h++)
	            error_grad[0][h][i] = out[0][h] * (out[1][i] - desired[i]) * sw[h][i];

	    return error_grad;
	}
		
	public static boolean norm_grad = false;
				
	int kk = 1;
	
	public static double calculateMagnitude(double[][][] errorGrad) {
        if (errorGrad == null || errorGrad.length == 0) {
            return 0.0;
        }

        double sumOfSquares = 0.0;

        for (int i = 0; i < errorGrad.length; i++) {
            // Check for null or empty second dimension
            if (errorGrad[i] == null) continue;

            for (int j = 0; j < errorGrad[i].length; j++) {
                // Check for null or empty third dimension
                if (errorGrad[i][j] == null) continue;

                for (int k = 0; k < errorGrad[i][j].length; k++) {
                    double value = errorGrad[i][j][k];
                    sumOfSquares += value * value;
                }
            }
        }

        return Math.sqrt(sumOfSquares);
    }
	
	@Deprecated
	void train_( List<double[]> x, List<double[]> y, List<double[][]>  gwWeights, int feature, double[] eta ) {
		List<double[][][]> l = new ArrayList<>();
		for( int i = 0; i < x.size(); i++ ) 
			l.add( getErrorGradient_bf( x.get(i), y.get(i), gwWeights.get(i), feature ) );
		double[][][] errorGrad = !norm_grad ? NNetUtils.calculateMeanOf3DList(l) : GWANNUtils.calculateSpatiallyNormalizedGradient(l);
		update(opt, errorGrad, eta);		
	}
		
	@Override
	public double[] getResiduals(List<double[]> inputs, List<double[]> targets) {
	    int output_size = targets.get(0).length;
	    assert output_size == 1;

	    double[] residuals = new double[inputs.size()];
	    for (int i = 0; i < inputs.size(); i++) {	    	
	        double[] output = present(inputs.get(i));
	        residuals[i] = output[i] - targets.get(i)[0];	        
	    }
	    return residuals;
	}
}