package supervised.nnet.gwann;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.jblas.DoubleMatrix;

import supervised.nnet.NNet;
import supervised.nnet.activation.Function;
import utils.GWUtils;
import utils.GWUtils.GWKernel;

public class GWANN extends NNet {
	
	private static Logger log = LogManager.getLogger(GWANN.class);
		
	public GWANN(Function[][] l, double[][][] weights, double[] eta, Optimizer m, double lambda ) {
		super(l,weights,eta,m,lambda); 
	}
	
	int bw_min, bw_max;
	
	double[][][] vWeights = null, mWeights = null;
	double[] vBW = null, mBW = null;
	
	DoubleMatrix W;
	GWKernel kernel;
	public double bw[];
		
	public GWANN(Function[][] ll, double[][][] weights, double[] eta, Optimizer m, double lambda, DoubleMatrix w, GWKernel kernel, int bw_min, int bw_max ) {
		super(ll,weights,eta,m,lambda);
		this.W = w;
		this.kernel = kernel;
		
		this.bw_min = bw_min;
		this.bw_max = bw_max;
		
		bw = new double[ll[ll.length-2].length];
		for( int i = 0; i < bw.length; i++ ) 
			bw[i] =  i*3+w.max()/3;	
		//bw = new double[] {5.791623676580325, 2.462568817211476, 321.9918945534378};
			
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
		
	private double[][][] getErrorGradient( double[] x, double[] desired, double[] sw ) {
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
		
	private double[][][] getErrorGradient( double[] x, double[] desired, double[][] sw ) {
		int ll = layer.length - 1; // index of last layer
		double[][] delta = new double[layer.length][];
		double[][][] error_grad = new double[layer.length-1][][];		
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
						if( !layer[l+1][j].isConstant() ) {
							if (l == ll - 1) {
								error_signal += delta[l][j] * weights[l][i][j] * sw[i][j]; // i hidden neuron, j output neuron
							} else
								error_signal += delta[l][j] * weights[l][i][j];
						}
					}
				}
				delta[l - 1][i] = layer[l][i].fDevFOut(out[l][i]) * error_signal;

				// keep gradients identical: for edges into the output layer, scale the gradient by sw
				for (int h = 0; h < layer[l - 1].length; h++) {
					if (l == ll) {
						error_grad[l - 1][h][i] = out[l - 1][h] * delta[l - 1][i] * sw[h][i];
					} else
						error_grad[l - 1][h][i] = out[l - 1][h] * delta[l - 1][i];
				}
			}
		}
		return error_grad;
	}
	
	private Object[] getErrorGradient_( double[] x, double[] desired, double[] w ) {
		int ll = layer.length - 1; // index of last layer
		double[][] delta = new double[layer.length][];
		double[][][] error_grad = new double[layer.length-1][][];		
		for( int l = ll; l > 0; l-- ) {
			delta[l-1] = new double[layer[l].length];
			error_grad[l-1] = new double[layer[l - 1].length][layer[l].length];
		}
		
	    double[] bw_grad = new double[layer[ll - 1].length];
				
		double[][] out = presentInt( x, weights )[0];							
		for (int l = ll; l > 0; l--) {
			for (int i = 0; i < layer[l].length; i++) {

				double error_signal = 0.0;
				if (l == ll) {
					error_signal = (out[l][i] - desired[i]);
				} else {
					for (int j = 0; j < weights[l][i].length; j++) {
						if( !layer[l+1][j].isConstant() ) {
							if (l == ll - 1) {
								double kVal = GWUtils.getKernelValue(kernel, W.get(i, j), bw[i]);
								error_signal += delta[l][j] * weights[l][i][j] * kVal;
							} else
								error_signal += delta[l][j] * weights[l][i][j];
						}
					}
				}
				delta[l - 1][i] = layer[l][i].fDevFOut(out[l][i]) * error_signal;

				for (int h = 0; h < layer[l - 1].length; h++) {
					if (l == ll) {
						
	                    double kVal = GWUtils.getKernelValue(kernel, w[i], bw[h]);	                    
	                    error_grad[l - 1][h][i] = out[l - 1][h] * delta[l - 1][i] * kVal;
	                    	                    
	                    double dK_dbw = GWUtils.getKernelDerivateValue(kernel, w[i], bw[h]); 
	                    if( use_log )
	                    	bw_grad[h] += delta[l - 1][i] * out[l - 1][h] * weights[l - 1][h][i] * dK_dbw * bw[h]; // log space
	                    else
	                    	bw_grad[h] += delta[l - 1][i] * out[l - 1][h] * weights[l - 1][h][i] * dK_dbw;
	                    
					} else
						error_grad[l - 1][h][i] = out[l - 1][h] * delta[l - 1][i];
				}
			}
		}
		return new Object[] {
				error_grad,
				bw_grad
		};
	}
	
	public static boolean use_log = false;
	
	public void train( List<double[]> x, List<double[]> y) {
		List<double[][][]> l = new ArrayList<>();
		List<double[]> l2 = new ArrayList<>();
		for( int i = 0; i < x.size(); i++ ) {
			Object[] o = getErrorGradient_( x.get(i), y.get(i), W.getRow(i).data );
			l.add( ((double[][][])(o[0])) );
			l2.add( ((double[])(o[1])) );
		}
		double[][][] errorGrad = calculateMeanOf3DList(l);		
		
		double[] bwGrad = new double[l2.get(0).length];
		for( double[] d : l2 )
			for( int i = 0; i < d.length; i++ )
				bwGrad[i] += d[i]/l2.size();
				
		update(opt, errorGrad, bwGrad, eta);
	}
		
	protected void update( Optimizer opt, double[][][] errorGrad, double[] bwGrad, double[] eta ) {
		t++;
		
		if( opt == Optimizer.SGD ) {
						
				// update weights
				for (int l = 0; l < weights.length; l++)
					for (int i = 0; i < weights[l].length; i++)
						for (int j = 0; j < weights[l][i].length; j++) 
							weights[l][i][j] -= eta[l] * errorGrad[l][i][j];
			
				// update bandwidth
				for( int i = 0; i < bw.length; i++ ) {
					
					// log space
					if( use_log ) {
						double phi = Math.log(bw[i]) - eta[0] * bwGrad[i];
						bw[i] = Math.exp(phi);
					} else {
						bw[i] -= eta[0] * bwGrad[i];
					}				
				}
						
		} else if( opt == Optimizer.Nesterov ) {
				
			for (int l = 0; l < weights.length; l++)
				for (int i = 0; i < weights[l].length; i++)
					for (int j = 0; j < weights[l][i].length; j++) {
						vWeights[l][i][j] = mWeights[l][i][j];
											
				        double grad_weights = errorGrad[l][i][j] + lambda * weights[l][i][j];    
				        mWeights[l][i][j] = mu * mWeights[l][i][j] - eta[l] * grad_weights;
				        weights[l][i][j] += -mu * vWeights[l][i][j] + (1 + mu) * mWeights[l][i][j];
					}
						
			for (int i = 0; i < bw.length; i++) {
				vBW[i] = mBW[i];
				double g = bwGrad[i]; 					
			    mBW[i] = mu * mBW[i] - fac*eta[0] * g;
			    
			    if( use_log) {
			        double phi = Math.log(bw[i]) -mu * vBW[i] + (1 + mu) * mBW[i];
				    bw[i] = Math.exp(phi);
			    } else {
			    	bw[i] += -mu * vBW[i] + (1 + mu) * mBW[i];
			    }
			}
			
		} else if( opt == Optimizer.Adam ) {
			double b1 = 0.9; //  exponential decay rate 
			double b2 = 0.999; // decay rate for the second moment estimates
			double eps = Math.pow(10, -8);

			double p1 = (1 - Math.pow(b1, t));
			double p2 = (1 - Math.pow(b2, t));

			// --- GEWICHTE UPDATE ---
		    for (int l = 0; l < weights.length; l++) {
		        for (int i = 0; i < weights[l].length; i++) {
		            for (int j = 0; j < weights[l][i].length; j++) {
		                double g = errorGrad[l][i][j];

		                // Momente aktualisieren
		                mWeights[l][i][j] = b1 * mWeights[l][i][j] + (1.0 - b1) * g;
		                vWeights[l][i][j] = b2 * vWeights[l][i][j] + (1.0 - b2) * (g * g);

		                // Bias-Korrektur
		                double mHat = mWeights[l][i][j] / p1;
		                double vHat = vWeights[l][i][j] / p2;

		                weights[l][i][j] -= eta[l] * mHat / (Math.sqrt(vHat) + eps);
		            }
		        }
		    }

		    // --- BANDBREITE (BW) UPDATE ---
		    for (int i = 0; i < bw.length; i++) {
		        double g = bwGrad[i];
		        
		        // Momente aktualisieren
		        mBW[i] = b1 * mBW[i] + (1.0 - b1) * g;
		        vBW[i] = b2 * vBW[i] + (1.0 - b2) * (g * g);

		        // Bias-Korrektur
		        double mHat = mBW[i] / p1;
		        double vHat = vBW[i] / p2;
		        		        
		        if( use_log) {
		        	double phi = Math.log(bw[i]) - fac*eta[0] * mHat / (Math.sqrt(vHat) + eps);
			        bw[i] = Math.exp(phi);
		        } else {
		        	bw[i] -= fac*eta[0] * mHat / (Math.sqrt(vHat) + eps);
		        }
		    }
			
		} else
			throw new RuntimeException(opt+" not supported yet!");
		
		for( int i = 0; i < bw.length; i++ ) {
			bw[i] = Math.min( bw[i], bw_max);
			bw[i] = Math.max( bw[i], bw_min);
		}
	}
		
	void train( List<double[]> x, List<double[]> y, List<double[]>  gwWeights) {
		List<double[][][]> l = new ArrayList<>();
		for( int i = 0; i < x.size(); i++ )
			l.add( getErrorGradient( x.get(i), y.get(i), gwWeights.get(i) ) );
		double[][][] errorGrad = calculateMeanOf3DList(l);		
		update(opt, errorGrad, eta);
	}
	
	void train_2d( List<double[]> x, List<double[]> y, List<double[][]>  gwWeights) {
		List<double[][][]> l = new ArrayList<>();
		for( int i = 0; i < x.size(); i++ ) 
			l.add( getErrorGradient(x.get(i), y.get(i), gwWeights.get(i) ) );	
		double[][][] errorGrad = calculateMeanOf3DList(l);
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