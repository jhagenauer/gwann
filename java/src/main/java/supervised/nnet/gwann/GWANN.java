package supervised.nnet.gwann;

import java.util.List;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.jblas.DoubleMatrix;

import supervised.nnet.NNet;
import supervised.nnet.activation.Function;
import utils.GWUtils.GWKernel;

public class GWANN extends NNet {
	
	private static Logger log = LogManager.getLogger(GWANN.class);

	public static boolean f = true;
		
	public GWANN(Function[][] l, double[][][] weights, boolean[][] n_act, double[] eta, Optimizer m, double lambda ) {
		super(l,weights, n_act, eta, m, lambda); 
	}
	
	double bw_min, bw_max;
	
	double[][][] vWeights = null, mWeights = null;
	double[] vBW = null, mBW = null;
	
	DoubleMatrix W;
	GWKernel kernel;
	public double bw[];
		
	public GWANN(Function[][] ll, double[][][] weights, boolean[][] n_act, double[] eta, Optimizer m, double lambda, DoubleMatrix w, GWKernel kernel, double bw_min, double bw_max ) {
		super(ll,weights,n_act,eta,m,lambda);
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
	
	double[][][] getErrorGradient_orig( List<double[]> lx, List<double[]> ly, List<double[]> lsw ) {
		int ll = layer.length - 1; // index of last layer (neuron)
		
		double[][] delta = new double[layer.length][];
		double[][][] error_grad = new double[layer.length-1][][];		
		for( int l = ll; l > 0; l-- ) {
			delta[l-1] = new double[layer[l].length];
			error_grad[l-1] = new double[layer[l - 1].length][layer[l].length];
		}
		
		for( int k = 0; k < lx.size(); k++ ) {	
			double[] x = lx.get(k);
			double[] y = ly.get(k);
			double[] sw = lsw.get(k);
			
			forwardInto(x);
			double[][] out = outBuf;
			
			for (int l = ll; l > 0; l--) {	
				
				for (int i = 0; i < layer[l].length; i++) { 	
					
					double error_signal = 0;				
					if( l == ll ) {
						error_signal = (out[l][i] - y[i]) * sw[i];
					} else {
						for (int j = 0; j < weights[l][i].length; j++)
							if( !n_act[l+1][j] ) 
								error_signal += delta[l][j] * weights[l][i][j];		
					}
					delta[l-1][i] = layer[l][i].fDevFOut(out[l][i]) * error_signal;
					
					for( int h = 0; h < layer[l-1].length; h++ ) 
						error_grad[l-1][h][i] += out[l-1][h] * delta[l-1][i]/lx.size();				
				}				
			}
		}
		return error_grad;
	}
				
	double[][][] getErrorGradient( List<double[]> lx, List<double[]> ly, List<double[][]> lsw ) {
		int ll = layer.length - 1; // index of last layer
		double[][] delta = new double[ll][];
		double[][][] error_grad = new double[ll][][];		
		for( int l = ll; l > 0; l-- ) {
			delta[l-1] = new double[layer[l].length];
			error_grad[l-1] = new double[layer[l - 1].length][layer[l].length];
		}
		
		for( int k = 0; k < lx.size(); k++ ) {
			double[] x = lx.get(k);
			double[] y = ly.get(k);
			double[][] sw = lsw.get(k);
			
			forwardInto(x);
			double[][] out = outBuf;	
		
			for (int l = ll; l > 0; l--) {
				for (int i = 0; i < layer[l].length; i++) {
	
					double error_signal = 0.0;
					if (l == ll) {
						error_signal = (out[l][i] - y[i]);
					} else {
						for (int j = 0; j < weights[l][i].length; j++) {
							if( !n_act[l+1][j] ) {
								double effective_weight = weights[l][i][j];
								if (l == ll - 1) 
									effective_weight *= sw[i][j];							
								error_signal += delta[l][j] * effective_weight;
							}
						}
					}
					delta[l - 1][i] = layer[l][i].fDevFOut(out[l][i]) * error_signal;
					for (int h = 0; h < layer[l - 1].length; h++) {
						double spatial_modifier = (l == ll) ? sw[h][i] : 1.0;					
						error_grad[l-1][h][i] += out[l-1][h] * delta[l-1][i] * spatial_modifier/lx.size();
					}
				}
			}
		}
		return error_grad;
	}
		
	public static boolean norm_grad = false;				
	int kk = 1;
		
	@Deprecated
	void train_( List<double[]> x, List<double[]> y, List<double[][]>  gwWeights, int feature, double[] eta ) {
		throw new RuntimeException();
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