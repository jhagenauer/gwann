package supervised.nnet.gwann;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import supervised.nnet.NNet;
import supervised.nnet.activation.Constant;
import supervised.nnet.activation.Function;

public class GWANN extends NNet {
		
	public GWANN(Function[][] l, double[][][] weights, double[] eta, Optimizer m, double lambda ) {
		super(l,weights,eta,m,lambda); 
	}
		
	private double[][][] getErrorGradient( double[] x, double[] desired, double[] sw ) {
		int ll = layer.length - 1; // index of last layer
		double[][] error_signal = new double[layer.length][];
		double[][][] error_grad = new double[layer.length-1][][];		
		for( int l = ll; l > 0; l-- ) {
			error_signal[l-1] = new double[layer[l].length];
			error_grad[l-1] = new double[layer[l - 1].length][layer[l].length];
		}
		
		double[][] out = presentInt( x, weights )[0];											
		for (int l = ll; l > 0; l--) {	
			
			for (int i = 0; i < layer[l].length; i++) { 
				
				double s = 0;
				
				if( l == ll ) {
					s = (out[l][i] - desired[i]) * sw[i];
					error_signal[l-1][i] = layer[l][i].fDevFOut(out[l][i]) * s;
				} else {
					for (int j = 0; j < weights[l][i].length; j++)
						if( !( layer[l+1][j] instanceof Constant ) )
							s += error_signal[l][j] * weights[l][i][j];		
				}

				error_signal[l-1][i] = layer[l][i].fDevFOut(out[l][i]) * s;						
				for( int h = 0; h < layer[l-1].length; h++ ) 
					error_grad[l-1][h][i] += out[l-1][h] * error_signal[l-1][i];	
			
			}				
		}		
		
		return error_grad;
	}
		
	void train( List<double[]> x, List<double[]> y, List<double[]>  gwWeights) {
		List<double[][][]> l = new ArrayList<>();
		for( int i = 0; i < x.size(); i++ ) 
			l.add( getErrorGradient( x.get(i), y.get(i), gwWeights.get(i) ) );
		double[][][] errorGrad = calculateMeanOf3DList(l);		
		update(opt, errorGrad, eta);
		t++;
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