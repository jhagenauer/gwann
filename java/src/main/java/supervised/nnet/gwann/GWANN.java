package supervised.nnet.gwann;

import java.util.List;
import supervised.nnet.NNet;
import supervised.nnet.activation.Constant;
import supervised.nnet.activation.Function;

public class GWANN extends NNet {
	
	public GWANN(Function[][] l, double[][][] weights, double eta, Optimizer m ) {
		super(l,weights,eta,m); 
	}
	
	public GWANN(Function[][] l, double[][][] weights, double[] eta, Optimizer m ) {
		super(l,weights,eta,m); 
	}
		
	private double[][][] getErrorGradient( List<double[]> x, List<double[]> y, List<double[]> sampleWeights ) {
		int ll = layer.length - 1; // index of last layer
		double[][] delta = new double[layer.length][];
		double[][][] errorGrad = new double[layer.length-1][][];
		
		for( int e = 0; e < x.size(); e++ ) {
			double[] desired = y.get(e);
			double[][] out = presentInt( x.get(e) )[0];
			double[] sw = sampleWeights.get( e );
												
			for (int l = ll; l > 0; l--) {	
				
				delta[l] = new double[layer[l].length];
				if( errorGrad[l-1] == null )
					errorGrad[l-1] = new double[layer[l-1].length][layer[l].length];
								
				if( l == ll ) {
					for (int i = 0; i < layer[l].length; i++) { 
																	
						double s = (out[l][i] - desired[i]) * sw[i];
						delta[l][i] = layer[l][i].fDevFOut(out[l][i]) * s;
						
						for( int h = 0; h < layer[l-1].length; h++ ) 
							errorGrad[l-1][h][i] += out[l-1][h] * delta[l][i];	
					}
				} else {
					for (int i = 0; i < layer[l].length; i++) {
						
						double s = 0;	
						for (int j = 0; j < weights[l][i].length; j++)
							if( !( layer[l+1][j] instanceof Constant ) )
								s += delta[l + 1][j] * weights[l][i][j];
																																						
						delta[l][i] = layer[l][i].fDevFOut(out[l][i]) * s;	
						
						for( int h = 0; h < layer[l-1].length; h++ ) 
							errorGrad[l-1][h][i] += out[l-1][h] * delta[l][i];	
					}
				}
			}
		}
		return errorGrad;
	}
		
	public void train( List<double[]> x, List<double[]> y, List<double[]>  gwWeights) {
		double[][][] errorGrad = getErrorGradient(x,y,gwWeights);
		double[] leta = new double[eta.length];
		for( int i = 0; i < leta.length; i++ )
			leta[i] = eta[i]/x.size();
				
		update(m,errorGrad, leta,lambda);
		t++;
	}
	
	public double[] weightsForLocation(int i ) {
		double[][] w = weights[weights.length-1];
		double[] d = new double[w.length];
		for( int j = 0; j < d.length; j++ )
			d[j] = w[j][i];
		return d;
	}
}