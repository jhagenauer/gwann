package supervised.nnet.gwann;

import java.util.List;

import supervised.nnet.NNet;
import supervised.nnet.activation.Constant;
import supervised.nnet.activation.Function;

public class GWANN extends NNet {
	
	public GWANN(Function[][] l, double[][][] weights, double eta, Optimizer m ) {
		super(l,weights,eta,m); 
	}

	public GWANN(Function[][] l, double[][][] weights, double eta ) {
		super(l,weights,eta); 
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
		double leta = eta/x.size();
		
		//leta = eta * Math.pow( 0.5, Math.floor( (double)t/iv));
		
		if( m == Optimizer.SGD )
			updateSGD(errorGrad, leta);
		else if( m == Optimizer.Momentum )
			updateMomentum(errorGrad, leta);
		else if( m == Optimizer.Nesterov )
			updateNestrov(errorGrad, leta);
		else if( m == Optimizer.RMSProp )
			updateRMSProp(errorGrad, leta);
		else if( m == Optimizer.Adam)
			updateAdam(errorGrad, leta);
		t++;
	}
}
