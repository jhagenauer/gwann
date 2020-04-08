package supervised.nnet.gwann;

import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.jblas.DoubleMatrix;

import supervised.nnet.NNet;
import supervised.nnet.activation.Constant;
import supervised.nnet.activation.Function;
import utils.DataUtils;
import utils.GWRUtils;
import utils.GWRUtils.GWKernel;

public class GWANN extends NNet {

	public GWANN(Function[][] l, double[][][] weights, double eta ) {
		super(l,weights,eta);
	}
		
	public void trainSimple( List<double[]> batch, Map<double[],double[]> sampleWeights, int[] fa, int[] ta ) {
		
		int ll = layer.length - 1; // index of last layer
		double[][] delta = new double[layer.length][];
		double[][][] update = new double[layer.length-1][][];
		
		for( double[] x : batch ) {
			double[] desired = DataUtils.strip(x, ta);
			double[][] out = presentInt( DataUtils.strip(x, fa) )[0];
			//double[][] net = presentInt( DataUtils.strip(x, fa) )[1];
			double[] sw = sampleWeights.get(x);
			
			assert ta.length == sw.length : ta.length+"!="+sw.length;
												
			for (int l = ll; l > 0; l--) {	
				
				delta[l] = new double[layer[l].length];
				if( update[l-1] == null )
					update[l-1] = new double[layer[l-1].length][layer[l].length];
								
				if( l == ll ) {
					for (int i = 0; i < layer[l].length; i++) { 
																	
						double s = (out[l][i] - desired[i]) * sw[i]/batch.size();
						
						delta[l][i] = layer[l][i].fDevFOut(out[l][i]) * s;
						//delta[l][i] = layer[l][i].fDev(net[l][i]) * s; 
						for( int h = 0; h < layer[l-1].length; h++ ) 
							update[l-1][h][i] += out[l-1][h] * delta[l][i];	
					}
				} else {
					for (int i = 0; i < layer[l].length; i++) { 
						
						double s = 0;	
						for (int j = 0; j < weights[l][i].length; j++)
							if( !( layer[l+1][j] instanceof Constant ) )
								s += delta[l + 1][j] * weights[l][i][j];
																																						
						delta[l][i] = layer[l][i].fDevFOut(out[l][i]) * s;	
						//delta[l][i] = layer[l][i].fDev(net[l][i]) * s; 
						for( int h = 0; h < layer[l-1].length; h++ ) 
							update[l-1][h][i] += out[l-1][h] * delta[l][i];	
					}
				}
			}
		}
		
		// change weights to layer i
		for (int l = 0; l < ll; l++) 
			for (int i = 0; i < weights[l].length; i++) 												
				for (int j = 0; j < weights[l][i].length; j++) 
					weights[l][i][j] -= eta * update[l][i][j];
	}
			
	public double[] present(double[] x, double[][][] sampleWeights) {
		double[][] out = presentInt(x, sampleWeights);
		return out[out.length - 1]; 
	}

	public double[][] presentInt(double[] x, double[][][] sampleWeights) {
		double[][] out = new double[layer.length][];
		double[] in = x;
		
		for (int l = 0;; l++) {

			out[l] = new double[layer[l].length];
			
			int inIdx = 0;
			for (int i = 0; i < layer[l].length; i++ )
				if( l == 0 && layer[l][i] instanceof Constant )
					out[l][i] = layer[l][i].f(Double.NaN);
				else
					out[l][i] = layer[l][i].f(in[inIdx++]);

			if (l == layer.length - 1)
				return out;

			in = new double[weights[l][0].length]; // number of non-constant neurons in l+1
			for (int i = 0; i < weights[l].length; i++)
				for (int j = 0; j < weights[l][i].length; j++)
					in[j] += weights[l][i][j] *	out[l][i] *	sampleWeights[l][i][j];
		}
	}	
	
	public static int[] toIntArray(Collection<Integer> c) {
		int[] j = new int[c.size()];
		int i = 0;
		for (int l : c)
			j[i++] = l;
		return j;
	}

	public static Map<double[], double[]> getSampleWeights(List<double[]> samplesTrain, DoubleMatrix W2, GWKernel kernel, double bw) {
		Map<double[], double[]> sampleWeights = new HashMap<>();
		for (int i = 0; i < W2.rows; i++) {
			double[] w = new double[W2.columns];
			for (int j = 0; j < w.length; j++)
				w[j] = GWRUtils.getKernelValue(kernel, W2.get(i,j), bw);
			sampleWeights.put(samplesTrain.get(i), w);
		}
		return sampleWeights;
	}
		
	// adaptive
	public static Map<double[], double[]> getSampleWeights(List<double[]> samplesTrain, DoubleMatrix W, DoubleMatrix W2, GWKernel kernel, int nb) {
		assert W.rows == W2.rows;
		
		Map<double[], double[]> sampleWeights = new HashMap<>();
		for (int i = 0; i < W.rows; i++) {
			double bw = W.getRow(i).sort().get(nb);
			double[] w = new double[W2.columns];
			for (int j = 0; j < w.length; j++)
				w[j] = GWRUtils.getKernelValue(kernel, W2.get(i, j), bw);
			sampleWeights.put(samplesTrain.get(i), w);
		}
		return sampleWeights;
	}
}
