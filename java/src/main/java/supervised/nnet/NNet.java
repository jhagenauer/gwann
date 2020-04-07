package supervised.nnet;

import java.util.Arrays;
import java.util.List;
import java.util.Random;

import org.apache.log4j.Logger;

import supervised.SupervisedNet;
import supervised.nnet.activation.Constant;
import supervised.nnet.activation.Function;
import utils.DataUtils;

public class NNet implements SupervisedNet {

	private static Logger log = Logger.getLogger(NNet.class);

	public Function[][] layer;
	public double[][][] weights;
	protected double eta = 0.05;
	
	public NNet(Function[][] layer, double[][][] weights, double eta) {
		this.layer = layer;
		this.eta = eta;	
		this.weights = weights;
	}
			
	@Override
	public double[] present(double[] x) {
		double[][] out = presentInt(x)[0];
		return out[out.length-1]; // output of last layer
	}
		
	public double[][][] presentInt(double[] x) {
		double[][] out = new double[layer.length][];
		double[][] net = new double[layer.length][];
		
		double[] in = x;
		for (int l = 0;; l++) {

			net[l] = in;
			out[l] = new double[layer[l].length];
			// apply activation to net input
			
			int inIdx = 0;
			for (int i = 0; i < layer[l].length; i++ )
				if( l == 0 && layer[l][i] instanceof Constant )
					out[l][i] = layer[l][i].f(Double.NaN);
				else
					out[l][i] = layer[l][i].f(in[inIdx++]);
							
			if (l == layer.length - 1)
				return new double[][][]{out,net};
						
			// apply weights, calculate new net input
			in = new double[weights[l][0].length]; // number of neurons in l+1 connected to neuron 0 in l
			for (int i = 0; i < weights[l].length; i++) 
				for (int j = 0; j < weights[l][i].length; j++) 
					in[j] += weights[l][i][j] *	out[l][i];		
		}
	}

	@Override
	public void train(double t, double[] x, double[] desired) {
		double[][][] p = presentInt(x);
		
		double[][] out = p[0];
		//double[][] net = p[1];
		
		// back propagation
		double[][] delta = new double[layer.length][];
		int ll = layer.length - 1; // index of last layer

		for (int l = ll; l > 0; l--) {
			
			delta[l] = new double[layer[l].length];
			for (int i = 0; i < layer[l].length; i++) { // for each neuron of layer l
				
				double s = 0;
				if( l == ll ) {
					s = (out[l][i] - desired[i]);
				} else {
					for (int j = 0; j < weights[l][i].length; j++)
						if( !( layer[l+1][j] instanceof Constant ) )
							s += delta[l + 1][j] * weights[l][i][j];
				}		
				delta[l][i] = layer[l][i].fDevFOut(out[l][i]) * s;
			}	
		}
		
		// https://stats.stackexchange.com/questions/29130/difference-between-neural-net-weight-decay-and-learning-rate
		// change weights to layer i
		for (int l = 0; l < ll; l++) 
			for (int i = 0; i < weights[l].length; i++) // neurons of layer l
				for (int j = 0; j < weights[l][i].length; j++) // weights to neurons j in layer l+1
					weights[l][i][j] -= eta * delta[l+1][j] * out[l][i];
	}
	
	public void train( List<double[]> batch, int[] fa, int[] ta ) {
		
		int ll = layer.length - 1; // index of last layer
		double[][] delta = new double[layer.length][];
		double[][][] update = new double[layer.length-1][][];		
		
		for( double[] x : batch ) {
			double[] desired = DataUtils.strip(x, ta);
			double[][] out = presentInt( DataUtils.strip(x, fa))[0];
												
			for (int l = ll; l > 0; l--) {	
				
				delta[l] = new double[layer[l].length];
				
				if( update[l-1] == null )
					update[l-1] = new double[layer[l-1].length][layer[l].length];
					
				for (int i = 0; i < layer[l].length; i++) { // for each neuron of layer l
					
					double s = 0;
					if( l == ll ) {
						s = (out[l][i] - desired[i]);		
					} else {
						for (int j = 0; j < weights[l][i].length; j++)
							if( !( layer[l+1][j] instanceof Constant ) )
								s += delta[l + 1][j] * weights[l][i][j]; // ?
					}
																
					delta[l][i] = layer[l][i].fDevFOut(out[l][i]) * s;				
					// delta[l][i] = layer[l][i].fDev(net[l][i]) * s; 
										
					for( int h = 0; h < layer[l-1].length; h++ ) 
						update[l-1][h][i] += out[l-1][h] * delta[l][i];
				}
			}
		}
						
		// change weights to layer i
		for (int l = 0; l < ll; l++) 
			for (int i = 0; i < weights[l].length; i++) 												
				for (int j = 0; j < weights[l][i].length; j++) 
					weights[l][i][j] -= eta * update[l][i][j]/batch.size();
	}
		
	@Override
	public double[] getResponse(double[] x, double[] weight) {
		return null;
	}
	
	public static enum initMode { gorot_unif, norm05 };
	public static double[][][] getFullyConnectedWeights(Function[][] layer, initMode im, int seed ) {
		Random r = new Random(seed);
		
		double[][][] weights = new double[layer.length-1][][];
		for( int l = 0; l < weights.length; l++ ) { // i, over layer
			int fanIn = layer[l].length;
			int fanOut = layer[l+1].length;
			
			weights[l] = new double[layer[l].length][];
			for( int i = 0; i < weights[l].length; i++ ) {
				
				weights[l][i] = new double[layer[l+1].length];
				for( int j = 0; j < weights[l][i].length; j++ )
					if( layer[l+1][j] instanceof Constant )
						weights[l][i][j] = Double.NaN;	
					else {						
						if( im == initMode.norm05 )
							weights[l][i][j] = r.nextGaussian()*0.5;
						else {
							// https://keras.rstudio.com/reference/initializer_glorot_uniform.html
							double limit = Math.sqrt( 6.0/(fanIn+fanOut) );
							weights[l][i][j] = r.nextDouble()*limit*2-limit;							
						}					
					}
			}			
		}
		return weights;
	}
	
	public void printNeurons() {
		System.out.println("--- Neurons ---");
		for( int l = 0; l < layer.length; l++ ) {
			System.out.println("Layer "+l);
			for( int i = 0; i < layer[l].length; i++ )
				System.out.println("l"+l+"n"+i+(layer[l][i] instanceof Constant ? "b" : "")+", "+layer[l][i].getClass());
		}	
	}
	
	public void printWeights() {		
		System.out.println("--- Weights ---");
		for( int l = 0; l < weights.length; l++ ) {
			System.out.println("Layer "+l);
			for( int i = 0; i < weights[l].length; i++ ) 
				for( int j = 0; j < weights[l][i].length; j++) 
					if( !( layer[l+1][j] instanceof Constant ) ) // skip NA-connections to constant neurons
					System.out.println("l"+l+"n"+i+(layer[l][i] instanceof Constant ? "b" : "")+" --> l"+(l+1)+"n"+j+(layer[l+1][j] instanceof Constant ? "b" : "")+": "+weights[l][i][j]);
		}			
	}
	
	public void setEta(double eta) {
		this.eta = eta;
	}
	
	public int getNumParameters() {
		int sum = 0;
		for( int i = 0; i < weights.length; i++ )
			for( int j = 0; j < weights[i].length; j++ )
				sum += weights[i][j].length;
		return sum;
	}
	
	public double[][][] getCopyOfWeights() {
		double[][][] w = new double[weights.length][][];
		for( int l = 0; l < weights.length; l++ ) {
			w[l] = new double[weights[l].length][];
			for( int i = 0; i < weights[l].length; i++ )
				w[l][i] = Arrays.copyOf(weights[l][i], weights[l][i].length);
		}		
		return w;
	}
	
	public void setWeights( double[][][] w ) {
		this.weights = w;
	}
}
