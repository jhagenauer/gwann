package supervised.nnet;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

import supervised.SupervisedNet;
import supervised.nnet.activation.Constant;
import supervised.nnet.activation.Function;

public class NNet implements SupervisedNet {

	public Function[][] layer;
	public double[][][] weights;
	protected double eta = 0.05;

	double[][][] v = null, v_prev = null;
	public static enum Optimizer { SGD, Momentum, Nesterov, RMSProp, Adam };
	protected Optimizer m;
		
	public NNet(Function[][] layer, double[][][] weights, double eta) {
		this(layer,weights,eta,Optimizer.Nesterov);
	}
	
	public NNet(Function[][] layer, double[][][] weights, double eta, Optimizer m ) {
		
		this.layer = layer;
		this.eta = eta;	
		this.weights = weights;
		this.m = m;
		
		this.v = new double[weights.length][][];
		this.v_prev = new double[weights.length][][];
		for( int i = 0; i < weights.length; i++ ) {
			this.v[i] = new double[weights[i].length][];
			this.v_prev[i] = new double[weights[i].length][];
			
			for( int j = 0; j < weights[i].length; j++ ) {
				this.v[i][j] = new double[weights[i][j].length];
				this.v_prev[i][j] = new double[weights[i][j].length];
			}
		}
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
		List<double[]> xl = new ArrayList<>();
		List<double[]> yl = new ArrayList<>();
		xl.add(x);
		yl.add(desired);
		train(xl,yl);
	}
	
	protected double[][][] getErrorGradient(List<double[]> x, List<double[]> y ) {
		int ll = layer.length - 1; // index of last layer
		double[][][] errorGrad = new double[layer.length-1][][];				
		double[][] delta = new double[layer.length][];		
		for( int e = 0; e < x.size(); e++ ) {
			double[] desired = y.get(e);
			double[][] out = presentInt( x.get(e) )[0];
											
			for (int l = ll; l > 0; l--) {	
				
				delta[l] = new double[layer[l].length];
				
				if( errorGrad[l-1] == null )
					errorGrad[l-1] = new double[layer[l-1].length][layer[l].length];
					
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
										
					for( int h = 0; h < layer[l-1].length; h++ ) 
						errorGrad[l-1][h][i] += out[l-1][h] * delta[l][i];
				}
			}
		}
		return errorGrad;
	}
	
	protected void updateSGD( double[][][] errorGrad, double leta ) {
		for (int l = 0; l < weights.length; l++) 
			for (int i = 0; i < weights[l].length; i++) 												
				for (int j = 0; j < weights[l][i].length; j++) 
					weights[l][i][j] -= leta * errorGrad[l][i][j];				
	}
	
	protected void updateMomentum( double[][][] errorGrad, double leta ) {
		double mu = 0.9;
		for (int l = 0; l < weights.length; l++) 
			for (int i = 0; i < weights[l].length; i++) 												
				for (int j = 0; j < weights[l][i].length; j++) {
					v[l][i][j] = mu * v[l][i][j] - leta * errorGrad[l][i][j]; // momentum
					weights[l][i][j] += v[l][i][j];
				}
	}
	
	protected void updateNestrov( double[][][] errorGrad, double leta ) {		
		double mu = 0.9;
		for (int l = 0; l < weights.length; l++) 
			for (int i = 0; i < weights[l].length; i++) 												
				for (int j = 0; j < weights[l][i].length; j++) {
					v_prev[l][i][j] = v[l][i][j];
					v[l][i][j] = mu * v[l][i][j] - leta * errorGrad[l][i][j];
					weights[l][i][j] += -mu * v_prev[l][i][j] + v[l][i][j] + mu*v[l][i][j]; 
				}
	} 
	
	protected void updateRMSProp( double[][][] errorGrad, double leta ) {
		double gamma = 0.9;
		for (int l = 0; l < weights.length; l++) 
			for (int i = 0; i < weights[l].length; i++) 												
				for (int j = 0; j < weights[l][i].length; j++) {
					v[l][i][j] = gamma*v[l][i][j]+(1-gamma)*errorGrad[l][i][j]*errorGrad[l][i][j];
					weights[l][i][j] -= leta/Math.sqrt(v[l][i][j])*errorGrad[l][i][j];
				}
		v_prev = errorGrad;
	}
	
	protected void updateAdam( double[][][] errorGrad, double leta) {
		double b1 = 0.9, b2 = 0.999, eps = Math.pow(10, -8);
		
		for (int l = 0; l < weights.length; l++) 
			for (int i = 0; i < weights[l].length; i++) 												
				for (int j = 0; j < weights[l][i].length; j++) {
					
					v_prev[l][i][j] = b1*v_prev[l][i][j]+(1-b1)*errorGrad[l][i][j];
					v[l][i][j] = b2*v[l][i][j]+(1-b2)*Math.pow(errorGrad[l][i][j], 2);
					
					double mt = v_prev[l][i][j]/(1-b1*b1);
					double vt = v[l][i][j]/(1-b2*b2);
					
					weights[l][i][j] -= leta*mt/(Math.sqrt(vt)+eps);
				}		
	}

	protected int t = 1;
	public void train( List<double[]> x, List<double[]> y ) {
		double[][][] errorGrad = getErrorGradient(x,y);
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
		else if( m == Optimizer.Adam )
			updateAdam(errorGrad, leta);
		t++;
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
	
	public double getEta() {
		return eta;
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
