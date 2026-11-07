package supervised.nnet;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import supervised.SupervisedNet;
import supervised.nnet.activation.Constant;
import supervised.nnet.activation.Function;

public class NNet implements SupervisedNet {
	
	private static Logger log = LogManager.getLogger(NNet.class);
	
	public Function[][] layer;
	public double[][][] weights;
	public boolean[][] n_act;

	double[][][] v = null, v_prev = null;

	public double lambda = Double.NaN;
	
	public static enum Optimizer {
		SGD, Momentum, NAG, RMSProp, AdamW, Adam, LM, LM_ANA, LBFGS, Nesterov_test, Adam_test
	};
	
	protected Optimizer opt;

	public NNet(Function[][] layer, double[][][] weights, boolean[][] n_act, double[] eta_0, Optimizer opt, double lambda) {

		this.lambda = lambda;
		this.layer = layer;
		this.weights = weights;
		this.n_act = n_act;
		this.opt = opt;

		this.v = new double[weights.length][][];
		this.v_prev = new double[weights.length][][];
		for (int i = 0; i < weights.length; i++) {
			this.v[i] = new double[weights[i].length][];
			this.v_prev[i] = new double[weights[i].length][];

			for (int j = 0; j < weights[i].length; j++) {
				this.v[i][j] = new double[weights[i][j].length];
				this.v_prev[i][j] = new double[weights[i][j].length];
			}
		}
	}
	
	public void resetVs() {
		this.t = 1;
		this.v = new double[weights.length][][];
		this.v_prev = new double[weights.length][][];
		for (int i = 0; i < weights.length; i++) {
			this.v[i] = new double[weights[i].length][];
			this.v_prev[i] = new double[weights[i].length][];

			for (int j = 0; j < weights[i].length; j++) {
				this.v[i][j] = new double[weights[i][j].length];
				this.v_prev[i][j] = new double[weights[i][j].length];
			}
		}
	}

	@Override
	public double[] present(double[] x) {		
		double[][] out = presentInt(x)[0];
		return out[out.length - 1]; // output of last layer
	}	

	public double[][][] presentInt(double[] x) {
	    double[][] out = new double[layer.length][];
	    double[][] net = new double[layer.length][];

	    double[] in = x;
	    int ll_0 = layer.length;

	    for (int l = 0; l < ll_0; l++) {
	        net[l] = in;
	        int ll_l = layer[l].length;
	        out[l] = new double[ll_l]; 

	        // Apply activation to net input
	        if (l == 0) 
	            for (int i = 0; i < ll_l; i++) 
	            	out[l][i] = n_act[l][i] ? layer[l][i].f(Double.NaN) : layer[l][i].f(in[i]);          
	        else 
	            for (int i = 0; i < ll_l; i++) 
	                out[l][i] = layer[l][i].f(in[i]);
	        
	        if (l == ll_0 - 1) 
	            return new double[][][] { out, net }; 
	        
	        // Compute new net input for the next layer
	        int ll_lp1 = layer[l + 1].length;
	        double[] nextIn = new double[ll_lp1];

	        for (int j = 0; j < ll_lp1; j++) 
	            for (int i = 0; i < ll_l; i++) 
	                nextIn[j] += weights[l][i][j] * out[l][i];	            
	        
	        in = nextIn; // Move to the next layer
	    }
	    throw new IllegalStateException("Unexpected execution flow in presentInt");
	}
	
	protected double[][] outBuf, netBuf;   // training-only scratch, one set per NNet instance
	protected void forwardInto(double[] x) {
	    if (outBuf == null) {
	        outBuf = new double[layer.length][]; 
	        netBuf = new double[layer.length][];
	        for (int l = 0; l < layer.length; l++) { 
	        	outBuf[l] = new double[layer[l].length]; 
	        	netBuf[l] = new double[layer[l].length]; 
	        }
	    }
	    for (int i = 0; i < layer[0].length; i++)
	        outBuf[0][i] = n_act[0][i] ? layer[0][i].f(Double.NaN) : layer[0][i].f(x[i]);
	    for (int l = 0; l < layer.length - 1; l++) {
	        double[] nx = netBuf[l + 1];
	        java.util.Arrays.fill(nx, 0.0);
	        for (int i = 0; i < layer[l].length; i++) {
	            double o = outBuf[l][i];
	            double[] w = weights[l][i];
	            for (int j = 0; j < nx.length; j++) nx[j] += w[j] * o;
	        }
	        for (int j = 0; j < nx.length; j++) outBuf[l + 1][j] = layer[l + 1][j].f(nx[j]);
	    }
	}
	
	@Override
	public void train(double t, double[] x, double[] desired) {		
		throw new RuntimeException("Dont use!");
	}
	
	protected double[][][] getErrorGradient( List<double[]> lx, List<double[]> ly ) {
		int ll = layer.length - 1; // index of last layer
		
		double[][][] error_grad = new double[ll][][];
		double[][] delta = new double[ll][];		
		for( int l = ll; l > 0; l-- ) {
			delta[l-1] = new double[layer[l].length];
			error_grad[l-1] = new double[layer[l - 1].length][layer[l].length];
		}
		
		for( int k = 0; k < lx.size(); k++ ) {
			double[] x = lx.get(k);
			double[] y = ly.get(k);
		
			forwardInto(x);
			double[][] out = outBuf;
			
			for (int l = ll; l > 0; l--) {
					
				for (int i = 0; i < layer[l].length; i++) { // for each neuron i of layer l
			
					double error_signal = 0;
					if( l == ll )
						error_signal = out[l][i] - y[i];
					else {
						for (int j = 0; j < weights[l][i].length; j++)
							if ( !n_act[l + 1][j] ) // ignore useless weight to constant neurons
								error_signal += delta[l][j] * weights[l][i][j];
					}			
					delta[l-1][i] = layer[l][i].fDevFOut(out[l][i]) * error_signal;
					
					for (int h = 0; h < layer[l - 1].length; h++)
						error_grad[l - 1][h][i] += out[l - 1][h] * delta[l-1][i]/lx.size();
				}
			}			
		}
		return error_grad;
	}
	
	public static double mu = 0.90;
	public static double fac = 1.5;
	public static boolean center = false;

	public static double decay_rate = 0.0; 
			
	public void update( Optimizer opt, double[][][] errorGrad, double[] eta ) {
								
		if( opt == Optimizer.SGD ) {
			for (int l = 0; l < weights.length; l++)
				for (int i = 0; i < weights[l].length; i++)
					for (int j = 0; j < weights[l][i].length; j++) {
						if( lambda > 0 )
							weights[l][i][j] -= eta[l] * (errorGrad[l][i][j] + lambda * weights[l][i][j]);
						else
							weights[l][i][j] -= eta[l] * errorGrad[l][i][j]; 
					}
						
		} else if( opt == Optimizer.Momentum ) {
			//double mu = 0.95;
			for (int l = 0; l < weights.length; l++)
				for (int i = 0; i < weights[l].length; i++)
					for (int j = 0; j < weights[l][i].length; j++) {
						v[l][i][j] = mu * v[l][i][j] - eta[l] * errorGrad[l][i][j]; 
						
						if( lambda > 0 )
							weights[l][i][j] += v[l][i][j] - lambda * weights[l][i][j];
						else
							weights[l][i][j] += v[l][i][j];
						
					}
		} else if( opt == Optimizer.NAG ) {
			//double mu = 0.2;
			for (int l = 0; l < weights.length; l++)
				for (int i = 0; i < weights[l].length; i++)
					for (int j = 0; j < weights[l][i].length; j++) {
						v_prev[l][i][j] = v[l][i][j];
											
				        double grad = errorGrad[l][i][j] + lambda * weights[l][i][j];    
				        v[l][i][j] = mu * v[l][i][j] - eta[l] * grad;
				        weights[l][i][j] += -mu * v_prev[l][i][j] + (1 + mu) * v[l][i][j];
					}
		} else if( opt == Optimizer.RMSProp ) {
			double eps = 1e-8;
			double gamma = 0.9;
			for (int l = 0; l < weights.length; l++)
				for (int i = 0; i < weights[l].length; i++)
					for (int j = 0; j < weights[l][i].length; j++) {
						v[l][i][j] = gamma * v[l][i][j] + (1 - gamma) * errorGrad[l][i][j] * errorGrad[l][i][j];
						weights[l][i][j] -= eta[l] * errorGrad[l][i][j] / Math.sqrt(v[l][i][j] + eps);
					}
		} else if( opt == Optimizer.AdamW ) {
			double b1 = 0.9;       // decay rate for first moment
			double b2 = 0.999;     // decay rate for second moment
			double eps = 1e-8;     // small constant for numerical stability
			double decay = lambda; // weight decay (lambda)
			
			for (int l = 0; l < weights.length; l++) {
			    for (int i = 0; i < weights[l].length; i++) {
			        for (int j = 0; j < weights[l][i].length; j++) {
			            double w = weights[l][i][j];
			            double g = errorGrad[l][i][j];

			            // --- AdamW update ---
			            // Update biased first moment estimate
			            v_prev[l][i][j] = b1 * v_prev[l][i][j] + (1 - b1) * g;
			            // Update biased second raw moment estimate
			            v[l][i][j] = b2 * v[l][i][j] + (1 - b2) * g * g;

			            // Compute bias-corrected first and second moments
			            double m_hat = v_prev[l][i][j] / (1 - Math.pow(b1, t));
			            double v_hat = v[l][i][j] / (1 - Math.pow(b2, t));

			            // Apply weight decay (decoupled from gradient)
			            w -= eta[l] * decay * w;

			            // Apply Adam step
			            w -= eta[l] * m_hat / (Math.sqrt(v_hat) + eps);

			            // Store updated weight
			            weights[l][i][j] = w;
			        }
			    }
			}

		} else if( opt == Optimizer.Adam ) {
			double b1 = 0.9; //  exponential decay rate 
			double b2 = 0.999; // decay rate for the second moment estimates
			double eps = 1e-4;

			double p1 = (1 - Math.pow(b1, t));
			double p2 = (1 - Math.pow(b2, t));

			for (int l = 0; l < weights.length; l++)
				for (int i = 0; i < weights[l].length; i++)
					for (int j = 0; j < weights[l][i].length; j++) {

						double g = errorGrad[l][i][j];
						v_prev[l][i][j] = b1 * v_prev[l][i][j] + (1 - b1) * g; // mt
						v[l][i][j] = b2 * v[l][i][j] + (1 - b2) * g * g; // vt

						// https://www.ruder.io/optimizing-gradient-descent/
						double mt_hat = v_prev[l][i][j] / p1;
						double vt_hat = v[l][i][j] / p2;
						weights[l][i][j] -= eta[l] * mt_hat / (Math.sqrt(vt_hat) + eps);
						
						if( lambda > 0)
							weights[l][i][j] -= eta[l] * lambda * weights[l][i][j];
					}
		}
		
		t++;
	}

	protected int t = 1;
	private double error_prev = Double.MAX_VALUE;
	
	public static double[] toDouble1D( double[][][] w) {				
		List<Double> flat = new ArrayList<>();
		for (double[][] mat : w)
			for (double[] row : mat)
				for (double val : row)
					flat.add(val);
		return flat.stream().mapToDouble(Double::doubleValue).toArray();			
	}
	
	public void setWeights1D( double[] w) {
		int k = 0;
		for( int a = 0; a < weights.length; a++ )
			for( int b = 0; b < weights[a].length; b++ )
				for( int c = 0; c < weights[a][b].length; c++ ) 
					weights[a][b][c] = w[k++];
	}
	
	public void train(List<double[]> x, List<double[]> y, double[] eta ) {
		double[][][] errorGrad = getErrorGradient( x, y);				
		update( opt, errorGrad, eta );
	}
	
	@Override
	public double[] getResponse(double[] x, double[] weight) {
		return null;
	}

	public void printNeurons() {
		System.out.println("--- Neurons ---");
		for (int l = 0; l < layer.length; l++) {
			System.out.println("Layer " + l);
			for (int i = 0; i < layer[l].length; i++)
				System.out.println("l" + l + "n" + i + (layer[l][i] instanceof Constant ? "b" : "") + ", " + layer[l][i].getClass());
		}
	}

	public void printWeights() {
		System.out.println("--- Weights ---");
		for (int l = 0; l < weights.length; l++) {
			System.out.println("Layer " + l);
			for (int i = 0; i < weights[l].length; i++)
				for (int j = 0; j < weights[l][i].length; j++)
					//if (!(layer[l + 1][j] instanceof Constant)) // skip NA-connections to constant neurons
						System.out.println("l" + l + "n" + i + (layer[l][i] instanceof Constant ? "b" : "") + " --> l" + (l + 1) + "n" + j + (layer[l + 1][j] instanceof Constant ? "b" : "") + ": " + weights[l][i][j]);
		}
	}

	public void printWeights(int l) {
		System.out.println("Layer " + l);
		for (int i = 0; i < weights[l].length; i++)
			for (int j = 0; j < weights[l][i].length; j++)
				if (!(layer[l + 1][j] instanceof Constant)) // skip NA-connections to constant neurons
					System.out.println("l" + l + "n" + i + (layer[l][i] instanceof Constant ? "b" : "") + " --> l" + (l + 1) + "n" + j + (layer[l + 1][j] instanceof Constant ? "b" : "") + ": " + weights[l][i][j]);
	}
	
	public void summarize() {
		System.out.println("Neurons:");
		for( int l = 0; l < layer.length; l++ ) {
			Map<String,Integer> counts = new HashMap<>();
			for( Function c : layer[l] ) {
				String s = c.getClass().toString();
				if( !counts.containsKey(s) )
					counts.put(s, 0);
				counts.put(s, counts.get(s)+1);
			}
			System.out.println(l+","+counts);
		}
		System.out.println("Weights:");	
		for (int l = 0; l < weights.length; l++) {
			DescriptiveStatistics ds = new DescriptiveStatistics();			
			int nans = 0;
			for (int i = 0; i < weights[l].length; i++)
				for (int j = 0; j < weights[l][i].length; j++)
					if( Double.isNaN( weights[l][i][j] ) )
							nans++;
					else
						ds.addValue(weights[l][i][j]);			
			System.out.println(l+", min:"+ds.getMin()+", mean: "+ds.getMean()+", max: "+ds.getMax()+", sd: "+ds.getStandardDeviation()+", N: "+ds.getN()+", NaNs: "+nans);							
		}
	}

	public int getNumWeights() {		
	    int nWeights = 0;
	    for (double[][] from : weights)
	        for (double[] to : from)
	            nWeights += to.length;
	    return nWeights;
	}

	public double[][][] getCopyOfWeights() {
		double[][][] w = new double[weights.length][][];
		for (int l = 0; l < weights.length; l++) {
			w[l] = new double[weights[l].length][];
			for (int i = 0; i < weights[l].length; i++)
				w[l][i] = Arrays.copyOf(weights[l][i], weights[l][i].length);
		}
		return w;
	}

	public void setWeights(double[][][] w) {
		this.weights = w;
	}
	
	public double[] getResiduals(List<double[]> inputs, List<double[]> targets) {
	    int output_size = targets.get(0).length;

	    double[] residuals = new double[inputs.size() * output_size];
	    for (int i = 0; i < inputs.size(); i++) {
	        double[] output = present(inputs.get(i) );
	        for (int j = 0; j < output_size; j++) {
	            residuals[i * output_size + j] = output[j] - targets.get(i)[j];
	        }
	    }
	    return residuals;
	}
    
    public static double[][][] deepCopy3DArray(double[][][] original) {
        if (original == null) {
            return null;
        }

        int dim1 = original.length;
        double[][][] copy = new double[dim1][][];

        for (int i = 0; i < dim1; i++) {
            if (original[i] != null) {
                int dim2 = original[i].length;
                copy[i] = new double[dim2][];
                for (int j = 0; j < dim2; j++) {
                    if (original[i][j] != null) {
                        copy[i][j] = original[i][j].clone(); // clone innermost 1D arrays
                    }
                }
            }
        }

        return copy;
    }
 }