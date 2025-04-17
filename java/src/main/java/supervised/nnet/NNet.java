package supervised.nnet;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.CholeskyDecomposition;
import org.apache.commons.math3.linear.DecompositionSolver;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.NonPositiveDefiniteMatrixException;
import org.apache.commons.math3.linear.QRDecomposition;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.linear.SingularValueDecomposition;
import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.jblas.DoubleMatrix;

import supervised.SupervisedNet;
import supervised.nnet.activation.Constant;
import supervised.nnet.activation.Function;

public class NNet implements SupervisedNet {
	
	private static Logger log = LogManager.getLogger(NNet.class);
	
	public Function[][] layer;
	public double[][][] weights;
	protected double[] eta;

	double[][][] v = null, v_prev = null;

	public double lambda = Double.NaN;
	
	public static enum Optimizer {
		SGD, Momentum, Nesterov, RMSProp, Adam, Adam_ruder, LM, LM_ANA, LBFGS
	};

	protected Optimizer opt;

	@Deprecated
	public NNet(Function[][] layer, double[][][] weights, double eta, Optimizer opt) {
		this(layer, weights, null, opt, 0.0);

		this.eta = new double[layer.length];
		for (int i = 0; i < this.eta.length; i++)
			this.eta[i] = eta;
	}

	public NNet(Function[][] layer, double[][][] weights, double[] eta, Optimizer opt, double lambda) {

		this.lambda = lambda;
		this.layer = layer;
		this.eta = eta;
		this.weights = weights;
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

	@Override
	public double[] present(double[] x) {		
		double[][] out = presentInt(x, weights)[0];
		return out[out.length - 1]; // output of last layer
	}

	protected double[][][] presentInt(double[] x, double[][][] w) {
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
	            	out[l][i] = (layer[l][i] instanceof Constant) ? layer[l][i].f(Double.NaN) : layer[l][i].f(in[i]);          
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
	                nextIn[j] += w[l][i][j] * out[l][i];	            
	        
	        in = nextIn; // Move to the next layer
	    }
	    throw new IllegalStateException("Unexpected execution flow in presentInt");
	}
	
	@Override
	public void train(double t, double[] x, double[] desired) {
		List<double[]> xl = new ArrayList<>();
		List<double[]> yl = new ArrayList<>();
		xl.add(x);
		yl.add(desired);
		train(xl, yl);
		
		throw new RuntimeException("Dont use!");
	}
		
	protected double[][][] getErrorGradient( double[] x, double[] y ) {
		int ll = layer.length - 1; // index of last layer
		double[][][] error_grad = new double[layer.length - 1][][];
		double[][] delta = new double[layer.length - 1][];
		
		for( int l = ll; l > 0; l-- ) {
			delta[l-1] = new double[layer[l].length];
			error_grad[l-1] = new double[layer[l - 1].length][layer[l].length];
		}
		
		double[][] out = presentInt(x,weights)[0];

		for (int l = ll; l > 0; l--) {
				
			for (int i = 0; i < layer[l].length; i++) { // for each neuron i of layer l

				double error_signal = 0;
				if( l == ll )
					error_signal = (out[l][i] - y[i]);
				else 
					for (int j = 0; j < weights[l][i].length; j++)
						if (!(layer[l + 1][j] instanceof Constant) ) // ignore useless weight to constant neurons
							error_signal += delta[l][j] * weights[l][i][j];
										
				delta[l-1][i] = layer[l][i].fDevFOut(out[l][i]) * error_signal;
				for (int h = 0; h < layer[l - 1].length; h++)
					error_grad[l - 1][h][i] += out[l - 1][h] * delta[l-1][i];
			}
		}			
		return error_grad;
	}
	
	public static double mu = 0.9;
	public static double fac = 1.5;
    public static double epsilon = 1e-8;
	public static boolean center = false; 
	
	protected void update( Optimizer opt, double[][][] errorGrad, double[] eta ) {
				
		if( opt == Optimizer.SGD ) {
			for (int l = 0; l < weights.length; l++)
				for (int i = 0; i < weights[l].length; i++)
					for (int j = 0; j < weights[l][i].length; j++) {
						if( lambda > 0 )
							weights[l][i][j] -= eta[l] * errorGrad[l][i][j] - lambda * weights[l][i][j];
						else
							weights[l][i][j] -= eta[l] * errorGrad[l][i][j]; 
					}
						
		} else if( opt == Optimizer.Momentum ) {
			//double mu = 0.9;
			for (int l = 0; l < weights.length; l++)
				for (int i = 0; i < weights[l].length; i++)
					for (int j = 0; j < weights[l][i].length; j++) {
						v[l][i][j] = mu * v[l][i][j] - eta[l] * errorGrad[l][i][j]; 
						
						if( lambda > 0 )
							weights[l][i][j] += v[l][i][j] - lambda * weights[l][i][j];
						else
							weights[l][i][j] += v[l][i][j];
						
					}
		} else if( opt == Optimizer.Nesterov ) {
			//double mu = 0.9;
			for (int l = 0; l < weights.length; l++)
				for (int i = 0; i < weights[l].length; i++)
					for (int j = 0; j < weights[l][i].length; j++) {
						v_prev[l][i][j] = v[l][i][j];
						
						if( lambda > 0 )
							v[l][i][j] = mu * v[l][i][j] - eta[l] * errorGrad[l][i][j] - lambda * weights[l][i][j];
						else 
							v[l][i][j] = mu * v[l][i][j] - eta[l] * errorGrad[l][i][j];
						
						weights[l][i][j] += -mu * v_prev[l][i][j] + v[l][i][j] + mu * v[l][i][j];
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
		} else if( opt == Optimizer.Adam ) {
			double b1 = 0.9, b2 = 0.999, eps = Math.pow(10, -8);
			double p1 = (1 - Math.pow(b1, t));

			for (int l = 0; l < weights.length; l++)
				for (int i = 0; i < weights[l].length; i++)
					for (int j = 0; j < weights[l][i].length; j++) {

						double g = errorGrad[l][i][j];
						v_prev[l][i][j] = b1 * v_prev[l][i][j] + (1 - b1) * g; // mt
						v[l][i][j] = b2 * v[l][i][j] + (1 - b2) * g * g; // vt

						double mt_hat = v_prev[l][i][j] / p1 + (1 - b1) * p1;
						double vt_hat = v[l][i][j] / p1;
						weights[l][i][j] -= eta[l] * mt_hat / (Math.sqrt(vt_hat) + eps);
					}
		} else if( opt == Optimizer.Adam_ruder ) {
			double b1 = 0.9, b2 = 0.999, eps = Math.pow(10, -8);

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
					}
		}
	}

	protected int t = 1;
	private double error_prev = Double.MAX_VALUE;
	
	public double[] toDouble1D( double[][][] w) {				
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
	
	public void train(List<double[]> x, List<double[]> y) {
						
		if( opt == Optimizer.LM ) {
						
			double[] residuals = computeResiduals(x, y);

			RealMatrix jacobian = new Array2DRowRealMatrix( computeJacobian_chatgpt(x, epsilon, center).toArray2() );
			
			RealMatrix jacobianT = jacobian.transpose();           
			RealMatrix hessianApprox = jacobianT.multiply(jacobian);
			RealVector gradient = jacobianT.operate(new ArrayRealVector(residuals));

			// Apply damping (Levenberg–Marquardt adjustment)
			RealMatrix identity = MatrixUtils.createRealIdentityMatrix(hessianApprox.getRowDimension());
			//hessianApprox = hessianApprox.add(identity.scalarMultiply(lambda));
			hessianApprox = hessianApprox.add(identity.scalarMultiply(lambda * hessianApprox.getNorm()));
			
			// Solve for weight updates
			RealVector deltaWeights;
			try {
				DecompositionSolver solver = new CholeskyDecomposition(hessianApprox).getSolver();
				deltaWeights = solver.solve(gradient.mapMultiply(-1));
			} catch(NonPositiveDefiniteMatrixException e) {
				try {
					DecompositionSolver solver = new QRDecomposition(hessianApprox).getSolver();
					deltaWeights = solver.solve(gradient.mapMultiply(-1));
				} catch (Exception e2) {
					DecompositionSolver solver = new SingularValueDecomposition(hessianApprox).getSolver();
					deltaWeights = solver.solve(gradient.mapMultiply(-1));
				}
			}							
			
			// Update weights			
			double alpha = eta[0];
			//double alpha = Math.max(eta[0], Math.min(1.0, 1.0 / (1.0 + lambda))); // not that useful
			int k = 0;
			for( int a = 0; a < weights.length; a++ )
				for( int b = 0; b < weights[a].length; b++ )
					for( int c = 0; c < weights[a][b].length; c++ ) 
						weights[a][b][c] += alpha * deltaWeights.getEntry(k++);
			
			double error = 0;
			for( double r : computeResiduals(x, y) )
				error += Math.pow(r,2);     
            
			lambda = (error > error_prev) ? lambda * fac : lambda / fac;       
			//lambda = Math.min(Math.max(lambda, 1e-6), 1e6);     
            error_prev = error;
                        				            
		} else {						
			List<double[][][]> l = new ArrayList<>();
			for( int i = 0; i < x.size(); i++ ) 
				l.add( getErrorGradient( x.get(i), y.get(i) ) );
			double[][][] errorGrad = calculateMeanOf3DList(l);							
			update( opt, errorGrad, eta );
		}
		t++;
	}
	
	protected static double[][][] calculateMeanOf3DList(List<double[][][]> arrays) {
		
		double[][][] first = arrays.get(0);
		
		double[][][] mean = new double[first.length][][];		
		for( int i = 0; i < first.length; i++ ) {
			mean[i] = new double[first[i].length][];
			for( int j = 0; j < first[i].length; j++ )
				mean[i][j] = new double[first[i][j].length];
		}
		
		for( double[][][] array : arrays ) 
			for( int i = 0; i < first.length; i++ ) 
				for( int j = 0; j < first[i].length; j++ )
					for( int k = 0; k < first[i][j].length; k++ )
						mean[i][j][k] += array[i][j][k]/arrays.size();	
		    
	    return mean;
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
		int sum = 0;
		for (int i = 0; i < weights.length; i++)
			for (int j = 0; j < weights[i].length; j++)
				sum += weights[i][j].length;
		return sum;
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
	
	private double[] forward(double[] x, double[][][] w ) {
		double[][] out = presentInt(x, w)[0];
		return out[out.length - 1]; // output of last layer
	}
		
	// chatgpt
	public DoubleMatrix computeJacobian_chatgpt(List<double[]> inputs, double epsilon, boolean central) {
	    // Calculate total number of weights
	    int nWeights = 0;
	    for (double[][] from : weights)
	        for (double[] to : from)
	            nWeights += to.length;
	    
	    // Store original outputs for all inputs
	    double[][] originalOutputs = new double[inputs.size()][];
	    for (int i = 0; i < inputs.size(); i++) 
	        originalOutputs[i] = forward(inputs.get(i), weights);
	    
	    // Initialize Jacobian matrix (rows: inputs, columns: weights * output_dims)
	    int outputDims = originalOutputs[0].length;
	    DoubleMatrix jacobian = new DoubleMatrix(inputs.size(), nWeights * outputDims);

	    int weightIndex = 0; 
	    
	    for (int a = 0; a < weights.length; a++) {
	        for (int b = 0; b < weights[a].length; b++) {
	            for (int c = 0; c < weights[a][b].length; c++) {

	                         	 
	                double[][] perturbedOutPlus = new double[inputs.size()][];
	                double[][] perturbedOutMinus = new double[inputs.size()][];
	                double originalWeight = weights[a][b][c]; 
	                
	                // Perturb positively
	                weights[a][b][c] = originalWeight + epsilon;
	                for (int i = 0; i < inputs.size(); i++) 
	                	perturbedOutPlus[i] = forward(inputs.get(i), weights);
	                	                
	                if (central) {
	                	// Perturb negatively
	                	weights[a][b][c] = originalWeight - epsilon;
	                	for (int i = 0; i < inputs.size(); i++) 
	                		perturbedOutMinus[i] = forward(inputs.get(i), weights);	                	
	                }	                
	                weights[a][b][c] = originalWeight;
	                
	                // Compute finite differences for each input and output dimension
	                for (int i = 0; i < inputs.size(); i++) {
	                    for (int j = 0; j < outputDims; j++) {
	                        double derivative;
	                        if (central) {
	                            derivative = (perturbedOutPlus[i][j] - perturbedOutMinus[i][j]) / (2 * epsilon);
	                        } else {
	                            derivative = (perturbedOutPlus[i][j] - originalOutputs[i][j]) / epsilon;
	                        }
	                        jacobian.put(i, weightIndex * outputDims + j, derivative);
	                    }
	                }
	                weightIndex++;
	            }
	        }
	    }
	    return jacobian;
	}

    // Compute residuals (errors) for all samples
    private double[] computeResiduals(List<double[]> inputs, List<double[]> targets) {
    	int output_size = targets.get(0).length;
    	    	
        double[] residuals = new double[inputs.size() * output_size];
        for (int i = 0; i < inputs.size(); i++) {
            double[] output = forward(inputs.get(i), weights);
            for (int j = 0; j < output_size; j++) {
                residuals[i * output_size + j] = output[j] - targets.get(i)[j];
            }
        }
        return residuals;
    }
    
    public double getENP(DoubleMatrix jacobian) {
        
        int nRows = jacobian.getRows();
        int nCols = jacobian.getColumns();
        DoubleMatrix fisherInformation = new DoubleMatrix(nCols, nCols);
        
        for (int i = 0; i < nRows; i++) {
        	DoubleMatrix jacobianRow = jacobian.getRow(i); 
        	DoubleMatrix outerProduct = jacobianRow.transpose().mmul(jacobianRow);  
        	fisherInformation.addi(outerProduct); 
        }
        
        fisherInformation.muli( 1.0/nRows );
        return fisherInformation.diag().sum();
    }
    
    public double getENP_2(List<double[]> x, List<double[]> y) {    	
    	List<double[][][]> grads = new ArrayList<>();
    	for( int i = 0; i < x.size(); i++ )
    		grads.add( getErrorGradient(x.get(i), y.get(i) ));

    	double trace = 0;
    	for (double[][][] grad : grads) {
    		double[] flat = toDouble1D(grad);
    		for (double v : flat)
    			trace += v * v;
    	}

    	return trace / grads.size(); // average over samples
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