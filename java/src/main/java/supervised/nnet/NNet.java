package supervised.nnet;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.DecompositionSolver;
import org.apache.commons.math3.linear.QRDecomposition;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;
import org.jblas.DoubleMatrix;

import supervised.SupervisedNet;
import supervised.SupervisedUtils;
import supervised.nnet.activation.Constant;
import supervised.nnet.activation.Function;

public class NNet implements SupervisedNet {
	
	public Function[][] layer;
	public double[][][] weights;
	protected double[] eta;

	double[][][] v = null, v_prev = null;

	public double lambda = Double.NaN;
	public static enum Optimizer {
		SGD, Momentum, Nesterov, RMSProp, Adam, Adam_ruder, LM, LBFGS
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
	                if (layer[l][i] instanceof Constant) 
	                    out[l][i] = layer[l][i].f(Double.NaN);
	                else 
	                    out[l][i] = layer[l][i].f(in[i]);	            
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

	protected double[][][] getErrorGradient(List<double[]> batch, List<double[]> y) {

		int ll = layer.length - 1; // index of last layer
		double[][][] error_grad = new double[layer.length - 1][][];
		double[][] delta = new double[layer.length - 1][];
		
		for( int l = ll; l > 0; l-- ) {
			delta[l-1] = new double[layer[l].length];
			error_grad[l-1] = new double[layer[l - 1].length][layer[l].length];
		}
		
		for (int e = 0; e < batch.size(); e++) {
			double[] desired = y.get(e);
			double[][] out = presentInt(batch.get(e),weights)[0];

			for (int l = ll; l > 0; l--) {
				
				for (int i = 0; i < layer[l].length; i++) { // for each neuron i of layer l

					double error_signal = 0;
					if( l == ll )
						error_signal = (out[l][i] - desired[i]);
					else 
						for (int j = 0; j < weights[l][i].length; j++)
							if (!(layer[l + 1][j] instanceof Constant) ) // ignore useless weight to constant neurons
								error_signal += delta[l][j] * weights[l][i][j];
										
					delta[l-1][i] = layer[l][i].fDevFOut(out[l][i]) * error_signal;
					for (int h = 0; h < layer[l - 1].length; h++)
						error_grad[l - 1][h][i] += out[l - 1][h] * delta[l-1][i];
				}
			}
		}
		
		for (int l = 0; l < error_grad.length; l++) 
		    for (int h = 0; h < error_grad[l].length; h++) 
		        for (int i = 0; i < error_grad[l][h].length; i++) 
		            error_grad[l][h][i] /= batch.size();
			
		return error_grad;
	}
	
	public static double mu = 0.9;
	public static double decay = -1;
	protected void update( Optimizer opt, double[][][] errorGrad, double[] leta ) {
		double[] eta = new double[leta.length];
		if( decay > 0 ) 
			for( int i = 0; i < leta.length; i++ )
				eta[i] = leta[i] * Math.pow(decay,t);
		else
			eta = leta;
				
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
			double eps = Math.pow(10, -8);
			double gamma = 0.9;
			for (int l = 0; l < weights.length; l++)
				for (int i = 0; i < weights[l].length; i++)
					for (int j = 0; j < weights[l][i].length; j++) {
						v[l][i][j] = gamma * v[l][i][j] + (1 - gamma) * errorGrad[l][i][j] * errorGrad[l][i][j];
						weights[l][i][j] -= eta[l] * errorGrad[l][i][j] / Math.sqrt(v[l][i][j] + eps);
					}
			v_prev = errorGrad;
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
		List<Double> l = new ArrayList<>();
		for( int a = 0; a < w.length; a++ )
			for( int b = 0; b < w[a].length; b++ )
				for( int c = 0; c < w[a][b].length; c++ ) 
					l.add( w[a][b][c] );
		return SupervisedUtils.toDoubleArray(l);				
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
						
		    double epsilon = 1e-4; 
			double[] residuals = computeResiduals(x, y);

			RealMatrix jacobian = new Array2DRowRealMatrix( computeJacobian(x, epsilon).toArray2() ); 
			RealMatrix jacobianT = jacobian.transpose();           
			RealMatrix hessianApprox = jacobianT.multiply(jacobian);
			RealVector gradient = jacobianT.operate(new ArrayRealVector(residuals));

			// Apply damping (Levenberg–Marquardt adjustment)
			for (int i = 0; i < hessianApprox.getRowDimension(); i++) 
				hessianApprox.addToEntry(i, i, lambda+0.0000000001 );

			// Solve for weight updates
			//DecompositionSolver solver = new LUDecomposition(hessianApprox).getSolver();            
			DecompositionSolver solver = new QRDecomposition(hessianApprox).getSolver();
			RealVector deltaWeights = solver.solve(gradient.mapMultiply(-1));
			
			// Update weights            
			int k = 0;
			for( int a = 0; a < weights.length; a++ )
				for( int b = 0; b < weights[a].length; b++ )
					for( int c = 0; c < weights[a][b].length; c++ ) 
						weights[a][b][c] += eta[a] * deltaWeights.getEntry(k++);
						
			double error = computeError(x, y);     
            lambda = (error > error_prev) ? lambda * 1.25 : lambda / 1.25;            
            error_prev = error;
				            
		} else if( opt == Optimizer.LBFGS ) {
			
			/*ObjectiveFunction objective = new ObjectiveFunction(point -> {
				setWeights1D(point);
				return computeError(x, y);
	       });

	        ObjectiveFunctionGradient gradient = new ObjectiveFunctionGradient(point -> {
	        	setWeights1D(point);
	        	return toDouble1D( getErrorGradient(x, y) );
	        });
	        
	        // Optimize with L-BFGS
	        LBFGS optimizer = new LBFGS();
	        PointValuePair result = optimizer.optimize(
	            new MaxEval(1000),
	            objective,
	            gradient,
	            GoalType.MINIMIZE,
	            new InitialGuess( toDouble1D(weights) )
	        );
	        
	        setWeights1D(result.getPoint() );*/
	        
		} else {
			double[][][] errorGrad = getErrorGradient(x, y);
			update( opt, errorGrad, eta );
		}
		t++;
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

	public int getNumParameters() {
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
	
	public DoubleMatrix computeJacobian(List<double[]> inputs, double epsilon) {
		
        int nWeights = 0;
        for( double[][] from : weights )
        	for( double[] to : from )
        		nWeights += to.length;
        
        double[][] originalOutputs = new double[inputs.size()][];
        for (int i = 0; i < inputs.size(); i++) 
            originalOutputs[i] = forward(inputs.get(i), weights);
               
        // rows: outputs/activation functions, columns: weights
        DoubleMatrix jacobian = new DoubleMatrix(inputs.size(), nWeights);
                      
    	int k = 0;
    	for( int a = 0; a < weights.length; a++ )
    		for( int b = 0; b < weights[a].length; b++ )
    			for( int c = 0; c < weights[a][b].length; c++ ) {

    				double[][] perturbed_out_plus = new double[inputs.size()][];
    				weights[a][b][c] += epsilon;  
    		        for (int i = 0; i < inputs.size(); i++) 
    		        	perturbed_out_plus[i] = forward(inputs.get(i), weights);        		            
    		        weights[a][b][c] -= epsilon; // Reset weight
    		            		            		            
    		         // Compute partial derivatives
    		         for (int i = 0; i < inputs.size(); i++) 	               		                		             
    		             for (int j = 0; j < originalOutputs[i].length; j++)     
    		                jacobian.put(i, k, (perturbed_out_plus[i][j] - originalOutputs[i][j]) / epsilon);    		                 		             		             		           
    		         k++;
    			}    
        return jacobian;
    }
	
	public DoubleMatrix computeJacobianLowRank(List<double[]> inputs, double epsilon, int rank) {
	    // Anzahl der Gewichte bestimmen
	    int nWeights = 0;
	    for (double[][] from : weights)
	        for (double[] to : from)
	            nWeights += to.length;

	    // Original-Outputs berechnen
	    double[][] originalOutputs = new double[inputs.size()][];
	    for (int i = 0; i < inputs.size(); i++)
	        originalOutputs[i] = forward(inputs.get(i), weights);

	    // Anzahl der Outputs
	    int nOutputs = originalOutputs[0].length;

	    // Zufällige Vektoren für die Low-Rank-Approximation generieren
	    DoubleMatrix randomVectors = DoubleMatrix.randn(nWeights, rank); // Zufällige Projektionen
	    DoubleMatrix lowRankOutputs = new DoubleMatrix(inputs.size(), rank); // Für gestörte Outputs

	    // Perturbationen anwenden
	    int k = 0;
	    for (int a = 0; a < weights.length; a++) {
	        for (int b = 0; b < weights[a].length; b++) {
	            for (int c = 0; c < weights[a][b].length; c++) {
	                // Zufällige Projektionen auf das Gewicht anwenden
	                for (int r = 0; r < rank; r++) {
	                    double perturbation = epsilon * randomVectors.get(k, r);
	                    weights[a][b][c] += perturbation;

	                    // Gestörte Outputs berechnen
	                    for (int i = 0; i < inputs.size(); i++) {
	                        double[] perturbedOutput = forward(inputs.get(i), weights);
	                        for (int j = 0; j < nOutputs; j++) {
	                            lowRankOutputs.put(i, r, lowRankOutputs.get(i, r) +
	                                    (perturbedOutput[j] - originalOutputs[i][j]) / perturbation);
	                        }
	                    }

	                    // Gewicht zurücksetzen
	                    weights[a][b][c] -= perturbation;
	                }
	                k++;
	            }
	        }
	    }

	    // Jacobi-Matrix als Low-Rank-Approximation (U * V^T)
	    return lowRankOutputs.mmul(randomVectors.transpose());
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

    // Compute total mean squared error
    private double computeError(List<double[]> inputs, List<double[]> targets) {
    	int outputSize = targets.get(0).length;
        double error = 0;
        for (int i = 0; i < inputs.size(); i++) {
            double[] output = forward(inputs.get(i), weights);
            for (int j = 0; j < outputSize; j++) 
                error += Math.pow( output[j] - targets.get(i)[j], 2);
        }
        return error / inputs.size();
    }
            
    public double getENP(List<double[]> inputs) {
    	DoubleMatrix jacobian = computeJacobian(inputs, 0.0001);
        
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
}