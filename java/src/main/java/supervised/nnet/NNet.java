package supervised.nnet;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;

import supervised.SupervisedNet;
import supervised.nnet.activation.Constant;
import supervised.nnet.activation.Function;

public class NNet implements SupervisedNet {
	
	public static double mu = 0.9;
	public Function[][] layer;
	public double[][][] weights;
	protected double[] eta;

	double[][][] v = null, v_prev = null;

	public static enum Optimizer {
		SGD, Momentum, Nesterov, RMSProp, Adam, Adam_ruder
	};

	protected Optimizer m;
	public double lambda = 0.0;

	public NNet(Function[][] layer, double[][][] weights, double eta, Optimizer m) {
		this(layer, weights, null, m);

		this.eta = new double[layer.length];
		for (int i = 0; i < this.eta.length; i++)
			this.eta[i] = eta;
	}

	public NNet(Function[][] layer, double[][][] weights, double[] eta, Optimizer m) {

		this.layer = layer;
		this.eta = eta;

		this.weights = weights;
		this.m = m;

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
		for (int l = 0;; l++) {

			net[l] = in;
			out[l] = new double[layer[l].length];

			// apply activation to net input
			int inIdx = 0;
			for (int i = 0; i < layer[l].length; i++)
				if (l == 0 && layer[l][i] instanceof Constant)
					out[l][i] = layer[l][i].f(Double.NaN);
				else
					out[l][i] = layer[l][i].f(in[inIdx++]);

			if (l == layer.length - 1)
				return new double[][][] { out, net };

			// apply weights, calculate new net input
			in = new double[weights[l][0].length]; // number of neurons in l+1 connected to neuron 0 in l
			for (int i = 0; i < weights[l].length; i++)
				for (int j = 0; j < weights[l][i].length; j++)
					in[j] += weights[l][i][j] * out[l][i];
		}
	}

	@Override
	public void train(double t, double[] x, double[] desired) {
		List<double[]> xl = new ArrayList<>();
		List<double[]> yl = new ArrayList<>();
		xl.add(x);
		yl.add(desired);
		train(xl, yl);
	}

	protected double[][][] getErrorGradient(List<double[]> x, List<double[]> y) {

		int ll = layer.length - 1; // index of last layer
		double[][][] errorGrad = new double[layer.length - 1][][];
		double[][] delta = new double[layer.length][];
		for (int e = 0; e < x.size(); e++) {
			double[] desired = y.get(e);
			double[][] out = presentInt(x.get(e))[0];

			for (int l = ll; l > 0; l--) {

				delta[l] = new double[layer[l].length];

				if (errorGrad[l - 1] == null)
					errorGrad[l - 1] = new double[layer[l - 1].length][layer[l].length];

				if (l == ll) {
					for (int i = 0; i < layer[l].length; i++) { // for each neuron of layer l

						double s = (out[l][i] - desired[i]);
						delta[l][i] = layer[l][i].fDevFOut(out[l][i]) * s;

						for (int h = 0; h < layer[l - 1].length; h++)
							errorGrad[l - 1][h][i] += out[l - 1][h] * delta[l][i];
					}
				} else {
					for (int i = 0; i < layer[l].length; i++) { // for each neuron of layer l

						double s = 0;
						for (int j = 0; j < weights[l][i].length; j++)
							if (!(layer[l + 1][j] instanceof Constant))
								s += delta[l + 1][j] * weights[l][i][j];
						delta[l][i] = layer[l][i].fDevFOut(out[l][i]) * s;

						for (int h = 0; h < layer[l - 1].length; h++)
							errorGrad[l - 1][h][i] += out[l - 1][h] * delta[l][i];
					}
				}
			}
		}
		return errorGrad;
	}
	
	protected void update( Optimizer opt, double[][][] errorGrad, double[] leta, double lambda ) {
		if( opt == Optimizer.SGD ) {
			int lInit = 0;
			for (int l = lInit; l < weights.length; l++)
				for (int i = 0; i < weights[l].length; i++)
					for (int j = 0; j < weights[l][i].length; j++)
						weights[l][i][j] -= leta[l] * errorGrad[l][i][j] - leta[l] * lambda * weights[l][i][j];
		} else if( opt == Optimizer.Momentum ) {
			int lInit = 0;
			double mu = 0.9;
			for (int l = lInit; l < weights.length; l++)
				for (int i = 0; i < weights[l].length; i++)
					for (int j = 0; j < weights[l][i].length; j++) {
						v[l][i][j] = mu * v[l][i][j] - leta[l] * errorGrad[l][i][j]; 
						weights[l][i][j] += v[l][i][j] - leta[l] * lambda * weights[l][i][j]; 
					}
		} else if( opt == Optimizer.Nesterov ) {
			int lInit = 0;
			//double mu = 0.9;
			for (int l = lInit; l < weights.length; l++)
				for (int i = 0; i < weights[l].length; i++)
					for (int j = 0; j < weights[l][i].length; j++) {
						v_prev[l][i][j] = v[l][i][j];
						v[l][i][j] = mu * v[l][i][j] - leta[l] * errorGrad[l][i][j] - leta[l] * lambda * weights[l][i][j];
						weights[l][i][j] += -mu * v_prev[l][i][j] + v[l][i][j] + mu * v[l][i][j];
					}
		} else if( opt == Optimizer.RMSProp ) {
			double eps = Math.pow(10, -8);
			int lInit = 0;
			double gamma = 0.9;
			for (int l = lInit; l < weights.length; l++)
				for (int i = 0; i < weights[l].length; i++)
					for (int j = 0; j < weights[l][i].length; j++) {
						v[l][i][j] = gamma * v[l][i][j] + (1 - gamma) * errorGrad[l][i][j] * errorGrad[l][i][j];
						weights[l][i][j] -= leta[l] * errorGrad[l][i][j] / Math.sqrt(v[l][i][j] + eps);
					}
			v_prev = errorGrad;
		} else if( opt == Optimizer.Adam ) {
			int lInit = 0;
			double b1 = 0.9, b2 = 0.999, eps = Math.pow(10, -8);
			double p1 = (1 - Math.pow(b1, t));

			for (int l = lInit; l < weights.length; l++)
				for (int i = 0; i < weights[l].length; i++)
					for (int j = 0; j < weights[l][i].length; j++) {

						double g = errorGrad[l][i][j];
						v_prev[l][i][j] = b1 * v_prev[l][i][j] + (1 - b1) * g; // mt
						v[l][i][j] = b2 * v[l][i][j] + (1 - b2) * g * g; // vt

						double mt_hat = v_prev[l][i][j] / p1 + (1 - b1) * p1;
						double vt_hat = v[l][i][j] / p1;
						weights[l][i][j] -= leta[l] * mt_hat / (Math.sqrt(vt_hat) + eps);
					}
		} else if( opt == Optimizer.Adam_ruder ) {
			int lInit = 0;
			double b1 = 0.9, b2 = 0.999, eps = Math.pow(10, -8);

			double p1 = (1 - Math.pow(b1, t));
			double p2 = (1 - Math.pow(b2, t));

			for (int l = lInit; l < weights.length; l++)
				for (int i = 0; i < weights[l].length; i++)
					for (int j = 0; j < weights[l][i].length; j++) {

						double g = errorGrad[l][i][j];
						v_prev[l][i][j] = b1 * v_prev[l][i][j] + (1 - b1) * g; // mt
						v[l][i][j] = b2 * v[l][i][j] + (1 - b2) * g * g; // vt

						// https://ruder.io/optimizing-gradient-descent/index.html#adam
						double mt_hat = v_prev[l][i][j] / p1;
						double vt_hat = v[l][i][j] / p2;
						weights[l][i][j] -= leta[l] * mt_hat / (Math.sqrt(vt_hat) + eps);
					}
		}
	}

	protected int t = 1;

	public void train(List<double[]> x, List<double[]> y) {
		double[][][] errorGrad = getErrorGradient(x, y);
		double[] leta = new double[eta.length];
		for (int i = 0; i < leta.length; i++)
			leta[i] = eta[i] / x.size();
		update( m, errorGrad, leta, lambda );
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
}