package supervised.nnet.gwann;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Random;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import org.apache.log4j.Logger;
import org.jblas.DoubleMatrix;

import supervised.SupervisedUtils;
import supervised.nnet.NNet;
import supervised.nnet.activation.Constant;
import supervised.nnet.activation.Function;
import supervised.nnet.activation.Linear;
import supervised.nnet.activation.Sigmoid;
import utils.DataUtils;
import utils.GWRUtils;
import utils.Normalizer;
import utils.GWRUtils.GWKernel;
import utils.Normalizer.Transform;

public class GWANN extends NNet {

	private static Logger log = Logger.getLogger(GWANN.class);

	public GWANN(Function[][] l, double[][][] weights, double eta ) {
		super(l,weights,eta);
	}
	
	// online training with geographically weighted last layer
	// Use trainSimple
	@Deprecated 
	public void train(double[] x, double[] desired, double[][] sw ) {
		int ll = layer.length - 1; // index of last layer
		
		double[][] out = presentInt( x )[0];
		double[][] delta = new double[layer.length][];
				
		for (int l = ll; l > 0; l--) {	
			
			delta[l] = new double[layer[l].length];
			
			for (int i = 0; i < layer[l].length; i++) { // for each neuron of layer l
					
				double s = 0;	
				if( l == ll )
					s = out[l][i] - desired[i];
				else if( l == ll-1 ) {
					for (int j = 0; j < weights[l][i].length; j++)
						if( !( layer[l+1][j] instanceof Constant ) )	
							s += delta[l + 1][j] * weights[l][i][j] * sw[i][j];
				} else {
					for (int j = 0; j < weights[l][i].length; j++)
						if( !( layer[l+1][j] instanceof Constant ) )	
							s += delta[l + 1][j] * weights[l][i][j];
				}
																																	
				delta[l][i] = layer[l][i].fDevFOut(out[l][i]) * s;				
				// delta[l][j] = layer[l][j].fDev(net[l][j]) * s; 
			}		
		}
		
		// change weights to layer i
		for (int l = 0; l < ll; l++) 
			if( l == ll-1 ) {
				for (int i = 0; i < weights[l].length; i++) 												
					for (int j = 0; j < weights[l][i].length; j++) 
						weights[l][i][j] -= eta * delta[l+1][j] * out[l][i] * sw[i][j];
			} else {
				for (int i = 0; i < weights[l].length; i++) 												
					for (int j = 0; j < weights[l][i].length; j++) 
						weights[l][i][j] -= eta * delta[l+1][j] * out[l][i];
			}
	}
	
	// Use trainSimple
	@Deprecated 
	public void train( List<double[]> batch, Map<double[],double[][]> sampleWeights, int[] fa, int[] ta ) {
		
		int ll = layer.length - 1; // index of last layer
		double[][] delta = new double[layer.length][];
		double[][][] update = new double[layer.length-1][][];
		
		for( double[] x : batch ) {
			double[] desired = DataUtils.strip(x, ta);
			double[][] out = presentInt( DataUtils.strip(x, fa) )[0];
			double[][] sw = sampleWeights.get(x);
												
			for (int l = ll; l > 0; l--) {	
				
				delta[l] = new double[layer[l].length];
				if( update[l-1] == null )
					update[l-1] = new double[layer[l-1].length][layer[l].length];
								
				if( l == ll ) {
					for (int i = 0; i < layer[l].length; i++) { 
						
						double s = out[l][i] - desired[i];																															
						delta[l][i] = layer[l][i].fDevFOut(out[l][i]) * s;							
						for( int h = 0; h < layer[l-1].length; h++ ) 
							update[l-1][h][i] += out[l-1][h] * delta[l][i] * sw[h][i];	
					}
				} else if( l == ll-1 ) {
					for (int i = 0; i < layer[l].length; i++) { 
						
						double s = 0;	
						for (int j = 0; j < weights[l][i].length; j++)
							if( !( layer[l+1][j] instanceof Constant ) )
								s += delta[l + 1][j] * weights[l][i][j] * sw[i][j];
																																
						delta[l][i] = layer[l][i].fDevFOut(out[l][i]) * s;															
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
					weights[l][i][j] -= eta * update[l][i][j]/batch.size();
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
	
	public void trainFirstLayer( List<double[]> batch, Map<double[],double[][]> sampleWeights, int[] fa, int[] ta ) {
		
		int ll = layer.length - 1; // index of last layer
		double[][] delta = new double[layer.length][];
		double[][][] update = new double[layer.length-1][][];
		
		for( double[] x : batch ) {
			double[] desired = DataUtils.strip(x, ta);
			double[][] out = presentInt( DataUtils.strip(x, fa) )[0];
			double[][] sw = sampleWeights.get(x);
												
			for (int l = ll; l > 0; l--) {	
				
				delta[l] = new double[layer[l].length];
				if( update[l-1] == null )
					update[l-1] = new double[layer[l-1].length][layer[l].length];
				
				for (int i = 0; i < layer[l].length; i++) { // for each neuron of layer l
						
					double s = 0;	
					if( l == ll )
						s = out[l][i] - desired[i];
					else
						for (int j = 0; j < weights[l][i].length; j++)
							if( !( layer[l+1][j] instanceof Constant ) )
								s += delta[l + 1][j] * weights[l][i][j] * ( l == 0 ? sw[i][j] : 1);
																															
					delta[l][i] = layer[l][i].fDevFOut(out[l][i]) * s;				
					// delta[l][j] = layer[l][j].fDev(net[l][j]) * s; 
										
					for( int h = 0; h < layer[l-1].length; h++ ) 
						update[l-1][h][i] += out[l-1][h] * delta[l][i] * ( l == 1 ? sw[h][i] : 1 );	
				}		
			}
		}
		
		// change weights to layer i
		for (int l = 0; l < ll; l++) 
			for (int i = 0; i < weights[l].length; i++) 												
				for (int j = 0; j < weights[l][i].length; j++) 
					weights[l][i][j] -= eta * update[l][i][j]/batch.size();
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
	
	// batch training where weight for each connection is explicitly given
	public void trainTest( List<double[]> batch, Map<double[],double[][][]> sampleWeights, int[] fa, int[] ta, boolean presentWeights ) {
		
		int ll = layer.length - 1; // index of last layer
		double[][] delta = new double[layer.length][];
		double[][][] update = new double[layer.length-1][][];
		double[][][] sum = new double[layer.length-1][][];
		
		for( double[] x : batch ) {
			double[][][] sw = sampleWeights.get(x);
			double[] desired = DataUtils.strip(x, ta);
									
			double[][] out;
			if( presentWeights )
				out = presentInt( DataUtils.strip(x, fa), sw );
			else
				out = presentInt( DataUtils.strip(x, fa) )[0];
			
			for (int l = ll; l > 0; l--) {	
				
				delta[l] = new double[layer[l].length];
				if( update[l-1] == null )
					update[l-1] = new double[layer[l-1].length][layer[l].length];
				if( sum[l-1] == null )
					sum[l-1] = new double[layer[l-1].length][layer[l].length];
				
				for (int i = 0; i < layer[l].length; i++) { // for each neuron of layer l
						
					double s = 0;	
					if( l == ll )
						s = out[l][i] - desired[i];
					else 
						for (int j = 0; j < weights[l][i].length; j++)
							s += delta[l + 1][j] * weights[l][i][j] * sw[l][i][j]; 
																										
					delta[l][i] = layer[l][i].fDevFOut(out[l][i]) * s;				
					// delta[l][j] += layer[l][j].fDev(net[l][j]) * s; 
										
					for( int h = 0; h < layer[l-1].length; h++ ) {		
						sum[l-1][h][i] += sw[l-1][h][i];
						update[l-1][h][i] += delta[l][i] * out[l-1][h] * sw[l-1][h][i];
					}
				}		
			}
		}
		
		// change weights to layer i
		for (int l = 0; l < ll; l++) 
			for (int i = 0; i < weights[l].length; i++) 												
				for (int j = 0; j < weights[l][i].length; j++) 
					weights[l][i][j] -= eta * update[l][i][j];///sum[l][i][j]; // normalize?
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
	
	public static double getGoldenRatioBW_CV(double minRadius, double maxRadius, List<double[]> samples, int[] fa, int ta, DoubleMatrix W, List<Entry<List<Integer>, List<Integer>>> cvList, int nrHidden, int batchSize, double eta, int maxEpochs, int maxNoImp, GWKernel kernel, int threads) {
		double xU = maxRadius;
		double xL = minRadius;
		double eps = 1e-04;
		double R = (Math.sqrt(5) - 1) / 2.0;
		double d = R * (xU - xL);
		
		double x1 = xL + d;
		double x2 = xU - d;
		
		double f1 = getCost( samples, fa, ta, W, cvList, nrHidden, batchSize, eta, maxEpochs, maxNoImp, kernel, x1, threads );
		double f2 = getCost( samples, fa, ta, W, cvList, nrHidden, batchSize, eta, maxEpochs, maxNoImp, kernel, x2, threads );
		
		while ( Math.abs(d) > eps && Math.abs(f2 - f1) > eps ) {
			d = R * d;
			if (f1 < f2) {
				xL = x2;
				x2 = x1;
				x1 = xL + d;
				f2 = f1;
				f1 = getCost( samples, fa, ta, W, cvList, nrHidden, batchSize, eta, maxEpochs, maxNoImp, kernel, x1, threads );
			} else {
				xU = x1;
				x1 = x2;
				x2 = xU - d;
				f1 = f2;
				f2 = getCost( samples, fa, ta, W, cvList, nrHidden, batchSize, eta, maxEpochs, maxNoImp, kernel, x2, threads );
			}
		}
		return (f1 < f2) ? x1 : x2;
	}
	
	public static double getCost( List<double[]> samples, int[] fa, int ta, DoubleMatrix W, List<Entry<List<Integer>, List<Integer>>> cvList, int nrHidden, int batchSize, double eta, int maxEpochs, int maxNoImp, GWKernel kernel, double bw, int threads ) {
		ExecutorService es = Executors.newFixedThreadPool(threads);
		List<Future<List<Double>>> futures = new ArrayList<Future<List<Double>>>();

		for (final Entry<List<Integer>, List<Integer>> cvEntry : cvList) {
			final int seed = cvList.indexOf(cvEntry);

			futures.add(es.submit(new Callable<List<Double>>() {
				@Override
				public List<Double> call() throws Exception {
					Random r = new Random(seed);

					List<double[]> samplesTrain = new ArrayList<>();
					for (int i : cvEntry.getKey()) {
						double[] d = samples.get(i);
						samplesTrain.add(Arrays.copyOf(d, d.length));
					}

					List<double[]> samplesTest = new ArrayList<>();
					for (int i : cvEntry.getValue()) {
						double[] d = samples.get(i);
						samplesTest.add(Arrays.copyOf(d, d.length));
					}

					Normalizer n = new Normalizer(Transform.zScore, samplesTrain, fa);
					n.normalize(samplesTest);

					DoubleMatrix cvW_ = W.get(toIntArray(cvEntry.getKey()), toIntArray(cvEntry.getValue()));
					Map<double[], double[]> sampleWeights = GWANN.getSampleWeights(samplesTrain, cvW_, kernel, bw);

					List<Function> input = new ArrayList<>();
					while (input.size() < fa.length)
						input.add(new Linear());
					input.add(new Constant(1.0));

					// Order of output is important because of consistent offset calculation
					List<Function> output = new ArrayList<>();
					while (output.size() < samplesTest.size())
						output.add(new Linear());

					Function[][] layers;
					if (nrHidden > 0) {
						List<Function> hidden = new ArrayList<>();
						while (hidden.size() < nrHidden)
							hidden.add(new Sigmoid());
						hidden.add(new Constant(1.0));

						layers = new Function[][] { input.toArray(new Function[] {}), hidden.toArray(new Function[] {}), output.toArray(new Function[] {}) };
					} else {
						layers = new Function[][] { input.toArray(new Function[] {}), output.toArray(new Function[] {}) };
					}

					double[][][] weights = NNet.getFullyConnectedWeights(layers, initMode.gorot_unif, 0);
					GWANN gwnnet = new GWANN(layers, weights, eta);

					int[] tas = new int[output.size()];
					for (int i = 0; i < tas.length; i++)
						tas[i] = ta;

					List<Double> errors = new ArrayList<>();
					int noImp = 0;
					double bestError = Double.POSITIVE_INFINITY;
					for (int epoch = 0;; epoch++) {

						List<double[]> batchReservoir = new ArrayList<>(samplesTrain); // one epoch
						while (!batchReservoir.isEmpty()) {
							List<double[]> batch = new ArrayList<>();
							while (batch.size() < batchSize && !batchReservoir.isEmpty())
								batch.add(batchReservoir.remove(r.nextInt(batchReservoir.size())));
							gwnnet.trainSimple(batch, sampleWeights, fa, tas);
						}

						List<Double> responseTest = new ArrayList<>();
						for (int i = 0; i < samplesTest.size(); i++)
							responseTest.add(gwnnet.present(DataUtils.strip(samplesTest.get(i), fa))[i]);
						double testError = SupervisedUtils.getRMSE(responseTest, samplesTest, ta);
						errors.add(testError);

						if (testError < bestError) {
							bestError = testError;
							noImp = 0;
						} else
							noImp++;

						if (epoch >= maxEpochs || noImp > maxNoImp)
							return errors;
					}
				}
			}));
		}
		es.shutdown();

		// calculate mean error
		List<List<Double>> result = new ArrayList<List<Double>>();
		int minResultLength = Integer.MAX_VALUE;
		for (Future<List<Double>> f : futures)
			try {
				List<Double> ll = f.get();
				minResultLength = Math.min(minResultLength, ll.size());
				result.add(ll);
			} catch (InterruptedException | ExecutionException ex) {
				ex.printStackTrace();
			}

		double minMean = Double.POSITIVE_INFINITY;
		for (int i = 0; i < minResultLength; i++) { // for each it
			double mean = 0;
			for (int j = 0; j < result.size(); j++) // for each fold
				mean += result.get(j).get(i);
			mean /= result.size();
			if (mean < minMean) 
				minMean = mean;
		}		
		return minMean;
	}
}
