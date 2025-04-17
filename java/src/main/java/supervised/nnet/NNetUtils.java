package supervised.nnet;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map.Entry;
import java.util.Random;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import supervised.SupervisedUtils;
import supervised.nnet.activation.Constant;
import supervised.nnet.activation.Function;
import supervised.nnet.activation.Linear;
import supervised.nnet.activation.TanH;
import utils.ListNormalizer;
import utils.Normalizer.Transform;

public class NNetUtils {
	
	private static Logger log = LogManager.getLogger(NNetUtils.class);

	public static enum initMode {
		gorot_unif, norm05
	}

	public static List<List<Double>> getErrors_CV(List<double[]> xArray, List<double[]> yArray, List<Entry<List<Integer>, List<Integer>>> innerCvList, double[] eta, int batchSize, NNet.Optimizer opt, double lambda, int[] nrHidden, int maxIt, int patience, int threads, Transform[] expTrans, Transform[] respTrans ) {
		ExecutorService innerEs = Executors.newFixedThreadPool(threads);
		List<Future<List<Double>>> futures = new ArrayList<Future<List<Double>>>();
		
		for (int innerK = 0; innerK < innerCvList.size(); innerK++) {
			Entry<List<Integer>, List<Integer>> innerCvEntry = innerCvList.get(innerK);
	
			futures.add(innerEs.submit(new Callable<List<Double>>() {
				@Override
				public List<Double> call() throws Exception {
					
					List<Integer> trainIdx = innerCvEntry.getKey();
					List<Integer> testIdx = innerCvEntry.getValue(); 
										
					List<double[]> xTrain = new ArrayList<>();
					List<double[]> yTrain = new ArrayList<>();
					for (int i : trainIdx) {
						xTrain.add(xArray.get(i));
						yTrain.add(yArray.get(i));
					}
				
					List<double[]> xVal = new ArrayList<>();
					List<double[]> yVal = new ArrayList<>();
					for (int i : testIdx) {
						xVal.add(xArray.get(i));
						yVal.add(yArray.get(i));
					}
				
					ReturnObject ro = NNetUtils.buildNNet(xTrain, yTrain, xVal, yVal, nrHidden, eta, opt, lambda, batchSize, maxIt, patience, expTrans, respTrans );
					return ro.errors;
				}
			}));
		}
		innerEs.shutdown();
		List<List<Double>> errors = new ArrayList<>();
		try {
			for( Future<List<Double>> f : futures )
				errors.add( f.get() );
		} catch (InterruptedException | ExecutionException e) {
			e.printStackTrace();
		} 
		return errors;
	}

	// cleaner interface
	public static ReturnObject buildNNet(
			List<double[]> xTrain_, List<double[]> yTrain_, 
			List<double[]> xTest_, List<double[]> yTest_, 
			int[] nrHidden, double[] eta, NNet.Optimizer opt, 
			double lambda, int batchSize, 
			int maxIt, int patience, 
			Transform[] expTrans, Transform[] respTrans
			) {
		Random r = new Random(0);
	
		List<double[]> x_train = new ArrayList<>();
		for (double[] d : xTrain_)
			x_train.add(Arrays.copyOf(d, d.length));	
		List<double[]> y_train = new ArrayList<>();
		for (double[] d : yTrain_) 
			y_train.add(Arrays.copyOf(d, d.length));
			
		List<double[]> x_test = new ArrayList<>();
		for (double[] d : xTest_)
			x_test.add(Arrays.copyOf(d, d.length));
	
		double[] test_desired_not_normalized = new double[yTest_.size()];
		List<double[]> y_test = new ArrayList<>();
		for( int i = 0; i < yTest_.size(); i++ ) {
			double[] d = yTest_.get(i);
			test_desired_not_normalized[i] = d[0];
			y_test.add( d );
		}
			
		ListNormalizer ln_x = new ListNormalizer(expTrans, x_train);
		ln_x.normalize(x_test);
		
		ListNormalizer ln_y = new ListNormalizer(respTrans, y_train);	
		ln_y.normalize(y_test);
	
		List<Function[]> layerList = new ArrayList<>();
		List<Function> input = new ArrayList<>();
		while (input.size() < x_train.get(0).length)
			input.add(new Linear());
		input.add(new Constant(1.0));
		layerList.add(input.toArray(new Function[] {}));
	
		for (int nh : nrHidden) {
			if (nh == 0)
				continue;
			List<Function> hidden0 = new ArrayList<>();
			while (hidden0.size() < nh)
				hidden0.add(new TanH());
			hidden0.add(new Constant(1.0));
			layerList.add(hidden0.toArray(new Function[] {}));
		}
	
		List<Function> output = new ArrayList<>();
		while (output.size() < y_test.get(0).length )
			output.add(new Linear());
		layerList.add(output.toArray(new Function[] {}));
	
		Function[][] layers = layerList.toArray(new Function[][] {});
		double[][][] weights = NNetUtils.getFullyConnectedWeights(layers, NNetUtils.initMode.gorot_unif, 0);
	
		NNet nnet = new NNet(layers, weights, eta, opt, lambda);
	
		List<Integer> batchReservoir = new ArrayList<>();
		List<Double> errors = new ArrayList<>();
		int no_imp = 0;
		
		double test_error_best = Double.POSITIVE_INFINITY;
		double[][][] weights_best = null;
		int it_best = 0;
		for (int it = 0; it < maxIt && no_imp < patience; it++) {
	
			if( batchSize > 0 ) {
				List<double[]> x = new ArrayList<>();
				List<double[]> y = new ArrayList<>();
				while (x.size() < batchSize) {
					if (batchReservoir.isEmpty())
						for (int j = 0; j < x_train.size(); j++)
							batchReservoir.add(j);
					int idx = batchReservoir.remove(r.nextInt(batchReservoir.size()));
					x.add(x_train.get(idx));
					y.add(y_train.get(idx));
				}
				nnet.train(x, y);
			} else 
				nnet.train(x_train, y_train);
						
			// get denormalized response
			double[] test_response= new double[x_test.size()];
			for (int i = 0; i < x_test.size(); i++) {
				double[] d = nnet.present(x_test.get(i));
				ln_y.denormalize(d, 0);
				test_response[i] = d[0];
			}
	
			double test_error = SupervisedUtils.getRMSE(test_response, test_desired_not_normalized);
			errors.add(test_error);
	
			if (test_error < test_error_best) {
				test_error_best = test_error;
				weights_best = nnet.getCopyOfWeights();
				it_best = it;
				no_imp = 0;
			} else
				no_imp++;
		}
		
		if( weights_best != null)
			nnet.setWeights(weights_best);
	
		// get test response and denormalize
		List<double[]> test_response = new ArrayList<>();
		for (int i = 0; i < x_test.size(); i++)
			test_response.add(nnet.present(x_test.get(i)));
		ln_y.denormalize(test_response);
				
		// response, only first
		double[] response = new double[test_response.size()];
		for (int i = 0; i < x_test.size(); i++)
			response[i] = test_response.get(i)[0];
												
		ReturnObject ro = new ReturnObject();
		ro.errors = errors.subList(0, it_best+1); // last error is cur error
		ro.rmse = SupervisedUtils.getRMSE(response, test_desired_not_normalized);
		ro.r2 = SupervisedUtils.getR2(response, test_desired_not_normalized );
		ro.nnet = nnet;
		ro.prediction = test_response;	
		ro.ln_x = ln_x;
		ro.ln_y = ln_y;
		return ro;
	}

	public static double[][][] getFullyConnectedWeights(Function[][] layer, NNetUtils.initMode im, int seed) {
		Random r = new Random(seed);
	
		double[][][] weights = new double[layer.length - 1][][];
		for (int l = 0; l < weights.length; l++) { // i, over layer
			int fanIn = layer[l].length;
			int fanOut = layer[l + 1].length;
	
			weights[l] = new double[layer[l].length][];
			for (int i = 0; i < weights[l].length; i++) {
	
				weights[l][i] = new double[layer[l + 1].length];
				for (int j = 0; j < weights[l][i].length; j++)
					if (layer[l + 1][j] instanceof Constant)
						weights[l][i][j] = Double.NaN;
					else {
						if (im == NNetUtils.initMode.norm05)
							weights[l][i][j] = r.nextGaussian() * 0.5;
						else {
							// https://keras.rstudio.com/reference/initializer_glorot_uniform.html
							double limit = Math.sqrt(6.0 / (fanIn + fanOut));
							weights[l][i][j] = r.nextDouble() * limit * 2 - limit;
						}
					}
			}
		}
		return weights;
	}
	
	public static double[] getBestErrorParams( List<List<Double>> errors ) {
		int minSize = Integer.MAX_VALUE;
		for( List<Double> f : errors )	
			minSize = Math.min(f.size(), minSize);
		assert minSize > 0;
		
		double[] mean = new double[minSize];
		for( List<Double> f : errors )
			for( int i = 0; i < mean.length; i++ )
				mean[i] += f.get(i)/errors.size();
		
		double minMean = mean[0];
		int minMeanIdx = 0;
		for( int i = 1; i < mean.length; i++ )
			if (mean[i] < minMean ) {
				minMean = mean[i];
				minMeanIdx = i;
			}	
		return new double[] { minMean, minMeanIdx };
	}
}
