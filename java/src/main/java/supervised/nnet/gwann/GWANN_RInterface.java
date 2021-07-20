package supervised.nnet.gwann;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Map.Entry;
import java.util.Random;
import java.util.Set;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import org.jblas.DoubleMatrix;

import supervised.SupervisedUtils;
import supervised.nnet.NNet;
import supervised.nnet.NNet.Optimizer;
import supervised.nnet.NNet.initMode;
import supervised.nnet.activation.Constant;
import supervised.nnet.activation.Function;
import supervised.nnet.activation.Linear;
import supervised.nnet.activation.TanH;
import utils.DataUtils;
import utils.GWRUtils;
import utils.GWRUtils.GWKernel;
import utils.Normalizer;

public class GWANN_RInterface {
	
	public static double[] getMinMeanIdx(List<Future<List<Double>>> futures) {
		List<List<Double>> result = new ArrayList<List<Double>>();
		for (Future<List<Double>> f : futures)
			try {
				result.add(f.get());
			} catch (InterruptedException | ExecutionException ex) {
				ex.printStackTrace();
			}

		double minMean = Double.POSITIVE_INFINITY;
		int minMeanIdx = -1;
		out: for (int i = 0;; i++) { // for each it
			double mean = 0;
			for (int j = 0; j < result.size(); j++) { // for each fold
				if (i >= result.get(j).size())
					break out;
				mean += result.get(j).get(i);
			}
			mean /= result.size();

			if (mean < minMean) {
				minMean = mean;
				minMeanIdx = i;
			}
		}
		return new double[] { minMean, minMeanIdx };
	}
	
	public static ReturnObject run(
			double[][] xArray_train, double[] yArray_train, double[][] W_train,
			double[][] xArray_pred, double[] yArray_pred, double[][] W_train_pred,
			double nrHidden, double batchSize, String optim, double eta, boolean linOut, 
			String krnl, double bw_, boolean adaptive, 
			String bwSearch, double bwMin, double bwMax, double steps_,
			double iterations, double patience, 
			double folds, double repeats,
			double permutations,
			double threads) {
		
		assert xArray_train.length == W_train.length &  
				W_train.length == W_train[0].length & // quadratic
				W_train_pred.length == xArray_train.length & W_train_pred[0].length == xArray_pred.length; 
		
		GWKernel kernel;
		if (krnl.equalsIgnoreCase("gaussian"))
			kernel = GWKernel.gaussian;
		else if (krnl.equalsIgnoreCase("bisquare"))
			kernel = GWKernel.bisquare;
		else if (krnl.equalsIgnoreCase("boxcar"))
			kernel = GWKernel.boxcar;
		else if (krnl.equalsIgnoreCase("tricube"))
			kernel = GWKernel.tricube;
		else if (krnl.equalsIgnoreCase("exponential"))
			kernel = GWKernel.exponential;
		else
			throw new RuntimeException("Unknown kernel");
		
		Optimizer opt;
		if( optim.equalsIgnoreCase("nesterov") )
			opt = Optimizer.Nesterov;
		else if( optim.equalsIgnoreCase("momentum"))
			opt = Optimizer.Momentum;
		else if( optim.equalsIgnoreCase("sgd"))
			opt = Optimizer.SGD;
		else
			throw new RuntimeException("Unknown optimizer");

		int seed = 0;

		double eps = 1e-03; // oder lieber 4?
		int pow = 3;
		final int steps = (int)steps_ < 0 ? 10 : (int)steps_;

		DoubleMatrix W = new DoubleMatrix(W_train);
		List<Entry<List<Integer>, List<Integer>>> innerCvList = SupervisedUtils.getKFoldCVList( (int)folds, (int)repeats, xArray_train.length, seed);	
				
		List<Double> v = new ArrayList<>();
		for (double w : W.data)
			v.add(w);
		v = new ArrayList<>(new HashSet<>(v));
		Collections.sort(v);
		
		double min = bwMin < 0 ? ( adaptive ? 5 : W.min() / 4 ) : bwMin;
		double max = bwMax < 0 ? ( adaptive ? W.rows / 2 : W.max() / 2 ) : bwMax;
				
		double bestValBw = adaptive ? (int) (min + (max - min) / 2) : min + (max - min) / 2;
		int bestIts = -1;	
		
		double bestValError = Double.POSITIVE_INFINITY;
		double prevBestValError = Double.POSITIVE_INFINITY;
				
		if( bw_ > 0 ) {
			System.out.println("Pre-specified bandwidth...");
			double[] m = getParamsCV(xArray_train, yArray_train, W, innerCvList, kernel, bw_, adaptive, eta, (int)batchSize, opt, (int)nrHidden, (int)iterations, (int)patience, (int)seed, (int)threads);
			bestValError = m[0];
			bestValBw = bw_;
			bestIts = (int)m[1];
		} else if( bwSearch.equalsIgnoreCase("goldenSection") ) { // determine best bw using golden section search 
			System.out.println("Golden section search...");
			double[] m = getParamsWithGoldenSection(min, max, xArray_train, yArray_train, W, innerCvList, kernel, adaptive, eta, (int)batchSize, opt, (int)nrHidden, (int)iterations, (int)patience, (int)seed, (int)threads);
			bestValError = m[0];
			bestValBw = m[1];
			bestIts = (int)m[2];
		} else { // determine best bw using grid search or local search routine 
			
			Set<Double> bwDone = new HashSet<>();
			for (int bwShrinkFactor = 2;; bwShrinkFactor *= 2) {
	
				Set<Double> l = new HashSet<>();
				for (int i = 1; i <= steps; i++) {
					double a = bestValBw + Math.pow((double) i / steps, pow) * Math.min(max - bestValBw, (max - min) / bwShrinkFactor);
					double b = bestValBw - Math.pow((double) i / steps, pow) * Math.min(bestValBw - min, (max - min) / bwShrinkFactor);
					if (adaptive) {
						a = (double) Math.round(a);
						b = (double) Math.round(b);
					}
					if (a <= max)
						l.add(a);
					if (b >= min)
						l.add(b);
				}
				l.add(bestValBw);
	
				List<Double> ll = new ArrayList<>(l);
				Collections.sort(ll);
				
				// bandwidth given?
				if( bwSearch.equalsIgnoreCase("grid") ) {
					ll.clear();
					for( double a = min; a <=max; a+= (max-min)/steps )
						ll.add(a);
					System.out.println("Grid search...");
				} else
					System.out.println("Local search routine...");
				ll.removeAll(bwDone);
	
				System.out.println(bwShrinkFactor + ", current best bandwidth: " + bestValBw + ", RMSE:" + bestValError + ", bandwidths to test: " + ll);
				for (double bw : ll) {				
					double[] mm = getParamsCV(xArray_train, yArray_train, W, innerCvList, kernel, bestValBw, adaptive, eta, (int)batchSize, opt, (int)nrHidden, (int)iterations, (int)patience, seed, (int)threads);
					if (mm[0] < bestValError) {
						bestValError = mm[0];
						bestIts = (int)mm[1];
						bestValBw = bw;
					}
					bwDone.add(bw);
					System.out.println(bw+" "+Arrays.toString(mm));
				}
				if (prevBestValError - bestValError < eps || ll.isEmpty() )
					break;
				prevBestValError = bestValError;
			}
		}

		System.out.println("Cross-validation results (folds: "+folds+", repeats: "+repeats+"):");
		System.out.println("Bandwidth: " + bestValBw);
		System.out.println("Iterations: " + bestIts);
		System.out.println("RMSE: " + bestValError);
		
		double[][][] imp = null;
		if( permutations > 0 ) { // importance
			System.out.println("Calculating feature importance...");
			BuiltGwann bg = buildGWANN(xArray_train, yArray_train, new DoubleMatrix(W_train), xArray_train, yArray_train, new DoubleMatrix(W_train), new int[] { (int)nrHidden }, eta, opt, (int)batchSize, bestIts, Integer.MAX_VALUE, kernel, bestValBw, adaptive, seed);
			double[][] preds = bg.predictions;
									
			imp = new double[xArray_train[0].length][preds.length][preds[0].length];
			for( int i = 0; i < xArray_train[0].length; i++ ) { // for each variable
				System.out.println("Feature "+i);
				
				for( int j = 0; j < permutations; j++ ) {
					
					double[][] copy = Arrays.stream(xArray_train).map(double[]::clone).toArray(double[][]::new);
					List<Double> l = new ArrayList<>();
					for( int k = 0; k < copy.length; k++ )
						l.add(copy[k][i]);
					Collections.shuffle(l);
					for( int k = 0; k < l.size(); k++ )
						copy[k][i] = l.get(k);
					
					double[][] preds_ = new double[copy.length][];
					for( int k = 0; k < copy.length; k++ )
						preds_[k] = bg.gwann.present(copy[k]);
					
					for( int k = 0; k < preds.length; k++ )
						for( int p = 0; p < preds[0].length; p++ )
							imp[i][k][p] += ( Math.pow(preds_[k][p] - yArray_train[k],2) - Math.pow(preds[k][p] - yArray_train[k],2) )/permutations;
				}			
			}
		}

		System.out.println("Building final model with bandwidth "+bestValBw+" and "+bestIts+" iterations...");				
		BuiltGwann tg = buildGWANN(xArray_train, yArray_train, new DoubleMatrix(W_train), xArray_pred, yArray_pred, new DoubleMatrix(W_train_pred), new int[] { (int)nrHidden }, eta, opt, (int)batchSize, bestIts, Integer.MAX_VALUE, kernel, bestValBw, adaptive, seed);
	
		ReturnObject ro = new ReturnObject();
		ro.predictions = tg.predictions;
		ro.importance = imp;
		ro.weights = tg.gwann.weights;
		ro.rmse = bestValError;
		ro.its = bestIts;
		ro.bw = bestValBw;
		return ro;			
	}
	
	static BuiltGwann buildGWANN(double[][] xArray_train, double[] yArray_train, DoubleMatrix W_train, double[][] xArray_test, double[] yArray_test, DoubleMatrix W_train_test, int[] nrHidden, double eta, Optimizer opt, 
			int batchSize, int iterations, int patience, GWKernel kernel, double bw, boolean adaptive, int seed ) {
		Random r = new Random(seed);
			
		List<double[]> xTrain = new ArrayList<>();
		List<double[]> yTrain = new ArrayList<>();
		for (int i = 0; i < xArray_train.length; i++ ) {
			xTrain.add(Arrays.copyOf(xArray_train[i], xArray_train[i].length));

			double[] y = new double[yArray_train.length];
			for (int j = 0; j < y.length; j++)
				y[j] = yArray_train[i];
			yTrain.add(y);
		}
		
		List<double[]> xVal = new ArrayList<>();
		List<double[]> yVal = new ArrayList<>();
		for (int i = 0; i < xArray_test.length; i++ ) {
			xVal.add(Arrays.copyOf(xArray_test[i], xArray_test[i].length));

			double[] y = new double[yArray_test.length]; // nr of outputs
			for (int j = 0; j < y.length; j++)
				y[j] = yArray_test[i];
			yVal.add(y);
		}
		
		DoubleMatrix kW = adaptive ? GWRUtils.getKernelWeights(W_train, W_train_test, kernel, (int) bw) : GWRUtils.getKernelWeights(W_train_test, kernel, bw);
		
		List<Function[]> layerList = new ArrayList<>();
		
		List<Function> input = new ArrayList<>();
		while (input.size() < xArray_test[0].length)
			input.add(new Linear());
		input.add(new Constant(1.0));
		layerList.add(input.toArray(new Function[] {} ) );
		
		for( int nh : nrHidden ) {
			List<Function> hidden0 = new ArrayList<>();
			while (hidden0.size() < nh)
				hidden0.add(new TanH());
			hidden0.add(new Constant(1.0));
			layerList.add(hidden0.toArray(new Function[] {} ) );
		}
		
		List<Function> output = new ArrayList<>();
		while (output.size() < yVal.size())
			output.add(new Linear());
		layerList.add(output.toArray(new Function[] {} ) );	
				
		Function[][] layers = layerList.toArray( new Function[][] {} );
		double[][][] weights = NNet.getFullyConnectedWeights(layers, initMode.gorot_unif, 0);
		GWANN gwann = new GWANN(layers, weights, eta, opt);
				
		List<Integer> batchReservoir = new ArrayList<>();
		List<Double> errors = new ArrayList<>();
		int noImp = 0;
		double localBestValError = Double.POSITIVE_INFINITY;
		for (int it = 0;; it++) {
	
			List<double[]> x = new ArrayList<>();
			List<double[]> y = new ArrayList<>();
			List<double[]> gwWeights = new ArrayList<>();
			while (x.size() < batchSize) {
				if (batchReservoir.isEmpty())
					for (int j = 0; j < xTrain.size(); j++)
						batchReservoir.add(j);
				int idx = batchReservoir.remove(r.nextInt(batchReservoir.size()));
				x.add(xTrain.get(idx));
				y.add(yTrain.get(idx));
				gwWeights.add(kW.getRow(idx).data);
			}
			gwann.train(x, y, gwWeights);
					
			double[][] preds = new double[xVal.size()][];
			for (int i = 0; i < xVal.size(); i++ ) {
				double[][] out = gwann.presentInt(xVal.get(i))[0];
				preds[i] = out[layers.length - 1];
			}

			List<Double> response = new ArrayList<>();
			for( int i = 0; i < preds.length; i++ )
				response.add( preds[i][i] );
			double valError = SupervisedUtils.getRMSE(response, yVal, 0);
			errors.add(valError);
	
			if (valError < localBestValError) {
				localBestValError = valError;
				noImp = 0;
			} else
				noImp++;
	
			if ( ( iterations >= 0 && it >= iterations ) || noImp >= patience  ) {
				BuiltGwann tg = new BuiltGwann();
				tg.gwann = gwann;
				tg.errors= errors; // error for each iteration
				tg.predictions = preds; // predictions of last iterations
				return tg;	
			}
		}
	}
	
	@Deprecated
	static BuiltGwann buildGWANN(double[][] xArray, double[] yArray, DoubleMatrix W, List<Integer> trainIdx, List<Integer> testIdx, int[] nrHidden, double eta, Optimizer opt, 
			int batchSize, int iterations, int patience, GWKernel kernel, double bw, boolean adaptive, int seed ) {
		Random r = new Random(seed);
			
		List<double[]> xTrain = new ArrayList<>();
		List<double[]> yTrain = new ArrayList<>();
		for (int i : trainIdx ) {
			xTrain.add(Arrays.copyOf(xArray[i], xArray[i].length));

			double[] y = new double[testIdx.size()];
			for (int j = 0; j < y.length; j++)
				y[j] = yArray[i];
			yTrain.add(y);
		}
		
		List<double[]> xVal = new ArrayList<>();
		List<double[]> yVal = new ArrayList<>();
		for (int i : testIdx) {
			xVal.add(Arrays.copyOf(xArray[i], xArray[i].length));

			double[] y = new double[testIdx.size()]; // nr of outputs
			for (int j = 0; j < y.length; j++)
				y[j] = yArray[i];
			yVal.add(y);
		}
	
		Normalizer n = new Normalizer(Normalizer.Transform.zScore, xTrain);
		n.normalize(xVal);
	
		int[] trainIdxA = DataUtils.toIntArray(trainIdx);
		DoubleMatrix cvValW = W.get( trainIdxA, DataUtils.toIntArray(testIdx)); // train to test
		DoubleMatrix kW = adaptive ? GWRUtils.getKernelWeights(W.get(trainIdxA,trainIdxA), cvValW, kernel, (int) bw) : GWRUtils.getKernelWeights(cvValW, kernel, bw);
		
		List<Function[]> layerList = new ArrayList<>();
		
		List<Function> input = new ArrayList<>();
		while (input.size() < xArray[0].length)
			input.add(new Linear());
		input.add(new Constant(1.0));
		layerList.add(input.toArray(new Function[] {} ) );
		
		for( int nh : nrHidden ) {
			List<Function> hidden0 = new ArrayList<>();
			while (hidden0.size() < nh)
				hidden0.add(new TanH());
			hidden0.add(new Constant(1.0));
			layerList.add(hidden0.toArray(new Function[] {} ) );
		}
		
		List<Function> output = new ArrayList<>();
		while (output.size() < yVal.size())
			output.add(new Linear());
		layerList.add(output.toArray(new Function[] {} ) );	
				
		Function[][] layers = layerList.toArray( new Function[][] {} );
		double[][][] weights = NNet.getFullyConnectedWeights(layers, initMode.gorot_unif, 0);
		GWANN gwann = new GWANN(layers, weights, eta, opt);
				
		List<Integer> batchReservoir = new ArrayList<>();
		List<Double> errors = new ArrayList<>();
		int noImp = 0;
		double localBestValError = Double.POSITIVE_INFINITY;
		for (int it = 0;; it++) {
	
			List<double[]> x = new ArrayList<>();
			List<double[]> y = new ArrayList<>();
			List<double[]> gwWeights = new ArrayList<>();
			while (x.size() < batchSize) {
				if (batchReservoir.isEmpty())
					for (int j = 0; j < xTrain.size(); j++)
						batchReservoir.add(j);
				int idx = batchReservoir.remove(r.nextInt(batchReservoir.size()));
				x.add(xTrain.get(idx));
				y.add(yTrain.get(idx));
				gwWeights.add(kW.getRow(idx).data);
			}
			gwann.train(x, y, gwWeights);
					
			double[][] preds = new double[xVal.size()][];
			for (int i = 0; i < xVal.size(); i++ ) {
				double[][] out = gwann.presentInt(xVal.get(i))[0];
				preds[i] = out[layers.length - 1];
			}

			List<Double> response = new ArrayList<>();
			for( int i = 0; i < preds.length; i++ )
				response.add( preds[i][i] );
			double valError = SupervisedUtils.getRMSE(response, yVal, 0);
			errors.add(valError);
	
			if (valError < localBestValError) {
				localBestValError = valError;
				noImp = 0;
			} else
				noImp++;
	
			if ( ( iterations >= 0 && it >= iterations ) || noImp >= patience  ) {
				BuiltGwann tg = new BuiltGwann();
				tg.gwann = gwann;
				tg.errors= errors; // error for each iteration
				tg.predictions = preds; // predictions of last iterations
				return tg;	
			}
		}
	}
	
	public static double[] getParamsWithGoldenSection(double minRadius, double maxRadius, 
			double[][] xArray, double[] yArray, DoubleMatrix W, List<Entry<List<Integer>, List<Integer>>> innerCvList, GWKernel kernel, boolean adaptive, double eta, int batchSize, Optimizer opt, int nrHidden, int iterations, int patience, int seed, int threads ) {
		double xU = maxRadius;
		double xL = minRadius;
		double eps = 1e-04;
		double R = (Math.sqrt(5) - 1) / 2.0;
		double d = R * (xU - xL);
		
		double x1 = xL + d;
		double x2 = xU - d;
		double[] f1 = getParamsCV(xArray, yArray, W, innerCvList, kernel, x1, adaptive, eta, batchSize, opt, nrHidden, iterations, patience, seed,threads);
		double[] f2 = getParamsCV(xArray, yArray, W, innerCvList, kernel, x2, adaptive, eta, batchSize, opt, nrHidden, iterations, patience, seed,threads);
		
		double d1 = f2[0] - f1[0];
		
		while ((Math.abs(d) > eps) && (Math.abs(d1) > eps)) {
			d = R * d;
			if (f1[0] < f2[0]) {
				xL = x2;
				x2 = x1;
				x1 = xL + d;
				f2 = f1;
				f1 = getParamsCV(xArray, yArray, W, innerCvList, kernel, x1, adaptive, eta, batchSize, opt, nrHidden, iterations, patience, seed,threads);
			} else {
				xU = x1;
				x1 = x2;
				x2 = xU - d;
				f1 = f2;
				f2 = getParamsCV(xArray, yArray, W, innerCvList, kernel, x2, adaptive, eta, batchSize, opt, nrHidden, iterations, patience, seed,threads);
			}
			d1 = f2[0] - f1[0];
			System.out.println(d+", "+d1);
		}
		// returns cost, bandwidth, iterations
		if( f1[0] < f2[0] )
			return new double[] { f1[0], adaptive ? Math.round(x1) : x1, f1[1]};
		else 
			return new double[] { f2[0], adaptive ? Math.round(x2) : x2, f2[1]};
	}
	
	public static double[] getParamsCV(double[][] xArray, double[] yArray, DoubleMatrix W, List<Entry<List<Integer>, List<Integer>>> innerCvList, GWKernel kernel, double bw, boolean adaptive, double eta, int batchSize, Optimizer opt, int nrHidden, int iterations, int patience, int seed, int threads ) {
		ExecutorService innerEs = Executors.newFixedThreadPool((int) threads);
		List<Future<List<Double>>> futures = new ArrayList<Future<List<Double>>>();				
		
		for ( Entry<List<Integer>, List<Integer>> innerCvEntry : innerCvList ) {
			futures.add(innerEs.submit(new Callable<List<Double>>() {
				@Override
				public List<Double> call() throws Exception {	
					return buildGWANN(xArray, yArray, W, innerCvEntry.getKey(), innerCvEntry.getValue(), new int[] { (int)nrHidden }, eta, opt, (int)batchSize, (int)iterations, (int)patience, kernel, bw, adaptive, seed).errors;								
				}
			}));
		}
		innerEs.shutdown();
		double[] mm;
		
		if( iterations >= 0 ) {
			double mean = 0;
			for (Future<List<Double>> f : futures) {
				try {
					if( iterations > f.get().size() )
						System.err.println("Iterations set too large. Either decrease iterations or increase patience!!!");
					mean += f.get().get((int)iterations);
				} catch (InterruptedException | ExecutionException ex) {
					ex.printStackTrace();
				}
			}
			mm = new double[] { mean/futures.size(), (int)iterations };
		} else
			mm = getMinMeanIdx(futures);
		return mm;
	}
}