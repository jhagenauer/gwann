package supervised.nnet.gwann;

import java.util.AbstractMap;
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
	
	public static ReturnObject run(
			double[][] xArray, double[] yArray, double[][] dm, 
			int[] tIdx, int[] pIdx, 
			double nrHidden, double batchSize, String optim, double eta, boolean linOut, 
			String krnl, double bw_, boolean adaptive, 
			boolean gridSearch, double minBw, double maxBw, double steps_,
			double iterations, double patience, 
			double folds, double repeats,
			double threads) {

		List<Integer>trainIdx = new ArrayList<>();
		for( int i = 0; i < tIdx.length; i++ )
			trainIdx.add( (int)tIdx[i]-1 );
		
		List<Integer> predIdx = new ArrayList<>();
		for( int i = 0; i < pIdx.length; i++ )
			predIdx.add( (int)pIdx[i]-1 );
		
		assert xArray.length == dm.length & dm.length == dm[0].length;
		
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

		DoubleMatrix W = new DoubleMatrix(dm);
		
		List<Double> v = new ArrayList<>();
		for (double w : W.data)
			v.add(w);
		v = new ArrayList<>(new HashSet<>(v));
		Collections.sort(v);

		double min = minBw < 0 ? ( adaptive ? 5 : v.get(1) / 4 ) : minBw;
		double max = maxBw < 0 ? ( adaptive ? W.rows / 2 : v.get(v.size() - 1) / 2 ) : maxBw;
		
		double bestValError = Double.POSITIVE_INFINITY;
		double bestValBw = adaptive ? (int) (min + (max - min) / 2) : min + (max - min) / 2;
		double prevBestValError = Double.POSITIVE_INFINITY;
		int bestIts = -1;
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
			if( bw_ > 0 ) {
				ll.clear();
				ll.add(bw_);
			} else if( gridSearch ) {
				ll.clear();
				for( double a = min; a <=max; a+= (max-min)/steps )
					ll.add(a);		
			}
			ll.removeAll(bwDone);

			if (!ll.isEmpty())
				System.out.println(bwShrinkFactor + ", current best bandwidth: " + bestValBw + ", RMSE:" + bestValError + ", bandwidths to test: " + ll);
			for (double bw : ll) {
				
				ExecutorService innerEs = Executors.newFixedThreadPool((int) threads);
				List<Future<List<Double>>> futures = new ArrayList<Future<List<Double>>>();
				
				List<Integer> ti = new ArrayList<>();
				for( int i : trainIdx )
					ti.add(i);

				List<Entry<List<Integer>, List<Integer>>> innerCvList = SupervisedUtils.getKFoldCVList( (int)folds, (int)repeats, ti, seed);
				for (int innerK = 0; innerK < innerCvList.size(); innerK++) {
					Entry<List<Integer>, List<Integer>> innerCvEntry = innerCvList.get(innerK);
					futures.add(innerEs.submit(new Callable<List<Double>>() {
						@Override
						public List<Double> call() throws Exception {							
							return buildGWANN(xArray, yArray, W, innerCvEntry, (int)nrHidden, 0, eta, opt, (int)batchSize, (int)iterations, (int)patience, kernel, bw, adaptive, seed).errors;								
						}
					}));
				}
				innerEs.shutdown();
				double[] mm;
				if( iterations >= 0 ) {
					double mean = 0;
					for (Future<List<Double>> f : futures) {
						try {
							mean += f.get().get((int)iterations);
						} catch (InterruptedException | ExecutionException ex) {
							ex.printStackTrace();
						}
					}
					mm = new double[] { mean/futures.size(), (int)iterations };
				} else
				 mm	= getMinMeanIdx(futures);
				if (mm[0] < bestValError) {
					bestValError = mm[0];
					bestValBw = bw;
					bestIts = (int)mm[1];
				}
				bwDone.add(bw);
				System.out.println(bw+" "+Arrays.toString(mm));
			}
			if (prevBestValError - bestValError < eps || ll.isEmpty() )
				break;
			prevBestValError = bestValError;
		}

		System.out.println("Cross-validation results (folds: "+folds+", repeats: "+repeats+"):");
		System.out.println("Bandwidth: " + bestValBw);
		System.out.println("Iterations: " + bestIts);
		System.out.println("RMSE: " + bestValError);

		System.out.println("Building final model with bandwidth "+bestValBw+" and "+bestIts+" iterations...");	
		Entry<List<Integer>,List<Integer>> ee = new AbstractMap.SimpleEntry<List<Integer>, List<Integer>>( trainIdx, predIdx ); 					
		BuiltGwann tg = buildGWANN(xArray, yArray, W, ee, (int)nrHidden, 0, eta, opt, (int)batchSize, bestIts, (int)patience, kernel, bestValBw, adaptive, seed);
	
		ReturnObject ro = new ReturnObject();
		ro.predictions = tg.predictions;
		ro.weights = tg.gwann.weights;
		ro.rmse = bestValError;
		ro.its = bestIts;
		ro.bw = bestValBw;

		return ro;			
		
	}

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
	
	static BuiltGwann buildGWANN(double[][] xArray, double[] yArray, DoubleMatrix W, Entry<List<Integer>,List<Integer>> innerCvEntry, int nrHidden0, int nrHidden1, double eta, Optimizer opt, 
			int batchSize, int iterations, int patience, GWKernel kernel, double bw, boolean adaptive, int seed ) {
		Random r = new Random(seed);
			
		List<double[]> xTrain = new ArrayList<>();
		List<double[]> yTrain = new ArrayList<>();
		for (int i : innerCvEntry.getKey()) {
			xTrain.add(Arrays.copyOf(xArray[i], xArray[i].length));

			double[] y = new double[innerCvEntry.getValue().size()];
			for (int j = 0; j < y.length; j++)
				y[j] = yArray[i];
			yTrain.add(y);
		}

		List<double[]> xVal = new ArrayList<>();
		List<double[]> yVal = new ArrayList<>();
		for (int i : innerCvEntry.getValue()) {
			xVal.add(Arrays.copyOf(xArray[i], xArray[i].length));

			double[] y = new double[innerCvEntry.getValue().size()];
			for (int j = 0; j < y.length; j++)
				y[j] = yArray[i];
			yVal.add(y);
		}
	
		Normalizer n = new Normalizer(Normalizer.Transform.zScore, xTrain);
		n.normalize(xVal);
	
		DoubleMatrix cvTrainW = W.getRows(DataUtils.toIntArray(innerCvEntry.getKey())); // train to samples
		DoubleMatrix cvValW = W.get(DataUtils.toIntArray(innerCvEntry.getKey()), DataUtils.toIntArray(innerCvEntry.getValue())); // train to test
		DoubleMatrix kW = adaptive ? GWRUtils.getKernelWeights(cvTrainW, cvValW, kernel, (int) bw) : GWRUtils.getKernelWeights(cvValW, kernel, bw);
		
		List<Function[]> layerList = new ArrayList<>();
		
		List<Function> input = new ArrayList<>();
		while (input.size() < xArray[0].length)
			input.add(new Linear());
		input.add(new Constant(1.0));
		layerList.add(input.toArray(new Function[] {} ) );
		
		if( nrHidden0 > 0 ) {
			List<Function> hidden0 = new ArrayList<>();
			while (hidden0.size() < nrHidden0)
				hidden0.add(new TanH());
			hidden0.add(new Constant(1.0));
			layerList.add(hidden0.toArray(new Function[] {} ) );
		}
		
		if( nrHidden1 > 0 ) {
			List<Function> hidden1 = new ArrayList<>();
			while (hidden1.size() < nrHidden1)
				hidden1.add(new TanH());
			hidden1.add(new Constant(1.0));
			layerList.add( hidden1.toArray(new Function[] {} ) );
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
	
			if (it >= iterations || noImp >= patience  ) {
				BuiltGwann tg = new BuiltGwann();
				tg.gwann = gwann;
				tg.errors= errors; // error for each iteration
				tg.predictions = preds; // predictions of last iterations
				return tg;	
			}
		}
	}
}