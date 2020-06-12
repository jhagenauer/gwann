package supervised.nnet.gwann;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
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
import utils.GWRUtils;
import utils.GWRUtils.GWKernel;
import utils.Normalizer;

public class GWANN_RInterface {
	volatile static Integer uLim = Integer.MAX_VALUE;
	
	public static ReturnObject run(double[][] xArray, double[] yArray, double[][] dmX, double[][] dmP, double nrHidden, double batchSize, String optim, double eta, boolean linOut, String krnl, double bw_, boolean adaptive, double iterations, double patience, double threads) {

		assert xArray.length == dmX.length & dmX.length == dmX[0].length;
		
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
		else if( optim.equalsIgnoreCase("adam"))
			opt = Optimizer.Adam;
		else if( optim.equalsIgnoreCase("sgd"))
			opt = Optimizer.SGD;
		else
			throw new RuntimeException("Unknown optimizer");

		int seed = 0;

		double eps = 1e-03; // oder lieber 4?
		int pow = 3;
		final int steps = 10;

		DoubleMatrix W = new DoubleMatrix(dmX);
		DoubleMatrix predW = new DoubleMatrix(dmP);
		
		List<Double> v = new ArrayList<>();
		for (double w : W.data)
			v.add(w);
		v = new ArrayList<>(new HashSet<>(v));
		Collections.sort(v);

		double min = adaptive ? 5 : v.get(1) / 4;
		double max = adaptive ? W.rows / 4 : v.get(v.size() - 1) / 4;

		double bestValError = Double.POSITIVE_INFINITY;
		double bestValBw = adaptive ? (int) ((max - min) / 4) : (max - min) / 4;
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
			
			if( bw_ > 0 ) {
				ll.clear();
				ll.add(bw_);
			}
			ll.removeAll(bwDone);

			if (!ll.isEmpty())
				System.out.println("sf: " + bwShrinkFactor + ", current best bandwidth: " + bestValBw + ", RMSE:" + bestValError + ", bandwidths to test: " + ll);
			for (double bw : ll) {
				uLim = Integer.MAX_VALUE;
				
				ExecutorService innerEs = Executors.newFixedThreadPool((int) threads);
				List<Future<List<Double>>> futures = new ArrayList<Future<List<Double>>>();

				List<Entry<List<Integer>, List<Integer>>> innerCvList = SupervisedUtils.getKFoldCVList(10, 1, xArray.length, seed);
				for (int innerK = 0; innerK < innerCvList.size(); innerK++) {
					Entry<List<Integer>, List<Integer>> innerCvEntry = innerCvList.get(innerK);

					futures.add(innerEs.submit(new Callable<List<Double>>() {

						@Override
						public List<Double> call() throws Exception {
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

							DoubleMatrix cvTrainW = W.getRows(GWANN_RInterface.toIntArray(innerCvEntry.getKey())); // train to samples
							DoubleMatrix cvValW = W.get(GWANN_RInterface.toIntArray(innerCvEntry.getKey()), GWANN_RInterface.toIntArray(innerCvEntry.getValue())); // train to test
							DoubleMatrix kW = adaptive ? GWRUtils.getKernelWeights(cvTrainW, cvValW, kernel, (int) bw) : GWRUtils.getKernelWeights(cvValW, kernel, bw);

							List<Function> input = new ArrayList<>();
							while (input.size() < xArray[0].length)
								input.add(new Linear());
							input.add(new Constant(1.0));

							// Order of output is important because of consistent offset calculation
							List<Function> output = new ArrayList<>();
							while (output.size() < yVal.size())
								output.add(new Linear());

							List<Function> hidden = new ArrayList<>();
							while (hidden.size() < nrHidden)
								hidden.add(new TanH());
							hidden.add(new Constant(1.0));

							Function[][] layers = new Function[][] { input.toArray(new Function[] {}), hidden.toArray(new Function[] {}), output.toArray(new Function[] {}) };

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
							
								List<Double> responseVal = new ArrayList<>();
								for (int i = 0; i < xVal.size(); i++)
									responseVal.add(gwann.present(xVal.get(i))[i]);
								double valError = SupervisedUtils.getRMSE(responseVal, yVal, 0);
								errors.add(valError);

								if (valError < localBestValError) {
									localBestValError = valError;
									noImp = 0;
								} else
									noImp++;
								
								if( ( iterations >= 0 && it >= iterations ) || 
										( iterations < 0 && ( noImp >= patience || it >= uLim ) ) ) {
									uLim = Math.min(uLim, it);
									return errors;
								}
							}
						}
					}));
				}
				innerEs.shutdown();
				double[] mm = getMinMeanIdx(futures);

				if (mm[0] < bestValError) {
					bestValError = mm[0];
					bestValBw = bw;
					bestIts = (int)mm[1];
				}
				bwDone.add(bw);
			}
			if (prevBestValError - bestValError < eps)
				break;
			prevBestValError = bestValError;
		}

		System.out.println("Cross-validation results:");
		System.out.println("Bandwidth: " + bestValBw);
		System.out.println("Iterations: " + bestIts);
		System.out.println("RMSE: " + bestValError);

		System.out.println("Building final model...");
		{			
			Random r = new Random(seed);
			
			List<double[]> xTrain = new ArrayList<>();
			List<double[]> yTrain = new ArrayList<>();
			for (int i = 0; i < xArray.length; i++ ) {
				xTrain.add(Arrays.copyOf(xArray[i], xArray[i].length));

				double[] y = new double[dmP[0].length];
				for (int j = 0; j < y.length; j++)
					y[j] = yArray[i];
				yTrain.add(y);
			}
			new Normalizer(Normalizer.Transform.zScore, xTrain);

			List<Function> input = new ArrayList<>();
			while (input.size() < xArray[0].length)
				input.add(new Linear());
			input.add(new Constant(1.0));

			List<Function> hidden = new ArrayList<>();
			while (hidden.size() < nrHidden)
				hidden.add(new TanH());
			hidden.add(new Constant(1.0));

			List<Function> output = new ArrayList<>();
			while (output.size() < dmP[0].length)
				output.add(new Linear());

			Function[][] layers = new Function[][] { input.toArray(new Function[] {}), hidden.toArray(new Function[] {}), output.toArray(new Function[] {}) };

			double[][][] weights = NNet.getFullyConnectedWeights(layers, initMode.gorot_unif, 0);
			GWANN gwann = new GWANN(layers, weights, eta, opt);

			DoubleMatrix kW = adaptive ? GWRUtils.getKernelWeights(W, predW, kernel, (int) bestValBw) : GWRUtils.getKernelWeights(predW, kernel, bestValBw);
			List<Integer> batchReservoir = new ArrayList<>();
			for (int it = 0; it <= bestIts; it++) {
				
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
			}
			// predictions
			double[][] preds = new double[xTrain.size()][];
			for (int i = 0; i < xTrain.size(); i++) {
				double[][] out = gwann.presentInt(xTrain.get(i))[0];
				preds[i] = out[layers.length - 1];
			}

			ReturnObject ro = new ReturnObject();
			ro.predictions = preds;
			ro.weights = weights;
			ro.rmse = bestValError;
			ro.its = bestIts;
			ro.bw = bestValBw;

			return ro;
			
		}
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

	public static int[] toIntArray(Collection<Integer> c) {
		int[] j = new int[c.size()];
		int i = 0;
		for (int l : c)
			j[i++] = l;
		return j;
	}
}