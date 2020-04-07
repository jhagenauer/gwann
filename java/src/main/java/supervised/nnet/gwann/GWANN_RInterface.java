package supervised.nnet.gwann;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
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
import supervised.nnet.NNet.initMode;
import supervised.nnet.activation.Constant;
import supervised.nnet.activation.Function;
import supervised.nnet.activation.Linear;
import supervised.nnet.activation.Sigmoid;
import utils.DataUtils;
import utils.Normalizer;
import utils.GWRUtils.GWKernel;
import utils.Normalizer.Transform;

public class GWANN_RInterface {
	
	public static ReturnObject run(double[][] xArray, double[] yArray, double[][] dmX, double[][] dmP, double nrHidden, double batchSize, double eta, boolean linOut, String krnl, double bw_, boolean adaptive, double globalMaxIts, double maxNoImp, double trainPerIt, double threads) {

		assert xArray.length == dmX.length & dmX.length == dmX[0].length;
		
		GWKernel kernel;
		if( krnl.equalsIgnoreCase("gaussian"))
			kernel = GWKernel.gaussian;
		else if( krnl.equalsIgnoreCase("bisquare") )
			kernel = GWKernel.bisquare;
		else if( krnl.equalsIgnoreCase("boxcar"))
			kernel = GWKernel.boxcar;
		else
			throw new RuntimeException("Unknown kernel");
		
		double eps = 1e-04;

		List<Entry<List<Integer>, List<Integer>>> cvList = SupervisedUtils.getKFoldCVList(10, 1, xArray.length, 0);

		DoubleMatrix samplesW = new DoubleMatrix(dmX);

		List<Double> v = new ArrayList<>();
		for (double w : samplesW.data)
			v.add(w);
		v = new ArrayList<>(new HashSet<>(v));
		Collections.sort(v);

		double min = adaptive ? 5 : v.get(1) / 4;
		double max = adaptive ? samplesW.rows / 4 : v.get(v.size() - 1) / 4;

		double prevMin = Double.POSITIVE_INFINITY;
		double localMin = Double.POSITIVE_INFINITY;
		double localBestBw = adaptive ? (int) ((max - min) / 2) : (max - min) / 2;
		
		if( bw_ > 0 )
			localBestBw = bw_;
		else
			System.out.println("Searching best bandwidth... this takes considerable time!!!");
						
		int localBestIts = -1;
		Set<Double> bwDone = new HashSet<>();
		int noImpBw = 0;
		for (int c = 2;; c *= 2) {

			Set<Double> l = new HashSet<>();
			int steps = 10;
			for (int i = 1; i <= steps; i++) {
				double a = localBestBw + Math.pow((double) i / steps, 2) * Math.min(max - localBestBw, (max - min) / c);
				double b = localBestBw - Math.pow((double) i / steps, 2) * Math.min(localBestBw - min, (max - min) / c);	
				if (adaptive) {
					a = (double) Math.round(a);
					b = (double) Math.round(b);
				} 
				if (a <= max)
					l.add(a);
				if (b >= min)
					l.add(b);
			}
			l.add(localBestBw);

			List<Double> ll = new ArrayList<>(l);
			if( bw_ > 0 ) {
				ll.clear();
				ll.add(bw_);
			}
			ll.removeAll(bwDone);
			Collections.sort(ll);
			
			System.out.println(c + ", current bandwidth: " + localBestBw + ", RMSE:" + localMin + ", nr. bandwidths to test: " + ll.size());
											
			for (double bw : ll) {

				ExecutorService es = Executors.newFixedThreadPool((int)threads);
				List<Future<List<Double>>> futures = new ArrayList<Future<List<Double>>>();

				for (final Entry<List<Integer>, List<Integer>> cvEntry : cvList) {
					final int seed = cvList.indexOf(cvEntry);

					futures.add(es.submit(new Callable<List<Double>>() {
						@Override
						public List<Double> call() throws Exception {
							Random r = new Random(seed);

							List<double[]> samplesTrain = new ArrayList<>();
							for (int i : cvEntry.getKey()) {
								double[] d = Arrays.copyOf(xArray[i], xArray[i].length + 1);
								d[d.length - 1] = yArray[i];
								samplesTrain.add(d);
							}

							List<double[]> samplesTest = new ArrayList<>();
							for (int i : cvEntry.getValue()) {
								double[] d = Arrays.copyOf(xArray[i], xArray[i].length + 1);
								d[d.length - 1] = yArray[i];
								samplesTest.add(d);
							}

							int[] fa = new int[xArray[0].length];
							for (int i = 0; i < fa.length; i++)
								fa[i] = i;
							int ta = fa.length;

							Normalizer n = new Normalizer(Transform.zScore, samplesTrain, fa);
							n.normalize(samplesTest);

							DoubleMatrix cvTrainW = samplesW.getRows(GWANN.toIntArray(cvEntry.getKey())); // train to samples
							DoubleMatrix cvTestW = samplesW.get(GWANN.toIntArray(cvEntry.getKey()), GWANN.toIntArray(cvEntry.getValue())); // train to test
							Map<double[], double[]> sampleWeights = adaptive ? GWANN.getSampleWeights(samplesTrain, cvTrainW, cvTestW, kernel, (int) bw) : GWANN.getSampleWeights(samplesTrain, cvTestW, kernel, bw);

							List<Function> input = new ArrayList<>();
							while (input.size() < fa.length)
								input.add(new Linear());
							input.add(new Constant(1.0));

							// Order of output is important because of consistent offset calculation
							List<Function> output = new ArrayList<>();
							while (output.size() < samplesTest.size())
								output.add( linOut ? new Linear() : new Sigmoid() );

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
							GWANN gwann = new GWANN(layers, weights, eta);

							int[] tas = new int[output.size()];
							for (int i = 0; i < tas.length; i++)
								tas[i] = ta;

							List<double[]> batchReservoir = new ArrayList<>();
							List<Double> errors = new ArrayList<>();
							int noImp = 0;
							double bestError = Double.POSITIVE_INFINITY;
							for (int it = 0;; it++) {

								for (int i = 0; i < trainPerIt; i++) {
									List<double[]> batch = new ArrayList<>();
									while (batch.size() < batchSize) {
										if (batchReservoir.isEmpty())
											batchReservoir.addAll(samplesTrain);
										batch.add(batchReservoir.remove(r.nextInt(batchReservoir.size())));
									}
									gwann.trainSimple(batch, sampleWeights, fa, tas);
								}

								List<Double> responseTest = new ArrayList<>();
								for (int i = 0; i < samplesTest.size(); i++)
									responseTest.add(gwann.present(DataUtils.strip(samplesTest.get(i), fa))[i]);
								double testError = SupervisedUtils.getRMSE(responseTest, samplesTest, ta);
								errors.add(testError);

								if (testError < bestError) {
									bestError = testError;
									noImp = 0;
								} else
									noImp++;
										
								if (it >= globalMaxIts || noImp > maxNoImp)
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
						List<Double> lll = f.get();
						minResultLength = Math.min(minResultLength, lll.size());
						result.add(lll);
					} catch (InterruptedException | ExecutionException ex) {
						ex.printStackTrace();
					}

				double minMean = Double.POSITIVE_INFINITY;
				int minMeanIdx = -1;
				for (int i = 0; i < minResultLength; i++) { // for each it
					double mean = 0;
					for (int j = 0; j < result.size(); j++) // for each fold
						mean += result.get(j).get(i);
					mean /= result.size();
					if (mean < minMean) {
						minMean = mean;
						minMeanIdx = i;
					}
				}

				// local min
				if (minMean < localMin) {
					localMin = minMean;
					localBestBw = bw;
					localBestIts = minMeanIdx;
					noImpBw = 0;
				} else
					noImpBw++;

				bwDone.add(bw);
			}
			if( prevMin-localMin < eps && ( noImpBw >= 4 || ll.isEmpty() ) ) 
				break;
			prevMin = localMin;
		}
		System.out.println("Cross-validation results:");
		System.out.println("Bandwidth: "+localBestBw);
		System.out.println("Iterations: "+localBestIts);
		System.out.println("RMSE: "+localMin);
		
		System.out.println("Building final model...");
		{
			int maxIts = localBestIts;
			double bw = localBestBw;

			List<Integer> trainIdx = new ArrayList<>();
			List<double[]> samplesTrain = new ArrayList<>();
			for (int i = 0; i < xArray.length; i++) {
				double[] d = Arrays.copyOf(xArray[i], xArray[i].length + 1);
				d[d.length - 1] = yArray[i];
				samplesTrain.add(d);
				trainIdx.add(i);
			}

			int[] fa = new int[xArray[0].length];
			for (int i = 0; i < fa.length; i++)
				fa[i] = i;
			int ta = fa.length;

			new Normalizer(Transform.zScore, samplesTrain, fa);

			List<Function> input = new ArrayList<>();
			while (input.size() < fa.length)
				input.add(new Linear());
			input.add(new Constant(1.0));

			List<Function> hidden = new ArrayList<>();
			while (hidden.size() < nrHidden)
				hidden.add(new Sigmoid());
			hidden.add(new Constant(1.0));

			DoubleMatrix predW = new DoubleMatrix(dmP);

			List<Function> output = new ArrayList<>();
			while (output.size() < predW.columns)
				output.add( linOut ? new Linear() : new Sigmoid() );

			Function[][] layers = new Function[][] { input.toArray(new Function[] {}), hidden.toArray(new Function[] {}), output.toArray(new Function[] {}) };
			double[][][] weights = NNet.getFullyConnectedWeights(layers, initMode.gorot_unif, 0);

			int[] tas = new int[output.size()];
			for (int i = 0; i < tas.length; i++)
				tas[i] = ta;

			Random r = new Random(0);
			List<double[]> batchReservoir = new ArrayList<>();
			Map<double[], double[]> sampleWeights = adaptive ? GWANN.getSampleWeights(samplesTrain, samplesW, predW, kernel, (int) bw) : GWANN.getSampleWeights(samplesTrain, predW, kernel, bw);
			GWANN gwann = new GWANN(layers, weights, eta);
			for (int it = 0;; it++) {

				for (int i = 0; i < trainPerIt; i++) {
					List<double[]> batch = new ArrayList<>();
					while (batch.size() < batchSize) {
						if (batchReservoir.isEmpty())
							batchReservoir.addAll(samplesTrain);
						batch.add(batchReservoir.remove(r.nextInt(batchReservoir.size())));
					}
					gwann.trainSimple(batch, sampleWeights, fa, tas);
				}

				// return weights and predictions
				if (it >= maxIts) {
					
					// predictions
					double[][] preds = new double[samplesTrain.size()][];
					for (int i = 0; i < samplesTrain.size(); i++) {
						double[] d2 = samplesTrain.get(i);
						double[][] out = gwann.presentInt(DataUtils.strip(d2, fa))[0];
						preds[i] = out[layers.length - 1];												
					}
					
					ReturnObject ro = new ReturnObject();
					ro.predictions = preds;
					ro.weights = weights[weights.length-1];
					ro.rmse = localMin;
					ro.its = it;
					ro.bw = bw;
					
					return ro;
				}
			}

		}
	}
}