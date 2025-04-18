package supervised.nnet.gwann;

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

import org.jblas.DoubleMatrix;

import supervised.SupervisedUtils;
import supervised.nnet.NNet.Optimizer;
import supervised.nnet.NNetUtils;
import supervised.nnet.ReturnObject;
import supervised.nnet.activation.Constant;
import supervised.nnet.activation.Function;
import supervised.nnet.activation.Linear;
import supervised.nnet.activation.Logistic;
import supervised.nnet.activation.TanH;
import utils.DataUtils;
import utils.GWUtils;
import utils.GWUtils.GWKernel;
import utils.ListNormalizer;
import utils.Normalizer.Transform;

public class GWANNUtils {
	
	public static boolean logistic = false;
			
	// cleaner interface
	public static ReturnObject buildGWANN( 
			List<double[]> x_train_, List<double[]> y_train_, 
			List<double[]> x_test_, List<double[]> y_test_, // only needed for error calculation etc
			DoubleMatrix W_train_test, 
			int[] nrHidden, double[] eta, Optimizer opt, 
			int batchSize, int maxIt, int patience,  
			GWKernel kernel, double bw, boolean adaptive, 
			double lambda,
			Transform[] expTrans, Transform[] respTrans, int seed ) {
		
		Random r = new Random(seed);		
		double[][] kW = adaptive ? GWUtils.getKernelWeightsAdaptive(W_train_test, kernel, (int) bw).toArray2() : GWUtils.getKernelWeights(W_train_test, kernel, bw).toArray2();
			
		List<double[]> x_train = new ArrayList<>();
		for( double[] d : x_train_ )
			x_train.add( Arrays.copyOf(d, d.length) );
		
		List<double[]> y_train = new ArrayList<>();
		for( double[] d : y_train_ ) {
			assert d.length == 1;
			double[] nd = new double[y_test_.size()];
			for( int i = 0; i < nd.length; i++ )
				nd[i] = d[0];
			y_train.add(nd);
		}
		
		List<double[]> x_test = new ArrayList<>();
		for( double[] d : x_test_ )
			x_test.add( Arrays.copyOf(d, d.length) );
				
		List<double[]> y_test = new ArrayList<double[]>();
		for( int i = 0; i < y_test_.size(); i++ ) {
			double[] d = y_test_.get(i);
			assert d.length == 1;
			double[] nd = new double[y_test_.size()];
			for( int j = 0; j < nd.length; j++ )
				nd[j] = d[0];
			y_test.add( nd );
		}
				
		double[] desired_orig_diag = new double[y_test_.size()];
		for( int i = 0; i < y_test.size(); i++ ) {
			double d = y_test.get(i)[i];
			desired_orig_diag[i] = d;
		}	
		
		ListNormalizer ln_x = new ListNormalizer( expTrans, x_train);
		ln_x.normalize(x_test);
		
		ListNormalizer ln_y = new ListNormalizer( respTrans, y_train);										
		ln_y.normalize(y_test);
				
		List<Function[]> layerList = new ArrayList<>();
		List<Function> input = new ArrayList<>();
		while (input.size() < x_train.get(0).length )
			input.add(new Linear());
		input.add(new Constant(1.0));
		layerList.add(input.toArray(new Function[] {} ) );
		
		for( int nh : nrHidden ) {
			if( nh == 0 )
				continue;
			List<Function> hidden0 = new ArrayList<>();
			while (hidden0.size() < nh) {
				if( logistic )
					hidden0.add(new Logistic());
				else
					hidden0.add(new TanH());				
			}
			hidden0.add(new Constant(1.0));
			layerList.add(hidden0.toArray(new Function[] {} ) );
		}
				
		List<Function> output = new ArrayList<>();
		while (output.size() < y_test_.size())
			output.add(new Linear());
		layerList.add(output.toArray(new Function[] {} ) );	
						
		Function[][] layers = layerList.toArray( new Function[][] {} );
		double[][][] weights = NNetUtils.getFullyConnectedWeights(layers, NNetUtils.initMode.gorot_unif, seed);						
		GWANN gwann = new GWANN(layers, weights, eta, opt,lambda);
						
		List<Integer> batchReservoir = new ArrayList<>();			
		List<Double> errors = new ArrayList<>();	
		int no_imp = 0;
		double test_error_best = Double.POSITIVE_INFINITY;
		for (int it = 0; it < maxIt && no_imp < patience; it++) {
			
			List<double[]> x = new ArrayList<>();
			List<double[]> y = new ArrayList<>();
			List<double[]> gwWeights = new ArrayList<>();
			
			if( batchSize < 0 ) {
				for( int k : batchReservoir ) {
					x.add(x_train.get(k));
					y.add(y_train.get(k));	
					gwWeights.add(kW[k]); 
				}
			} else {						
				while (x.size() < batchSize) {
					if (batchReservoir.isEmpty())
						for (int j = 0; j < x_train.size(); j++)
							batchReservoir.add(j);
					int k = batchReservoir.remove(r.nextInt(batchReservoir.size()));
					x.add(x_train.get(k));
					y.add(y_train.get(k));	
					gwWeights.add(kW[k]); 
				}
			}
			gwann.train(x, y, gwWeights);
											
			// get denormalized response-diag 
			double[] response_diag_denormalized= new double[x_test.size()];
			for (int i = 0; i < x_test.size(); i++) {
				double[] d = gwann.present(x_test.get(i));
				ln_y.denormalize(d, i);
				response_diag_denormalized[i] = d[i]; // only diagonal
			}		
			
			double test_error = SupervisedUtils.getRMSE(response_diag_denormalized, desired_orig_diag);
			errors.add(test_error);
			
			if (test_error < test_error_best) {
				test_error_best = test_error;
				no_imp = 0;
			} else
				no_imp++;			
		}
							
		// get full response and denormalize
		List<double[]> response_denormalized= new ArrayList<>();
		for (int i = 0; i < x_test.size(); i++)
			response_denormalized.add(gwann.present(x_test.get(i)));
		ln_y.denormalize(response_denormalized);
		
		// response_diag
		double[] response_denormalized_diag = new double[response_denormalized.size()];
		for (int i = 0; i < x_test.size(); i++)
			response_denormalized_diag[i] = response_denormalized.get(i)[i];
														
		ReturnObject ro = new ReturnObject();
		ro.errors = errors; 
		ro.rmse = SupervisedUtils.getRMSE(response_denormalized_diag, desired_orig_diag);
		ro.r2 = SupervisedUtils.getR2(response_denormalized_diag, desired_orig_diag );
		ro.nnet = gwann;
		ro.prediction = response_denormalized;
		ro.ln_x = ln_x;
		ro.ln_y = ln_y;
		return ro;
	}
		
	public static double[] getParamsWithGoldenSection(double minRadius, double maxRadius, 
			List<double[]> xArray, List<double[]> yArray, DoubleMatrix W, List<Entry<List<Integer>, List<Integer>>> innerCvList, 
			GWKernel kernel, boolean adaptive, double[] eta, int batchSize, Optimizer opt, int[] nrHidden, int iterations, int patience, int threads, double a, Transform[] explTrans, Transform[] respTrans, int seed ) {
		double xU = maxRadius;
		double xL = minRadius;
		double eps = 1e-04;
		double R = (Math.sqrt(5) - 1) / 2.0;
		double d = R * (xU - xL);
		
		double x1 = xL + d;
		double x2 = xU - d;
		double[] f1 = NNetUtils.getBestErrorParams( getErrors_CV(xArray, yArray, W, innerCvList, kernel, x1, adaptive, eta, batchSize, opt, nrHidden, iterations, patience, threads, a, explTrans, respTrans, seed, true) );
		double[] f2 = NNetUtils.getBestErrorParams( getErrors_CV(xArray, yArray, W, innerCvList, kernel, x2, adaptive, eta, batchSize, opt, nrHidden, iterations, patience, threads, a, explTrans, respTrans, seed, true) );
		
		double d1 = f2[0] - f1[0];
		
		while ((Math.abs(d) > eps) && (Math.abs(d1) > eps)) {
			d = R * d;
			if (f1[0] < f2[0]) {
				xL = x2;
				x2 = x1;
				x1 = xL + d;
				f2 = f1;
				f1 = NNetUtils.getBestErrorParams( getErrors_CV(xArray, yArray, W, innerCvList, kernel, x1, adaptive, eta, batchSize, opt, nrHidden, iterations, patience, threads, a, explTrans, respTrans, seed, true ) );
			} else {
				xU = x1;
				x1 = x2;
				x2 = xU - d;
				f1 = f2;
				f2 = NNetUtils.getBestErrorParams( getErrors_CV(xArray, yArray, W, innerCvList, kernel, x2, adaptive, eta, batchSize, opt, nrHidden, iterations, patience, threads, a, explTrans, respTrans, seed, true ) );
			}
			d1 = f2[0] - f1[0];
		}
		// returns cost, bandwidth, iterations
		if( f1[0] < f2[0] )
			return new double[] { f1[0], adaptive ? Math.round(x1) : x1, f1[1]};
		else 
			return new double[] { f2[0], adaptive ? Math.round(x2) : x2, f2[1]};
	}
			
	public static List<List<Double>> getErrors_CV(
			List<double[]> xArray, List<double[]> yArray, DoubleMatrix W, List<Entry<List<Integer>, List<Integer>>> innerCvList, 
			GWKernel kernel, double bw, boolean adaptive, double[] eta, int batchSize, Optimizer opt, int[] nrHidden, int iterations, int patience, 
			int threads, double lambda, Transform[] explTrans, Transform[] respTrans, int seed ) {
		return getErrors_CV(xArray,yArray,W,innerCvList,kernel,bw,adaptive,eta,batchSize,opt,nrHidden,iterations,patience,threads,lambda, explTrans, respTrans, seed, false);		
	}
	
	private static int max_its;
	public static List<List<Double>> getErrors_CV(
			List<double[]> xArray, List<double[]> yArray, DoubleMatrix W, List<Entry<List<Integer>, List<Integer>>> innerCvList, 
			GWKernel kernel, double bw, boolean adaptive, double[] eta, int batchSize, Optimizer opt, int[] nrHidden, int iterations, int patience, 
			int threads, double lambda, Transform[] explTrans, Transform[] respTrans, int seed, boolean fast_cv ) {
						
		ExecutorService innerEs = Executors.newFixedThreadPool((int) threads);
		List<Future<List<Double>>> futures = new ArrayList<Future<List<Double>>>();		
			
		max_its = iterations;
		for ( Entry<List<Integer>, List<Integer>> innerCvEntry : innerCvList ) {
			futures.add(innerEs.submit(new Callable<List<Double>>() {
				@Override
				public List<Double> call() throws Exception {	
					List<Integer> trainIdx = innerCvEntry.getKey();
					List<Integer> testIdx = innerCvEntry.getValue();
					
					List<double[]> xTrain = new ArrayList<>();
					List<double[]> yTrain = new ArrayList<>();
					for (int i = 0; i < trainIdx.size(); i++ ) {
						int idx = trainIdx.get(i);
						xTrain.add(xArray.get(idx));
						yTrain.add(yArray.get(idx));
					}
					
					List<double[]> xTest = new ArrayList<>();
					List<double[]> yTest = new ArrayList<>();
					for (int i = 0; i < testIdx.size(); i++ ) {
						int idx = testIdx.get(i);
						xTest.add(xArray.get(idx));
						yTest.add(yArray.get(idx));
					}
								
					int[] trainIdxA = DataUtils.toIntArray(trainIdx);
					DoubleMatrix W_train_test = W.get( trainIdxA, DataUtils.toIntArray(testIdx)); // train to test
										
					if( fast_cv ) {
						List<Double> errors = buildGWANN(xTrain, yTrain, xTest, yTest, W_train_test, nrHidden, eta, opt, batchSize, max_its, patience, kernel, bw, adaptive, lambda, explTrans, respTrans, seed).errors;
						max_its = Math.min(max_its,errors.size());
						return errors;	
					} else {
						List<Double> errors = buildGWANN(xTrain, yTrain, xTest, yTest, W_train_test, nrHidden, eta, opt, batchSize, iterations, patience, kernel, bw, adaptive, lambda, explTrans, respTrans, seed).errors;
						return errors;	
					}
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
//		
//	public static class GWANN_CV {
//		
//		GWANN gwann;
//		double[][] kW; 
//		
//		Random r;
//		int batchSize;
//		List<Integer> batchReservoir = new ArrayList<>();
//		
//		List<double[]> x_train, y_train;
//		List<double[]> x_test, y_test;
//		
//		ListNormalizer lnYTrain;
//		double[] desired_orig_diag;
//		int idx = 0;
//		
//		public GWANN_CV( List<double[]> xArray, List<Double> yArray, DoubleMatrix W, List<Integer> trainIdx, List<Integer> testIdx, 
//				GWKernel kernel, double bw, boolean adaptive, double[] eta, int batchSize, Optimizer opt, int[] nrHidden, 
//				double lambda, Transform[] explTrans, Transform[] respTrans, int seed ) {
//				
//			this.r = new Random(seed);
//			this.batchSize = batchSize;
//			
//			// extract train and test sets
//			x_train = new ArrayList<>();
//			y_train = new ArrayList<>();
//			for (int i = 0; i < trainIdx.size(); i++ ) {
//				int idx = trainIdx.get(i);
//				x_train.add(Arrays.copyOf(xArray.get(idx), xArray.get(idx).length));
//				
//				double y = yArray.get(idx);
//				double[] nd = new double[testIdx.size()];
//				for( int j = 0; j < nd.length; j++ )
//					nd[j] = y;
//				y_train.add(nd);
//			}
//				
//			x_test = new ArrayList<>();
//			y_test = new ArrayList<>();
//			for (int i = 0; i < testIdx.size(); i++ ) {
//				int idx = testIdx.get(i);
//				x_test.add(Arrays.copyOf(xArray.get(idx), xArray.get(idx).length));
//				
//				double y = yArray.get(idx);
//				double[] nd = new double[testIdx.size()];
//				for( int j = 0; j < nd.length; j++ )
//					nd[j] = y;
//				y_test.add( nd );
//			}
//							
//			int[] trainIdxA = DataUtils.toIntArray(trainIdx);
//			DoubleMatrix W_train_test = W.get( trainIdxA, DataUtils.toIntArray(testIdx)); // train to test
//			DoubleMatrix W_train_train = W.get(trainIdxA,trainIdxA);
//							
//			kW = adaptive ? GWUtils.getKernelWeightsAdaptive(W_train_test, kernel, (int) bw).toArray2() : GWUtils.getKernelWeights(W_train_test, kernel, bw).toArray2();
//								
//			desired_orig_diag = new double[testIdx.size()];
//			for( int i = 0; i < y_test.size(); i++ ) {
//				double d = y_test.get(i)[i];
//				desired_orig_diag[i] = d;
//			}	
//				
//			ListNormalizer lnXTrain = new ListNormalizer( explTrans, x_train);
//			lnXTrain.normalize(x_test);
//				
//			lnYTrain = new ListNormalizer( respTrans, y_train);										
//			lnYTrain.normalize(y_test);
//								
//			List<Function[]> layerList = new ArrayList<>();
//			List<Function> input = new ArrayList<>();
//			while (input.size() < x_train.get(0).length )
//				input.add(new Linear());
//			input.add(new Constant(1.0));
//			layerList.add(input.toArray(new Function[] {} ) );
//				
//			for( int nh : nrHidden ) {
//				if( nh == 0 )
//					continue;
//				List<Function> hidden0 = new ArrayList<>();
//				while (hidden0.size() < nh) {
//					if( logistic )
//						hidden0.add(new Logistic());
//					else
//						hidden0.add(new TanH());				
//				}
//				hidden0.add(new Constant(1.0));
//				layerList.add(hidden0.toArray(new Function[] {} ) );
//			}
//						
//			List<Function> output = new ArrayList<>();
//			while (output.size() < testIdx.size())
//				output.add(new Linear());
//			layerList.add(output.toArray(new Function[] {} ) );	
//								
//			Function[][] layers = layerList.toArray( new Function[][] {} );
//			double[][][] weights = NNetUtils.getFullyConnectedWeights(layers, NNetUtils.initMode.gorot_unif, seed);						
//			gwann = new GWANN(layers, weights, eta, opt,lambda);
//			
//			batchReservoir = new ArrayList<>();
//			for (int k = 0; k < x_train.size(); k++)
//				batchReservoir.add(k);							
//		}
//		
//		public double train() {
//			List<double[]> x = new ArrayList<>();
//			List<double[]> y = new ArrayList<>();
//			List<double[]> gwWeights = new ArrayList<>();
//			
//			if( batchSize < 0 ) {
//				for( int k : batchReservoir ) {
//					x.add(x_train.get(k));
//					y.add(y_train.get(k));	
//					gwWeights.add(kW[k]); 
//				}
//			} else {						
//				while (x.size() < batchSize) {
//					
//					if( idx == batchReservoir.size() ) {
//						Collections.shuffle(batchReservoir,r);
//						idx = 0;
//					}
//					
//					int k = batchReservoir.get(idx++);
//					x.add(x_train.get(k));
//					y.add(y_train.get(k));	
//					gwWeights.add(kW[k]); 
//				}
//			}
//			gwann.train(x, y, gwWeights);
//											
//			// get denormalized response-diag 
//			double[] response_diag_denormalized= new double[x_test.size()];
//			for (int i = 0; i < x_test.size(); i++) {
//				double[] d = gwann.present(x_test.get(i));
//				lnYTrain.denormalize(d, i); // only diagonal
//				response_diag_denormalized[i] = d[i]; // only diagonal
//			}						
//			return SupervisedUtils.getRMSE(response_diag_denormalized, desired_orig_diag);
//		}
//	}
//		
//	public static double[] getErrors_CV_shortcut_2(
//			List<double[]> samplesA, DoubleMatrix W, List<Entry<List<Integer>, List<Integer>>> innerCvList, int[] fa, int ta,
//			GWKernel kernel, double bw, boolean adaptive, double[] eta, int batchSize, Optimizer opt, int[] nrHidden, int iterations, int patience, 
//			double a, Transform[] explTrans, Transform[] respTrans, int seed ) {
//		
//		// strip from full samples
//		List<double[]> xArray = new ArrayList<>();
//		List<Double> yArray = new ArrayList<>();
//		for (double[] d : samplesA ) {
//			xArray.add(DataUtils.strip(d, fa));
//			yArray.add(d[ta]);
//		}
//					
//		List<GWANN_CV> ml = new ArrayList<>();
//		List<List<Double>> e = new ArrayList<>();
//		int[] min_idx = new int[innerCvList.size()];
//		for ( Entry<List<Integer>, List<Integer>> innerCvEntry : innerCvList ) {
//			List<Integer> trainIdx = innerCvEntry.getKey();
//			List<Integer> testIdx = innerCvEntry.getValue();			
//			ml.add( new GWANN_CV(xArray,yArray,W,trainIdx,testIdx,kernel,bw,adaptive,eta,batchSize,opt,nrHidden,a,explTrans,respTrans,seed ) );
//			e.add( new ArrayList<>());
//		}		
//		
//		for( int i = 0; i < iterations; i++ ) {
//			for( int j = 0; j < ml.size(); j++ ) {
//				double error = ml.get(j).train();
//				e.get(j).add( error );	
//				
//				if( error < e.get(j).get(min_idx[j]) )
//					min_idx[j] = i;
//				else if( i - min_idx[j] >= patience ) 					
//					return NNetUtils.getBestErrorParams( e );				
//			}			
//		}
//		return NNetUtils.getBestErrorParams( e );
//	}
						
	public static double[] getParamsWithGridSearch(int minRadius, int maxRadius, int steps,
			List<double[]> xArray, List<double[]> yArray, DoubleMatrix W, List<Entry<List<Integer>, List<Integer>>> innerCvList, int[] fa, int ta, GWKernel kernel, boolean adaptive, double[] eta, int batchSize, 
			Optimizer opt,  
			int[] nrHidden, int iterations, int patience, int threads, double a, Transform[] expTrans, Transform[] respTrans, int seed ) {
		
		if( !adaptive )
			throw new RuntimeException("Not implemented yet");
				
		double[] bestF = null;
		int bestBw = -1;
		for( int i = (int)minRadius; i <= maxRadius; i+=(maxRadius-minRadius)/steps ) {
			double[] f = NNetUtils.getBestErrorParams( getErrors_CV(xArray, yArray, W, innerCvList, kernel, i, adaptive, eta, batchSize, opt, nrHidden, iterations, patience, threads, a, expTrans, respTrans, seed) );
			if( bestF == null || f[0] < bestF[0] ) { 
				bestF = f;
				bestBw = i;
			}
		}
		return new double[] { bestF[0], bestBw, bestF[1]};
	}
	
	// shortcuts
	public static ReturnObject buildGWANN( 
			List<double[]> samples, DoubleMatrix W, List<Integer> trainIdx, List<Integer> testIdx, 
			int[] nrHidden, int[] ga, int[] fa, int ta, double[] eta, Optimizer opt,  
			int batchSize, int maxIt, int patience, GWKernel kernel, double bw, boolean adaptive, double a, Transform[] expTrans, Transform[] respTrans, int seed ) {
					
		List<double[]> xTrain = new ArrayList<>();
		List<double[]> yTrain = new ArrayList<>();
		for (int i : trainIdx) {
			double[] d = samples.get(i);
			xTrain.add(DataUtils.strip(d, fa));
			yTrain.add( new double[] { d[ta] } );
		}
	
		List<double[]> xVal = new ArrayList<>();
		List<double[]> yVal = new ArrayList<>();
		for (int i : testIdx) {
			double[] d = samples.get(i);
			xVal.add(DataUtils.strip(d, fa));
			yVal.add( new double[] { d[ta] });
		}
			
		DoubleMatrix W_train_val = W.get( DataUtils.toIntArray(trainIdx), DataUtils.toIntArray(testIdx));		
		return buildGWANN(xTrain, yTrain, xVal, yVal, W_train_val, nrHidden, eta, opt, batchSize, maxIt, patience, kernel, bw, adaptive, a, expTrans, respTrans, seed);
	}
	
	public static double[] getParamsWithGoldenSection(double minRadius, double maxRadius, 
			List<double[]> samples, DoubleMatrix W, List<Entry<List<Integer>, List<Integer>>> innerCvList, int[] fa, int ta, GWKernel kernel, boolean adaptive, double[] eta, int batchSize, 
			Optimizer opt,  
			int[] nrHidden, int iterations, int patience, int threads, double a, Transform[] expTrans, Transform[] respTrans, int seed ) {
		
		List<double[]> xTrain = new ArrayList<>();
		List<double[]> yTrain = new ArrayList<>();
		for (double[] d : samples ) {
			xTrain.add(DataUtils.strip(d, fa));
			yTrain.add( new double[] { d[ta] } );
		}
		return getParamsWithGoldenSection(minRadius,maxRadius,xTrain, yTrain, W, innerCvList, kernel, adaptive, eta, batchSize, opt, nrHidden, iterations, patience, threads, a, expTrans, respTrans, seed);
	}
	
	public static double[] getParamsWithGridSearch(int minRadius, int maxRadius, int steps, 
			List<double[]> samples, DoubleMatrix W, List<Entry<List<Integer>, List<Integer>>> innerCvList, int[] fa, int ta, GWKernel kernel, boolean adaptive, double[] eta, int batchSize, 
			Optimizer opt, 
			int[] nrHidden, int iterations, int patience, int threads, double a, Transform[] expTrans, Transform[] respTrans, int seed ) {
		
		List<double[]> xTrain = new ArrayList<>();
		List<double[]> yTrain = new ArrayList<>();
		for (double[] d : samples ) {
			xTrain.add(DataUtils.strip(d, fa));
			yTrain.add( new double[] { d[ta] } );
		}
		return getParamsWithGridSearch(minRadius, maxRadius, steps, xTrain, yTrain, W, innerCvList, fa, ta, kernel, adaptive, eta, batchSize, opt, nrHidden, iterations, patience, threads, a, expTrans, respTrans, seed);
	}
}
