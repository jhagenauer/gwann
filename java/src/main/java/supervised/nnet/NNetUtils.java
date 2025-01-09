package supervised.nnet;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
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
import supervised.nnet.NNet.Optimizer;
import supervised.nnet.activation.Constant;
import supervised.nnet.activation.Function;
import supervised.nnet.activation.Linear;
import supervised.nnet.activation.Logistic;
import supervised.nnet.activation.TanH;
import utils.DataUtils;
import utils.ListNormalizer;
import utils.Normalizer.Transform;

public class NNetUtils {
	
	private static Logger log = LogManager.getLogger(NNetUtils.class);

	public static enum initMode {
		gorot_unif, norm05
	}

	public static List<List<Double>> getErrors_CV(List<double[]> xArray, List<Double> yArray, List<Entry<List<Integer>, List<Integer>>> innerCvList, double[] eta, int batchSize, NNet.Optimizer opt, double lambda, int[] nrHidden, int maxIt, int patience, int threads, Transform[] expTrans, Transform[] respTrans ) {
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
					List<Double> yTrain = new ArrayList<>();
					for (int i : trainIdx) {
						xTrain.add(xArray.get(i));
						yTrain.add(yArray.get(i));
					}
				
					List<double[]> xVal = new ArrayList<>();
					List<Double> yVal = new ArrayList<>();
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
			List<double[]> xTrain_, List<Double> yTrain_, List<double[]> xVal_, List<Double> yVal_, 
			int[] nrHidden, double[] eta, NNet.Optimizer opt, 
			double lambda, int batchSize, int maxIt, int patience, Transform[] expTrans, Transform[] respTrans
			) {
		Random r = new Random(0);
	
		List<double[]> xTrain = new ArrayList<>();
		for (double[] d : xTrain_)
			xTrain.add(Arrays.copyOf(d, d.length));
	
		List<double[]> yTrain = new ArrayList<>();
		for (double d : yTrain_) {
			double[] nd = new double[] {d};
			yTrain.add(nd);
		}
	
		List<double[]> xVal = new ArrayList<>();
		for (double[] d : xVal_)
			xVal.add(Arrays.copyOf(d, d.length));
	
		double[] des = new double[yVal_.size()];
		List<double[]> yVal = new ArrayList<>();
		for( int i = 0; i < yVal_.size(); i++ ) {
			double d = yVal_.get(i);
			des[i] = yVal_.get(i);

			yVal.add( new double[] {d});
		}
			
		ListNormalizer ln_x = new ListNormalizer(expTrans, xTrain);
		ln_x.normalize(xVal);
		
		ListNormalizer ln_y = new ListNormalizer(respTrans, yTrain);	
		ln_y.normalize(yVal);
	
		List<Function[]> layerList = new ArrayList<>();
		List<Function> input = new ArrayList<>();
		while (input.size() < xTrain.get(0).length)
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
		while (output.size() < yVal.get(0).length )
			output.add(new Linear());
		layerList.add(output.toArray(new Function[] {}));
	
		Function[][] layers = layerList.toArray(new Function[][] {});
		double[][][] weights = NNetUtils.getFullyConnectedWeights(layers, NNetUtils.initMode.gorot_unif, 0);
	
		NNet nnet = new NNet(layers, weights, eta, opt, lambda);
	
		List<Integer> batchReservoir = new ArrayList<>();
		List<Double> errors = new ArrayList<>();
		int noImp = 0;
		double localBestValError = Double.POSITIVE_INFINITY;
	
		for (int it = 0; it < maxIt && noImp < patience; it++) {
	
			if( batchSize > 0 ) {
				List<double[]> x = new ArrayList<>();
				List<double[]> y = new ArrayList<>();
				while (x.size() < batchSize) {
					if (batchReservoir.isEmpty())
						for (int j = 0; j < xTrain.size(); j++)
							batchReservoir.add(j);
					int idx = batchReservoir.remove(r.nextInt(batchReservoir.size()));
					x.add(xTrain.get(idx));
					y.add(yTrain.get(idx));
				}
				nnet.train(x, y);
			} else 
				nnet.train(xTrain, yTrain);
						
			// get denormalized response
			double[] res= new double[xVal.size()];
			for (int i = 0; i < xVal.size(); i++) {
				double[] d = nnet.present(xVal.get(i));
				ln_y.denormalize(d, 0);
				res[i] = d[0];
			}
	
			double valError = SupervisedUtils.getRMSE(res, des);
			errors.add(valError);
	
			if (valError < localBestValError) {
				localBestValError = valError;
				noImp = 0;
			} else
				noImp++;
		}
	
		// get response and denormalize
		List<double[]> response= new ArrayList<>();
		for (int i = 0; i < xVal.size(); i++)
			response.add(nnet.present(xVal.get(i)));
		ln_y.denormalize(response);
				
		// response, only first
		double[] res = new double[response.size()];
		for (int i = 0; i < xVal.size(); i++)
			res[i] = response.get(i)[0];
												
		ReturnObject ro = new ReturnObject();
		ro.errors = errors;
		ro.rmse = SupervisedUtils.getRMSE(res, des);
		ro.r2 = SupervisedUtils.getR2(res, des );
		ro.nnet = nnet;
		ro.prediction = response;	
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
	
	public static class NNET_CV {
		
		NNet nnet;
		
		Random r;
		int batchSize;
		List<Integer> batchReservoir = new ArrayList<>();
		
		List<double[]> x_train, x_test;
		List<double[]> y_train, y_test;
		
		ListNormalizer lnYTrain;
		double[] desired_orig;
		int idx = 0;
		boolean logistic = false;
		
		public NNET_CV( List<double[]> xArray, List<Double> yArray, List<Integer> trainIdx, List<Integer> testIdx, 
				double[] eta, int batchSize, Optimizer opt, int[] nrHidden, 
				double lambda, Transform[] explTrans, Transform[] respTrans, int seed ) {
				
			this.r = new Random(seed);
			this.batchSize = batchSize;
			
			// extract train and test sets
			x_train = new ArrayList<>();
			y_train = new ArrayList<>();
			for (int i = 0; i < trainIdx.size(); i++ ) {
				int idx = trainIdx.get(i);
				x_train.add(Arrays.copyOf(xArray.get(idx), xArray.get(idx).length));				
				y_train.add( new double[] { yArray.get(idx) } );
			}
				
			x_test = new ArrayList<>();
			y_test = new ArrayList<>();
			for (int i = 0; i < testIdx.size(); i++ ) {
				int idx = testIdx.get(i);
				x_test.add(Arrays.copyOf(xArray.get(idx), xArray.get(idx).length));
				y_test.add( new double[] { yArray.get(idx) } );
			}
															
			desired_orig = new double[testIdx.size()];
			for( int i = 0; i < y_test.size(); i++ )
				desired_orig[i] = y_test.get(i)[0];
							
			ListNormalizer lnXTrain = new ListNormalizer( explTrans, x_train);
			lnXTrain.normalize(x_test);
				
			lnYTrain = new ListNormalizer( respTrans, y_train);										
			lnYTrain.normalize(y_test);
								
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
			while (output.size() < y_test.get(0).length)
				output.add(new Linear());
			layerList.add(output.toArray(new Function[] {} ) );	
								
			Function[][] layers = layerList.toArray( new Function[][] {} );
			double[][][] weights = NNetUtils.getFullyConnectedWeights(layers, NNetUtils.initMode.gorot_unif, seed);						
			nnet = new NNet(layers, weights, eta, opt, lambda);
			
			batchReservoir = new ArrayList<>();
			for (int k = 0; k < x_train.size(); k++)
				batchReservoir.add(k);							
		}
		
		public double train() {
			List<double[]> x = new ArrayList<>();
			List<double[]> y = new ArrayList<>();
						
			if( batchSize > 0 ) {
				while (x.size() < batchSize) {
					
					if( idx == batchReservoir.size() ) {
						Collections.shuffle(batchReservoir,r);
						idx = 0;
					}
					
					int k = batchReservoir.get(idx++);
					x.add(x_train.get(k));
					y.add(y_train.get(k));	
				}
				nnet.train(x, y);
			} else 
				nnet.train(x_train, y_train);
													
			// get denormalized response
			double[] response_denormalized= new double[x_test.size()];
			for (int i = 0; i < x_test.size(); i++) {
				double[] d = nnet.present(x_test.get(i));
				lnYTrain.denormalize(d, 0);
				response_denormalized[i] = d[0]; 
			}						
			return SupervisedUtils.getRMSE(response_denormalized, desired_orig);
		}
	}
		
	public static double[] getErrors_CV_shortcut_2(
			List<double[]> samplesA, List<Entry<List<Integer>, List<Integer>>> innerCvList, int[] fa, int ta,
			double[] eta, int batchSize, Optimizer opt, int[] nrHidden, int iterations, int patience, 
			double a, Transform[] explTrans, Transform[] respTrans, int seed ) {
		
		// strip from full samples
		List<double[]> xArray = new ArrayList<>();
		List<Double> yArray = new ArrayList<>();
		for (double[] d : samplesA ) {
			xArray.add(DataUtils.strip(d, fa));
			yArray.add(d[ta]);
		}
					
		List<NNET_CV> ml = new ArrayList<>();
		List<List<Double>> e = new ArrayList<>();
		int[] min_idx = new int[innerCvList.size()];
		for ( Entry<List<Integer>, List<Integer>> innerCvEntry : innerCvList ) {
			List<Integer> trainIdx = innerCvEntry.getKey();
			List<Integer> testIdx = innerCvEntry.getValue();			
			ml.add( new NNET_CV(xArray,yArray,trainIdx,testIdx,eta,batchSize,opt,nrHidden,a,explTrans,respTrans,seed ) );
			e.add( new ArrayList<>());
		}		
		
		for( int i = 0; i < iterations; i++ ) {
			for( int j = 0; j < ml.size(); j++ ) {
				double error = ml.get(j).train();
				e.get(j).add( error );	
				
				if( error < e.get(j).get(min_idx[j]) )
					min_idx[j] = i;
				else if( i - min_idx[j] >= patience ) 					
					return NNetUtils.getBestErrorParams( e );				
			}			
		}
		return NNetUtils.getBestErrorParams( e );
	}
}
