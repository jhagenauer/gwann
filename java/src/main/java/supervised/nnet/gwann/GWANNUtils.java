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

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
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
	
	private static Logger log = LogManager.getLogger(GWANNUtils.class);
	
	public static boolean logistic = false;	
			
	// cleaner interface
	public static ReturnObject buildGWANN( 
			List<double[]> xTrain_, List<Double> yTrain_, DoubleMatrix W_train, 
			List<double[]> xVal_, List<Double> yVal_, DoubleMatrix W_train_val, 
			int[] nrHidden, double[] eta, Optimizer opt, double lambda,
			int batchSize, int maxIt, int patience,  
			GWKernel kernel, double bw, boolean adaptive, 
			double[][][] baseWeights, double a,
			Transform[] expTrans, Transform[] respTrans, int seed ) {
		
		Random r = new Random(seed);		
		DoubleMatrix kW = adaptive ? GWUtils.getKernelWeights(W_train, W_train_val, kernel, (int) bw) : GWUtils.getKernelWeights(W_train_val, kernel, bw);
		
		List<double[]> xTrain = new ArrayList<>();
		for( double[] d : xTrain_ )
			xTrain.add( Arrays.copyOf(d, d.length) );
		
		List<double[]> yTrain = new ArrayList<>();
		for( double d : yTrain_ ) {
			double[] nd = new double[yVal_.size()];
			for( int i = 0; i < nd.length; i++ )
				nd[i] = d;
			yTrain.add(nd);
		}
		
		List<double[]> xVal = new ArrayList<>();
		for( double[] d : xVal_ )
			xVal.add( Arrays.copyOf(d, d.length) );
		
		List<double[]> yVal = new ArrayList<double[]>();
		for( double d : yVal_ ) {
			double[] nd = new double[yVal_.size()];
			for( int i = 0; i < nd.length; i++ )
				nd[i] = d;
			yVal.add(nd);
		}
		
		ListNormalizer lnXTrain = new ListNormalizer( expTrans, xTrain);
		ListNormalizer lnYTrain = new ListNormalizer( respTrans, yTrain);										
		lnXTrain.normalize(xVal);
		lnYTrain.normalize(yVal);
		
		List<Function[]> layerList = new ArrayList<>();
		List<Function> input = new ArrayList<>();
		while (input.size() < xTrain.get(0).length )
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
		while (output.size() < yVal_.size())
			output.add(new Linear());
		layerList.add(output.toArray(new Function[] {} ) );	
						
		Function[][] layers = layerList.toArray( new Function[][] {} );
		double[][][] weights = NNetUtils.getFullyConnectedWeights(layers, NNetUtils.initMode.gorot_unif, seed);
				
		if( baseWeights != null ) 			
			for (int l = 0; l < weights.length-1; l++) // skip last layer
				for (int i = 0; i < weights[l].length; i++)
					for( int j = 0; j < weights[l][i].length; j++ )
						weights[l][i][j] = (1 - a) * weights[l][i][j] + a * baseWeights[l][i][j];
						
		GWANN gwann = new GWANN(layers, weights, eta, opt);
		gwann.lambda = lambda;
				
		List<Double> errors = new ArrayList<>();
		List<Integer> batchReservoir = new ArrayList<>();		
		int noImp = 0;
		double localBestValError = Double.POSITIVE_INFINITY;	
			
		for (int it = 0; it < maxIt && noImp < patience; it++) {
			
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
		}
								
		List<double[]> response= new ArrayList<>();
		for (int i = 0; i < xVal.size(); i++)
			response.add(gwann.present(xVal.get(i)));
		
		double[] res = new double[response.size()];
		for (int i = 0; i < xVal.size(); i++)
			res[i] = response.get(i)[i];
		
		double[] des = new double[yVal.size()];
		for( int i = 0; i < yVal.size(); i++ )
			des[i] = yVal.get(i)[0];
					
		List<double[]> response_denormed = new ArrayList<>();
		for( double[] d : response )
			response_denormed.add( Arrays.copyOf(d, d.length));
		lnYTrain.denormalize(response_denormed);
								
		ReturnObject ro = new ReturnObject();
		ro.errors = errors;
		ro.rmse = SupervisedUtils.getRMSE(res, des);
		ro.r2 = SupervisedUtils.getR2(res, des );
		ro.nnet = gwann;
		ro.prediction = response;	
		ro.prediction_denormed = response_denormed;
		return ro;
	}
		
	public static double[] getParamsWithGoldenSection(double minRadius, double maxRadius, 
			List<double[]> xArray, List<Double> yArray, DoubleMatrix W, List<Entry<List<Integer>, List<Integer>>> innerCvList, 
			GWKernel kernel, boolean adaptive, double[] eta, int batchSize, Optimizer opt, double lambda, int[] nrHidden, int iterations, int patience, int threads, double[][][] baseWeights, double a, Transform[] explTrans, Transform[] respTrans ) {
		double xU = maxRadius;
		double xL = minRadius;
		double eps = 1e-04;
		double R = (Math.sqrt(5) - 1) / 2.0;
		double d = R * (xU - xL);
		
		double x1 = xL + d;
		double x2 = xU - d;
		double[] f1 = NNetUtils.getBestErrorParams( getErrors_CV(xArray, yArray, W, innerCvList, kernel, x1, adaptive, eta, batchSize, opt, lambda, nrHidden, iterations, patience, threads, baseWeights, a, explTrans, respTrans) );
		double[] f2 = NNetUtils.getBestErrorParams( getErrors_CV(xArray, yArray, W, innerCvList, kernel, x2, adaptive, eta, batchSize, opt, lambda, nrHidden, iterations, patience, threads, baseWeights, a, explTrans, respTrans) );
		
		double d1 = f2[0] - f1[0];
		
		while ((Math.abs(d) > eps) && (Math.abs(d1) > eps)) {
			d = R * d;
			if (f1[0] < f2[0]) {
				xL = x2;
				x2 = x1;
				x1 = xL + d;
				f2 = f1;
				f1 = NNetUtils.getBestErrorParams( getErrors_CV(xArray, yArray, W, innerCvList, kernel, x1, adaptive, eta, batchSize, opt, lambda, nrHidden, iterations, patience, threads, baseWeights, a, explTrans, respTrans ) );
			} else {
				xU = x1;
				x1 = x2;
				x2 = xU - d;
				f1 = f2;
				f2 = NNetUtils.getBestErrorParams( getErrors_CV(xArray, yArray, W, innerCvList, kernel, x2, adaptive, eta, batchSize, opt, lambda, nrHidden, iterations, patience, threads, baseWeights, a, explTrans, respTrans ) );
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
			List<double[]> xArray, List<Double> yArray, DoubleMatrix W, List<Entry<List<Integer>, List<Integer>>> innerCvList, 
			GWKernel kernel, double bw, boolean adaptive, double[] eta, int batchSize, Optimizer opt, double lambda, int[] nrHidden, int iterations, int patience, int threads, double[][][] baseWeights, double a, Transform[] explTrans, Transform[] respTrans ) {
				
		ExecutorService innerEs = Executors.newFixedThreadPool((int) threads);
		List<Future<List<Double>>> futures = new ArrayList<Future<List<Double>>>();			
				
		for ( Entry<List<Integer>, List<Integer>> innerCvEntry : innerCvList ) {
			futures.add(innerEs.submit(new Callable<List<Double>>() {
				@Override
				public List<Double> call() throws Exception {	
					List<Integer> trainIdx = innerCvEntry.getKey();
					List<Integer> testIdx = innerCvEntry.getValue();
					
					List<double[]> xTrain = new ArrayList<>();
					List<Double> yTrain = new ArrayList<>();
					for (int i = 0; i < trainIdx.size(); i++ ) {
						int idx = trainIdx.get(i);
						xTrain.add(Arrays.copyOf(xArray.get(idx), xArray.get(idx).length));
						yTrain.add(yArray.get(idx));
					}
					
					List<double[]> xTest = new ArrayList<>();
					List<Double> yTest = new ArrayList<>();
					for (int i = 0; i < testIdx.size(); i++ ) {
						int idx = testIdx.get(i);
						xTest.add(Arrays.copyOf(xArray.get(idx), xArray.get(idx).length));
						yTest.add(yArray.get(idx));
					}
								
					int[] trainIdxA = DataUtils.toIntArray(trainIdx);
					DoubleMatrix W_train_test = W.get( trainIdxA, DataUtils.toIntArray(testIdx)); // train to test
					DoubleMatrix W_train_train = W.get(trainIdxA,trainIdxA);
					
					List<double[]> xArray_train = new ArrayList<>();
					for( int i = 0; i < xTrain.size(); i++ )
						xArray_train.add( xTrain.get(i) );
					
					List<double[]> xArray_test = new ArrayList<>();
					for( int i = 0; i < xTest.size(); i++ )
						xArray_test.add(xTest.get(i));	
					
					List<Double> errors = buildGWANN(xArray_train, yTrain, W_train_train, xArray_test, yTest, W_train_test, nrHidden, eta, opt, 0.0, batchSize, iterations, patience, kernel, bw, adaptive, baseWeights, a, explTrans, respTrans, 0).errors;
					return errors;
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
	
	public static double[] getParamsWithGridSearch(int minRadius, int maxRadius, int steps,
			List<double[]> xArray, List<Double> yArray, DoubleMatrix W, List<Entry<List<Integer>, List<Integer>>> innerCvList, int[] fa, int ta, GWKernel kernel, boolean adaptive, double[] eta, int batchSize, 
			Optimizer opt, double lambda, 
			int[] nrHidden, int iterations, int patience, int threads, double[][][] weights, double a, Transform[] expTrans, Transform[] respTrans ) {
		
		if( !adaptive )
			throw new RuntimeException("Not implemented yet");
				
		double[] bestF = null;
		int bestBw = -1;
		for( int i = (int)minRadius; i <= maxRadius; i+=steps ) {
			double[] f = NNetUtils.getBestErrorParams( getErrors_CV(xArray, yArray, W, innerCvList, kernel, i, adaptive, eta, batchSize, opt, lambda, nrHidden, iterations, patience, threads, weights, a, expTrans, respTrans) );
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
			int[] nrHidden, int[] ga, int[] fa, int ta, double[] eta, Optimizer opt, double lambda, 
			int batchSize, int maxIt, int patience, GWKernel kernel, double bw, boolean adaptive, double[][][] baseWeights, double a, Transform[] expTrans, Transform[] respTrans ) {
					
		List<double[]> xTrain = new ArrayList<>();
		List<Double> yTrain = new ArrayList<>();
		for (int i : trainIdx) {
			double[] d = samples.get(i);
			xTrain.add(DataUtils.strip(d, fa));
			yTrain.add(d[ta]);
		}
	
		List<double[]> xVal = new ArrayList<>();
		List<Double> yVal = new ArrayList<>();
		for (int i : testIdx) {
			double[] d = samples.get(i);
			xVal.add(DataUtils.strip(d, fa));
			yVal.add(d[ta]);
		}
			
		DoubleMatrix W_train = W.get( DataUtils.toIntArray(trainIdx), DataUtils.toIntArray(trainIdx));
		DoubleMatrix W_train_val = W.get( DataUtils.toIntArray(trainIdx), DataUtils.toIntArray(testIdx));
		
		return buildGWANN(xTrain, yTrain, W_train, xVal, yVal, W_train_val, nrHidden, eta, opt, lambda, batchSize, maxIt, patience, kernel, bw, adaptive, baseWeights, a, expTrans, respTrans, 0);
	}
	
	public static double[] getErrors_CV(List<double[]> samplesA, DoubleMatrix W, List<Entry<List<Integer>, List<Integer>>> innerCvList, int[] fa, int ta, GWKernel kernel, double bw, boolean adaptive, double[] eta, int batchSize, Optimizer opt, double lambda, int[] nrHidden, int maxIt, int patience, int threads, double[][][] baseWeights, double a, Transform[] expTrans, Transform[] respTrans ) {
		List<double[]> xTrain = new ArrayList<>();
		List<Double> yTrain = new ArrayList<>();
		for (double[] d : samplesA ) {
			xTrain.add(DataUtils.strip(d, fa));
			yTrain.add(d[ta]);
		}
		
		List<List<Double>> errors = getErrors_CV(xTrain, yTrain, W, innerCvList, kernel, bw, adaptive, eta, batchSize, opt, lambda, nrHidden, maxIt, patience, threads, baseWeights, a, expTrans, respTrans);		
		return NNetUtils.getBestErrorParams( errors );
	}
	
	public static double[] getParamsWithGoldenSection(double minRadius, double maxRadius, 
			List<double[]> samples, DoubleMatrix W, List<Entry<List<Integer>, List<Integer>>> innerCvList, int[] fa, int ta, GWKernel kernel, boolean adaptive, double[] eta, int batchSize, 
			Optimizer opt, double lambda, 
			int[] nrHidden, int iterations, int patience, int threads, double[][][] baseWeights, double a, Transform[] expTrans, Transform[] respTrans ) {
		
		List<double[]> xTrain = new ArrayList<>();
		List<Double> yTrain = new ArrayList<>();
		for (double[] d : samples ) {
			xTrain.add(DataUtils.strip(d, fa));
			yTrain.add(d[ta]);
		}
		return getParamsWithGoldenSection(minRadius,maxRadius,xTrain, yTrain, W, innerCvList, kernel, adaptive, eta, batchSize, opt, lambda, nrHidden, iterations, patience, threads, baseWeights, a, expTrans, respTrans);
	}
	
	public static double[] getParamsWithGridSearch(int minRadius, int maxRadius, int steps, 
			List<double[]> samples, DoubleMatrix W, List<Entry<List<Integer>, List<Integer>>> innerCvList, int[] fa, int ta, GWKernel kernel, boolean adaptive, double[] eta, int batchSize, 
			Optimizer opt, double lambda, 
			int[] nrHidden, int iterations, int patience, int threads, double[][][] weights, double a, Transform[] expTrans, Transform[] respTrans ) {
		
		List<double[]> xTrain = new ArrayList<>();
		List<Double> yTrain = new ArrayList<>();
		for (double[] d : samples ) {
			xTrain.add(DataUtils.strip(d, fa));
			yTrain.add(d[ta]);
		}
		return getParamsWithGridSearch(minRadius, maxRadius, steps, xTrain, yTrain, W, innerCvList, fa, ta, kernel, adaptive, eta, batchSize, opt, lambda, nrHidden, iterations, patience, threads, weights, a, expTrans, respTrans);
	}
}
