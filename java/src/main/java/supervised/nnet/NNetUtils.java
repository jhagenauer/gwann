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

import supervised.SupervisedUtils;
import supervised.nnet.activation.Constant;
import supervised.nnet.activation.Function;
import supervised.nnet.activation.Linear;
import supervised.nnet.activation.TanH;
import utils.DataUtils;
import utils.ListNormalizer;
import utils.Normalizer.Transform;

public class NNetUtils {

	public static enum initMode {
		gorot_unif, norm05
	}

	public static double[] getParamsCV(List<double[]> samplesA, List<Entry<List<Integer>, List<Integer>>> innerCvList, int[] fa, int ta, double[] eta, int batchSize, NNet.Optimizer opt, double lambda, int[] nrHidden, int maxIt, int patience, int threads, Transform[] expTrans, Transform[] respTrans ) {
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
						double[] d = samplesA.get(i);
						xTrain.add(DataUtils.strip(d, fa));
						yTrain.add(d[ta]);
					}
				
					List<double[]> xVal = new ArrayList<>();
					List<Double> yVal = new ArrayList<>();
					for (int i : testIdx) {
						double[] d = samplesA.get(i);
						xVal.add(DataUtils.strip(d, fa));
						yVal.add(d[ta]);
					}
				
					ReturnObject ro = NNetUtils.buildNNet(xTrain, yTrain, xVal, yVal, nrHidden, eta, opt, lambda, batchSize, maxIt, patience, expTrans, respTrans );
					return ro.errors;
				}
			}));
		}
		innerEs.shutdown();
		return NNetUtils.getMinMeanIdx(futures);
	}

	// cleaner interface
	public static ReturnObject buildNNet(List<double[]> xTrain_, List<Double> yTrain_, List<double[]> xVal_, List<Double> yVal_, int[] nrHidden, double[] eta, NNet.Optimizer opt, double lambda, int batchSize, int maxIt, int patience, Transform[] expTrans, Transform[] respTrans) {
		Random r = new Random(0);
	
		List<double[]> xTrain = new ArrayList<>();
		for (double[] d : xTrain_)
			xTrain.add(Arrays.copyOf(d, d.length));
	
		List<double[]> yTrain = new ArrayList<>();
		for (double d : yTrain_) {
			/*double[] nd = new double[yVal_.size()];
			for (int i = 0; i < nd.length; i++)
				nd[i] = d;*/
			double[] nd = new double[] {d};
			yTrain.add(nd);
		}
	
		List<double[]> xVal = new ArrayList<>();
		for (double[] d : xVal_)
			xVal.add(Arrays.copyOf(d, d.length));
	
		List<double[]> yVal = new ArrayList<double[]>();
		for (double d : yVal_) {
			/*double[] nd = new double[yVal_.size()];
			for (int i = 0; i < nd.length; i++)
				nd[i] = d;*/
			double[] nd = new double[] {d};
			yVal.add(nd);
		}
	
		ListNormalizer lnXTrain = new ListNormalizer(expTrans, xTrain);
		lnXTrain.normalize(xVal);
		
		ListNormalizer lnYTrain = new ListNormalizer(respTrans, yTrain);	
		lnYTrain.normalize(yVal);
	
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
	
		NNet nnet = new NNet(layers, weights, eta, opt);
		nnet.lambda = lambda;
	
		List<Integer> batchReservoir = new ArrayList<>();
		List<Double> errors = new ArrayList<>();
		int noImp = 0;
		double localBestValError = Double.POSITIVE_INFINITY;
	
		for (int it = 0; it < maxIt && noImp < patience; it++) {
	
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
	
			List<Double> responseVal = new ArrayList<>();
			for (int i = 0; i < xVal.size(); i++)
				responseVal.add( nnet.present(xVal.get(i))[0] );  // only the first
			double valError = SupervisedUtils.getRMSE(responseVal, yVal, 0);
			errors.add(valError);
	
			if (valError < localBestValError) {
				localBestValError = valError;
				noImp = 0;
			} else
				noImp++;
		}
	
		double[] des = new double[yVal.size()];
		for (int i = 0; i < yVal.size(); i++)
			des[i] = yVal.get(i)[0];
	
		List<double[]> response = new ArrayList<>();
		for (int i = 0; i < xVal.size(); i++)
			response.add(nnet.present(xVal.get(i)));
	
		double[] res = new double[response.size()];
		for (int i = 0; i < xVal.size(); i++)
			res[i] = response.get(i)[0]; // only the first
		double valError = SupervisedUtils.getRMSE(res, des);
	
		List<double[]> response_denormed = new ArrayList<>();
		for (double[] d : response)
			response_denormed.add(Arrays.copyOf(d, d.length));
		lnYTrain.denormalize(response_denormed);
		
		ReturnObject ro = new ReturnObject();
		ro.errors = errors;
		ro.rmse = valError;
		ro.r2 = SupervisedUtils.getR2(res, des);
		ro.nnet = nnet;
		ro.prediction = response;
		ro.prediction_denormed = response_denormed;
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
	
	public static double[] getMinMeanIdx(List<Future<List<Double>>> futures) {			
		int minSize = Integer.MAX_VALUE;
		double[] mean = null; 
		try {
			for( Future<List<Double>> f : futures )	
				minSize = Math.min(f.get().size(), minSize);
						
			mean = new double[minSize];
			for( Future<List<Double>> f : futures )
				for( int i = 0; i < mean.length; i++ )
					mean[i] += f.get().get(i)/futures.size();
		} catch (InterruptedException | ExecutionException e) {
			e.printStackTrace();
		} 
		
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
