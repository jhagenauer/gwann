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

import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import supervised.SupervisedUtils;
import supervised.nnet.activation.Constant;
import supervised.nnet.activation.Function;
import supervised.nnet.activation.Linear;
import supervised.nnet.activation.Logistic;
import supervised.nnet.activation.ReLu;
import supervised.nnet.activation.TanH;
import utils.DataUtils;
import utils.ListNormalizer;
import utils.Normalizer.Transform;

public class NNetUtils {
	
	private static Logger log = LogManager.getLogger(NNetUtils.class);

	public static enum initMode {
		gorot_unif, norm05
	}
	
	public static enum Neuron_type {
		Constant, TanH, Logistic, ReLu
	}
	
	public static Neuron_type hidden_neuron_type = Neuron_type.TanH;
	
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
										
					List<double[]> xTrain = DataUtils.subset_rows(xArray, trainIdx);
					List<double[]> yTrain = DataUtils.subset_rows(yArray, trainIdx);				
					List<double[]> xVal = DataUtils.subset_rows(xArray, testIdx);
					List<double[]> yVal = DataUtils.subset_rows(yArray, testIdx);
									
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
			int[] nr_hidden, double[] eta, NNet.Optimizer opt, 
			double lambda, int batchSize, 
			int max_its, int patience, 
			Transform[] expTrans, Transform[] respTrans
			) {
		Random r = new Random(0);
		
		assert nr_hidden.length+1 == eta.length :nr_hidden.length+"<->"+eta.length;
			
		List<double[]> x_train = xTrain_.stream().map(arr -> arr.clone()).toList();		
		List<double[]> y_train = yTrain_.stream().map(arr -> arr.clone()).toList();	
		List<double[]> x_test = xTest_.stream().map(arr -> arr.clone()).toList();	
	
		double[] test_desired_not_normalized = new double[yTest_.size()];
		List<double[]> y_test = new ArrayList<>();
		for( int i = 0; i < yTest_.size(); i++ ) {
			double[] d = yTest_.get(i);
			test_desired_not_normalized[i] = d[0];
			y_test.add( Arrays.copyOf(d, d.length) );
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
	
		for (int nh : nr_hidden) {
			if( nh == 0 )
				continue;
			List<Function> hidden0 = new ArrayList<>();
			while (hidden0.size() < nh) {
				switch(hidden_neuron_type) {
				case Constant:
					hidden0.add(new Constant(1.0));
					break;
				case Logistic:
					hidden0.add(new Logistic());
					break;
				case ReLu:
					hidden0.add(new ReLu());
					break;
				case TanH:
					hidden0.add(new TanH());
					break;
				default:
					throw new RuntimeException(hidden_neuron_type+" not supported");						
				}					
			}
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
		for (int it = 0; it < max_its && no_imp < patience; it++) {
	
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
				no_imp = 0;
			} else
				no_imp++;
		}
		
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
		ro.errors = errors;
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
	
	public static enum error_param {last,argmin_mean,ose_rule,argmin_median,median_argmin};
	public static double[] getErrorParameters(List<List<Double>> errors, error_param mode) {
	   
	    final int k = errors.size();
	    int smallest_common_length = Integer.MAX_VALUE;
	    assert !errors.isEmpty();
	    for (List<Double> foldErrors : errors) {
	        if (foldErrors == null || foldErrors.isEmpty()) {
	            throw new IllegalArgumentException("One of the fold error lists is null or empty");
	        }
	        smallest_common_length = Math.min(smallest_common_length, foldErrors.size());
	    }
	    if (smallest_common_length <= 0) {
	        throw new IllegalStateException("No common length across folds.");
	    }

	    // For "last" mode we require equal lengths to avoid ambiguity.
	    if (mode == error_param.last) {
	        for (List<Double> foldErrors : errors) {
	            if (foldErrors.size() != smallest_common_length) {
	                throw new RuntimeException("Error lists have different length! " +
	                                           smallest_common_length + " != " + foldErrors.size());
	            }
	        }
	    }

	    double[] mean = new double[smallest_common_length];
	    double[] median = new double[smallest_common_length];
	    double[] sd   = new double[smallest_common_length];
	    double[] se   = new double[smallest_common_length];
	    	   
	    // Mean across folds for each epoch index
	    for (int t = 0; t < smallest_common_length; t++) {
	        	        
	        DescriptiveStatistics ds = new DescriptiveStatistics();
	        for (List<Double> foldErrors : errors) 
	            ds.addValue( foldErrors.get(t) );
	        	       	        
	        mean[t] = ds.getMean();
	        median[t] = ds.getPercentile(50);
	    }

	    // Sample standard deviation and standard error
	    for (int t = 0; t < smallest_common_length; t++) {
	        double sumsq = 0.0;
	        for (List<Double> foldErrors : errors) {
	            double diff = foldErrors.get(t) - mean[t];
	            sumsq += diff * diff;
	        }
	        if (k > 1) {
	            sd[t] = Math.sqrt(sumsq / (k - 1));   // sample SD
	        } else {
	            sd[t] = 0.0;
	        }
	        se[t] = (k > 0) ? sd[t] / Math.sqrt(k) : Double.NaN;
	    }

	    int idx;
	    if (mode == error_param.last) {
	        // last valid epoch index
	        idx = smallest_common_length - 1;

	    } else if (mode == error_param.argmin_mean || mode == error_param.argmin_median ) {
	        // argmin of mean
	        idx = 0;
	        double min = Double.POSITIVE_INFINITY;
	        for (int t = 1; t < smallest_common_length; t++) {	        	
	        	double v = mode == error_param.argmin_median ? mean[t] : median[t];	        	
	            if (v < min ) {
	                min = v;
	                idx = t;
	            }
	        }

	    } else if (mode == error_param.ose_rule) {
	        // one-standard-error rule
	        int idxMin = 0;
	        double minMean = mean[0];
	        for (int t = 1; t < smallest_common_length; t++) {
	            if (mean[t] < minMean) {
	                minMean = mean[t];
	                idxMin = t;
	            }
	        }
	        double threshold = minMean + se[idxMin];

	        // Choose the earliest epoch within the threshold
	        idx = 0;
	        while (idx < smallest_common_length && mean[idx] > threshold) {
	            idx++;
	        }
	        if (idx >= smallest_common_length) {
	            // Fallback (should be rare due to numerical precision)
	            idx = idxMin;
	        }

	    } else if(mode == error_param.median_argmin) {
	    	 DescriptiveStatistics ds = new DescriptiveStatistics();
	    	 for (List<Double> foldErrors : errors) {
	    		 
	    		 double best_e = Double.POSITIVE_INFINITY;
	    		 int best_idx = -1;
	    		 for( int i = 0; i < smallest_common_length; i++ ) {
	    			 double e = foldErrors.get(i);
	    			 if( e < best_e ) {
	    				 best_e = e;
	    				 best_idx = i;
	    			 }
	    		 }
	    		 ds.addValue(best_idx);		            
	    	 }
	    	 idx = (int)ds.getPercentile(50);
	    	 
	    	 if( idx > mean.length ) {
	    		 log.debug( Arrays.toString(ds.getSortedValues())+":::"+mean.length);
	    	 }
	    	 
	    } else {
	        throw new IllegalArgumentException("Unknown mode: " + mode + " (expected 0=last, 1=min, 2=one-SE)");
	    }

	    return new double[] {
	        mean[idx],
	        (double) idx,
	        sd[idx]    // returning SD at the chosen epoch; SE = sd[idx] / sqrt(k) if you need it
	    };
	}


	public static double[] getArray(double value, int length) {
		double[] r = new double[length];
		for( int i = 0; i < r.length; i++ )
			r[i] = value;
		return r;
	}
}
