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
import java.util.concurrent.atomic.AtomicInteger;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.jblas.DoubleMatrix;

import supervised.SupervisedUtils;
import supervised.nnet.NNet;
import supervised.nnet.NNet.Optimizer;
import supervised.nnet.NNetUtils;
import supervised.nnet.NNetUtils.error_param;
import supervised.nnet.ReturnObject;
import supervised.nnet.activation.Constant;
import supervised.nnet.activation.Function;
import supervised.nnet.activation.Linear;
import supervised.nnet.activation.TanH;
import utils.DataUtils;
import utils.GWUtils;
import utils.GWUtils.GWKernel;
import utils.ListNormalizer;
import utils.Normalizer.Transform;

public class GWANNUtils {
	
	public static boolean norm = false;
	
	private static Logger log = LogManager.getLogger(GWANNUtils.class);
	public static double eta_decay = 1.0;	
	public static boolean new_grad = false;
							
	// cleaner interface
	public static ReturnObject buildGWANN( 
			List<double[]> x_train_, List<double[]> y_train_, 
			List<double[]> x_test_, List<double[]> y_test_, // only needed for error calculation etc
			DoubleMatrix W_train_test, 
			int[] hidden_layers, double[] eta, Optimizer opt, 
			int batch_size, int max_it, int patience,  
			GWKernel kernel, 
			double[] bw, 
			boolean adaptive, double lambda,
			Transform[] expTrans, Transform[] respTrans, 
			double[][][] init_weights, 
			int seed ) {
		
		assert y_train_.get(0).length == 1;
		assert y_test_.get(0).length == 1;
		assert hidden_layers.length + 1 == eta.length;
		
		Random r = new Random(seed);		
		
		double[][] kW = null; // [train][test]
		double[][][] kW_2d = null; // [bw][train][test]
		if( bw.length==1) {
			DoubleMatrix mat = adaptive ? GWUtils.getKernelWeightsAdaptive(W_train_test, kernel, (int) bw[0]) : GWUtils.getKernelWeights(W_train_test, kernel, bw[0]);
			if( norm )
				DataUtils.rowNormalizeMatrix(mat);
			kW = mat.toArray2();
		} else {
						
			kW_2d = new double[bw.length][][];
			for( int k = 0; k < bw.length; k++ ) {
				DoubleMatrix mat = adaptive ? GWUtils.getKernelWeightsAdaptive(W_train_test, kernel, (int) bw[k] ) : GWUtils.getKernelWeights(W_train_test, kernel, bw[k]);
				if( norm )
					DataUtils.columnNormalizeMatrix(mat);
				kW_2d[k] = mat.toArray2();
			}	
		}
			
		List<double[]> x_train = x_train_.stream().map(arr -> arr.clone()).toList();	
		List<double[]> y_train = y_train_.stream().map(arr -> arr.clone()).toList();
						
		List<double[]> x_test = x_test_.stream().map(arr -> arr.clone()).toList();	
		List<double[]> y_test = y_test_.stream().map(arr -> arr.clone()).toList();	
				
		double[] desired_orig = new double[y_test_.size()];
		for( int i = 0; i < y_test.size(); i++ ) {
			double d = y_test.get(i)[0];
			desired_orig[i] = d;
		}	
		
		y_train = broadcast1D(y_train,y_test.size());
		y_test = broadcast1D(y_test,y_test.size());
		
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
		
		for( int nh : hidden_layers ) {
			if( nh == 0 )
				continue;
			List<Function> hidden0 = new ArrayList<>();
			while (hidden0.size() < nh) 
					hidden0.add(new TanH());				
			hidden0.add(new Constant(1.0));
			layerList.add(hidden0.toArray(new Function[] {} ) );
		}
				
		List<Function> output = new ArrayList<>();
		while (output.size() < y_test_.size())
			output.add(new Linear());
		layerList.add(output.toArray(new Function[] {} ) );	
				
		if( bw.length > 1 && layerList.get(layerList.size()-2).length != bw.length ) {
			log.debug("bw: "+bw.length+", ll.s: "+layerList.size()+","+Arrays.toString(bw));
			for( Function[] l : layerList )
				log.debug(layerList.indexOf(l)+"->"+l.length);
			throw new RuntimeException(bw.length+" != "+layerList.get(layerList.size()-2).length+" ||| "+input.size()+" -> "+Arrays.toString(hidden_layers)+" -> "+output.size() );
		}
						
		Function[][] layers = layerList.toArray( new Function[][] {} );
		
		double[][][] weights = init_weights == null ? NNetUtils.getFullyConnectedWeights(layers, NNetUtils.initMode.gorot_unif, seed) : NNet.deepCopy3DArray(init_weights);
		GWANN gwann = new GWANN(layers, weights, eta, opt, lambda);
		List<Double> errors = new ArrayList<>();	
													
		List<Integer> batchReservoir = new ArrayList<>();			
		int no_imp = 0;
		double test_error_best = Double.POSITIVE_INFINITY;
		double[] eta_current = eta.clone();
		for (int it = 0; it < max_it && no_imp < patience; it++) {
						
			if( bw.length == 1 ) {	
				
				List<double[]> x = new ArrayList<>();
				List<double[]> y = new ArrayList<>();
				List<double[]> gwWeights = new ArrayList<>();			
				if( batch_size < 0 ) {
					for( int k = 0; k < x_train.size(); k++ ) {
						x.add(x_train.get(k));
						y.add(y_train.get(k));	
						gwWeights.add(kW[k]); 
					}
				} else {						
					while (x.size() < batch_size) {
						if (batchReservoir.isEmpty())
							for (int j = 0; j < x_train.size(); j++)
								batchReservoir.add(j);
						int k = batchReservoir.remove(r.nextInt(batchReservoir.size()));
						x.add(x_train.get(k));
						y.add(y_train.get(k));	
						gwWeights.add(kW[k]); 
					}
				}
				
				// train 
				List<double[][][]> l = new ArrayList<>();
				for( int i = 0; i < x.size(); i++ )
					l.add( gwann.getErrorGradient( x.get(i), y.get(i), gwWeights.get(i) ) );
				//double[][][] errorGrad = NNetUtils.calculateMeanOf3DList(l);	
				double[][][] errorGrad = GWANN.norm_grad ? GWANNUtils.calculateSpatiallyNormalizedGradient(l) : NNetUtils.calculateMeanOf3DList(l);
				gwann.update(opt, errorGrad, eta_current);
	
			} else {
											
				List<double[]> x = new ArrayList<>();
				List<double[]> y = new ArrayList<>();
				List<double[][]> gwWeights = new ArrayList<>();			
				if( batch_size < 0 ) {
					for( int k = 0; k < x_train.size(); k++ ) {
						x.add(x_train.get(k));
						y.add(y_train.get(k));	
						
						double[][] w = new double[bw.length][];
						for( int i = 0; i < bw.length; i++ )
							w[i] = kW_2d[i][k];
						gwWeights.add(w); 
					}
				} else {						
					while (x.size() < batch_size) {
						if (batchReservoir.isEmpty())
							for (int j = 0; j < x_train.size(); j++)
								batchReservoir.add(j);
						int k = batchReservoir.remove(r.nextInt(batchReservoir.size()));
						x.add(x_train.get(k));
						y.add(y_train.get(k));	
						
						double[][] w = new double[bw.length][]; // [bw][test]
						for( int i = 0; i < bw.length; i++ )
							w[i] = kW_2d[i][k];
						gwWeights.add(w); 
					}
				}
								
				// train				
				List<double[][][]> l = new ArrayList<>();
				for( int i = 0; i < x.size(); i++ ) {
					double[][][] eg = new_grad ? gwann.getErrorGradient_new(x.get(i), y.get(i), gwWeights.get(i) ) : gwann.getErrorGradient(x.get(i), y.get(i), gwWeights.get(i) ); 
					l.add( eg );
				}	
				double[][][] errorGrad = GWANN.norm_grad ? GWANNUtils.calculateSpatiallyNormalizedGradient(l) : NNetUtils.calculateMeanOf3DList(l);
				gwann.update(opt, errorGrad, eta_current);				
			}
																	
			// get denormalized response-diag 
			double[] response_denormalized= new double[x_test.size()];
			for (int i = 0; i < x_test.size(); i++) {
				double[] d = gwann.present(x_test.get(i));
				ln_y.denormalize(d,i);
				response_denormalized[i] = d[i]; 
			}		
			
			double test_error = SupervisedUtils.getRMSE(response_denormalized, desired_orig);
			errors.add(test_error);
			
			if (test_error < test_error_best) {
				test_error_best = test_error;
				no_imp = 0;
			} else
				no_imp++;		
			
			for (int j = 0; j < eta_current.length; j++) 
		        eta_current[j] *= eta_decay; 		    
		}
									
		// get response and denormalize
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
		ro.mse = SupervisedUtils.getMSE(response_denormalized_diag, desired_orig);
		ro.rmse = Math.sqrt(ro.mse);
		ro.r2 = SupervisedUtils.getR2(response_denormalized_diag, desired_orig );
		ro.nnet = gwann;
		ro.prediction = response_denormalized;
		ro.ln_x = ln_x;
		ro.ln_y = ln_y;
		return ro;
	}
	
	public static boolean debug = false;
	
	public static ReturnObject buildGWANN_backfitting( 
			List<double[]> x_train_, List<double[]> y_train_, 
			List<double[]> x_val_, List<double[]> y_val_, DoubleMatrix W_train_val, 
			List<double[]> x_test_, List<double[]> y_test_, DoubleMatrix W_train_test, 
			int[] hidden_layers, double[] eta, Optimizer opt, 
			int batch_size, int max_it, int patience,  
			GWKernel kernel, 
			double[] bw, 
			boolean adaptive, double lambda,
			Transform[] expTrans, Transform[] respTrans, int seed ) {
		
		assert y_train_.get(0).length == 1;
		assert y_test_.get(0).length == 1;
		assert hidden_layers.length + 1 == eta.length;
		
		Random r = new Random(seed);	
		
		DoubleMatrix W = DoubleMatrix.concatHorizontally(W_train_val, W_train_test);
		
		double[][][] kW_2d = null; // [bw][train][test]
		kW_2d = new double[bw.length][][];
		for( int k = 0; k < bw.length; k++ ) {
			DoubleMatrix mat = adaptive ? GWUtils.getKernelWeightsAdaptive(W, kernel, (int) bw[k] ) : GWUtils.getKernelWeights(W, kernel, bw[k]);
			if( norm )
				DataUtils.columnNormalizeMatrix(mat);	
			kW_2d[k] = mat.toArray2();
		}	
			
		List<double[]> x_train = x_train_.stream().map(arr -> arr.clone()).toList();	
		List<double[]> y_train = y_train_.stream().map(arr -> arr.clone()).toList();
		
		List<double[]> x_val = x_val_.stream().map(arr -> arr.clone()).toList();	
		List<double[]> y_val = y_val_.stream().map(arr -> arr.clone()).toList();	
						
		List<double[]> x_test = x_test_.stream().map(arr -> arr.clone()).toList();	
		List<double[]> y_test = y_test_.stream().map(arr -> arr.clone()).toList();	
						
		double[] desired_test_orig = new double[y_test_.size()];
		for( int i = 0; i < y_test.size(); i++ )
			desired_test_orig[i] = y_test.get(i)[0];
		
		int os = y_val.size() + y_test.size();
						
		y_train = broadcast1D(y_train,os);
		y_val = broadcast1D(y_val,os);
		y_test = broadcast1D(y_test,os);
		
		ListNormalizer ln_x = new ListNormalizer( expTrans, x_train);
		ln_x.normalize(x_val);
		ln_x.normalize(x_test);
		
		ListNormalizer ln_y = new ListNormalizer( respTrans, y_train);		
		ln_y.normalize(y_val);
		ln_y.normalize(y_test);
		
		assert y_train.get(0).length==y_val.get(0).length && y_val.get(0).length == y_test.get(0).length;
		
		double[] desired_val = new double[y_val_.size()];
		for( int i = 0; i < y_val.size(); i++ )
			desired_val[i] = y_val.get(i)[i];
					
		List<Function[]> layerList = new ArrayList<>();
		List<Function> input = new ArrayList<>();
		while (input.size() < x_train.get(0).length )
			input.add(new Linear());
		input.add(new Constant(1.0));
		layerList.add(input.toArray(new Function[] {} ) );
		
		for( int nh : hidden_layers ) {
			if( nh == 0 )
				continue;
			List<Function> hidden0 = new ArrayList<>();
			while (hidden0.size() < nh) 
					hidden0.add(new TanH());				
			hidden0.add(new Constant(1.0));
			layerList.add(hidden0.toArray(new Function[] {} ) );
		}
				
		List<Function> output = new ArrayList<>();
		while (output.size() < os)
			output.add(new Linear());
		layerList.add(output.toArray(new Function[] {} ) );	
		
		if( bw.length > 1 && layerList.get(layerList.size()-2).length != bw.length ) 
			throw new RuntimeException(bw.length+" != "+layerList.get(layerList.size()-2).length+"|"+input.size()+" -> "+Arrays.toString(hidden_layers)+" -> "+output.size() );
								
		Function[][] layers = layerList.toArray( new Function[][] {} );
		double[][][] weights = NNetUtils.getFullyConnectedWeights(layers, NNetUtils.initMode.gorot_unif, seed);						
		GWANN gwann = new GWANN(layers, weights, eta, opt, lambda);
		
		double error_last = Double.POSITIVE_INFINITY;
		
		for( int n = 0; n < 500; n++ ) {
			
			for( int f = 0; f < layers[0].length; f++ )	{
								
				gwann.resetVs();
					
				// train feature f
				List<Integer> batchReservoir = new ArrayList<>();
				int no_imp = 0;
				double error_val_best = Double.POSITIVE_INFINITY;
				double[][][] weights_best = null;
				for (int it = 0; it < max_it && no_imp < patience; it++) {
							
					List<double[]> x = new ArrayList<>();
					List<double[]> y = new ArrayList<>();									
					List<double[][]> gwWeights = new ArrayList<>();			
					if( batch_size < 0 ) {
						for( int k = 0; k < x_train.size(); k++ ) {
							x.add(x_train.get(k));
							y.add(y_train.get(k));	
							
							double[][] w = new double[bw.length][];
							for( int i = 0; i < bw.length; i++ )
								w[i] = kW_2d[i][k];
							gwWeights.add(w); 
						}
					} else {						
						while (x.size() < batch_size) {
							if (batchReservoir.isEmpty())
								for (int j = 0; j < x_train.size(); j++)
									batchReservoir.add(j);
							int k = batchReservoir.remove(r.nextInt(batchReservoir.size()));
							x.add(x_train.get(k));
							y.add(y_train.get(k));	
								
							double[][] w = new double[bw.length][]; // [bw][test]
							for( int i = 0; i < bw.length; i++ )
								w[i] = kW_2d[i][k];
							gwWeights.add(w); 
						}
					}				
					gwann.train_(x,y,gwWeights, f, null);				
																									
					// get response-diag val
					double[] response_val= new double[x_val.size()];
					for (int i = 0; i < x_val.size(); i++) {
						double[] d = gwann.present(x_val.get(i));
						response_val[i] = d[i]; 
					}	
					
					double error_val = SupervisedUtils.getRSS(response_val, desired_val);						
					if (error_val < error_val_best) {
						error_val_best = error_val;
						weights_best = gwann.getCopyOfWeights();
						no_imp = 0;
					} else
						no_imp++;			
				}
				gwann.setWeights(weights_best); // ... and next feature
			}
			
			double[] response_val= new double[x_val.size()];
			for (int i = 0; i < x_val.size(); i++) {
				double[] d = gwann.present(x_val.get(i));
				response_val[i] = d[i]; 
			}		
			double error_val = SupervisedUtils.getRSS(response_val, desired_val);
					
			{
				// get response and denormalize
				List<double[]> response_test_denorm= new ArrayList<>();
				for (int i = 0; i < x_test.size(); i++)
					response_test_denorm.add(gwann.present(x_test.get(i)));
				ln_y.denormalize(response_test_denorm);
							
				// response_diag
				double[] response_denorm_diag = new double[response_test_denorm.size()];
				for (int i = 0; i < x_test.size(); i++)
					response_denorm_diag[i] = response_test_denorm.get(i)[x_val.size()+i];
				
				log.debug(n+" --> rmse val: "+error_val+", r2 test: "+SupervisedUtils.getR2(response_denorm_diag, desired_test_orig ));
			}
			
			double rel_tol = 0.001;
			if ((error_last - error_val) / error_last < rel_tol) // no significant relative improvement
			    break;
			error_last = error_val;				
		} 
								
		// get response and denormalize
		List<double[]> response_test_denorm= new ArrayList<>();
		for (int i = 0; i < x_test.size(); i++)
			response_test_denorm.add(gwann.present(x_test.get(i)));
		ln_y.denormalize(response_test_denorm);
				
		// response_diag
		double[] response_denorm_diag = new double[response_test_denorm.size()];
		for (int i = 0; i < x_test.size(); i++)
			response_denorm_diag[i] = response_test_denorm.get(i)[x_val.size()+i];
														
		ReturnObject ro = new ReturnObject();
		ro.mse = SupervisedUtils.getMSE(response_denorm_diag, desired_test_orig);
		ro.rmse = Math.sqrt(ro.mse);
		ro.r2 = SupervisedUtils.getR2(response_denorm_diag, desired_test_orig );
		ro.nnet = gwann;
		ro.prediction = response_test_denorm;
		ro.ln_x = ln_x;
		ro.ln_y = ln_y;
		return ro;
	}
		
	public static List<double[]> broadcast1D( List<double[]> y, int i ) {
		List<double[]> r = new ArrayList<>();
		for(  double[] d : y ) {			
			assert d.length == 1;
			
			double[] nd = new double[i];
			for( int j = 0; j < i; j++ )
				nd[j] = d[0];
			r.add( nd );
		}
		return r;
	}
		
	public static double[] getParamsWithGoldenSection(double minRadius, double maxRadius, 
			List<double[]> xArray, List<double[]> yArray, DoubleMatrix W, List<Entry<List<Integer>, List<Integer>>> cv_list, 
			GWKernel kernel, boolean adaptive, double[] eta, int batchSize, Optimizer opt, int[] nrHidden, int iterations, int patience, int threads, double lambda, Transform[] explTrans, Transform[] respTrans, double[][][] init_weights, int seed ) {
		double xU = maxRadius;
		double xL = minRadius;
		double eps = 1e-04;
		double R = (Math.sqrt(5) - 1) / 2.0;
		double d = R * (xU - xL);
		
		double x1 = xL + d;
		double x2 = xU - d;
		
		boolean fast_cv = false;
						
		double[] f1 = NNetUtils.getErrorParameters( getErrors_CV(xArray, yArray, W, cv_list, kernel, new double[] {x1}, adaptive, eta, batchSize, opt, nrHidden, iterations, patience, threads, lambda, explTrans, respTrans, init_weights, seed, fast_cv), error_param.argmin_mean );
		double[] f2 = NNetUtils.getErrorParameters( getErrors_CV(xArray, yArray, W, cv_list, kernel, new double[] {x2}, adaptive, eta, batchSize, opt, nrHidden, iterations, patience, threads, lambda, explTrans, respTrans, init_weights, seed, fast_cv), error_param.argmin_mean );
		
		double d1 = f2[0] - f1[0];
		while ((Math.abs(d) > eps) && (Math.abs(d1) > eps)) {
			d = R * d;
			if (f1[0] < f2[0]) {
				xL = x2;
				x2 = x1;
				x1 = xL + d;
				f2 = f1;
				f1 = NNetUtils.getErrorParameters( getErrors_CV(xArray, yArray, W, cv_list, kernel, new double[] {x1}, adaptive, eta, batchSize, opt, nrHidden, iterations, patience, threads, lambda, explTrans, respTrans, init_weights, seed, fast_cv ), error_param.argmin_mean );
			} else {
				xU = x1;
				x1 = x2;
				x2 = xU - d;
				f1 = f2;
				f2 = NNetUtils.getErrorParameters( getErrors_CV(xArray, yArray, W, cv_list, kernel, new double[] {x2}, adaptive, eta, batchSize, opt, nrHidden, iterations, patience, threads, lambda, explTrans, respTrans, init_weights, seed, fast_cv ), error_param.argmin_mean );
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
			GWKernel kernel, double[] bw, boolean adaptive, double[] eta, int batchSize, Optimizer opt, int[] nrHidden, int iterations, int patience, 
			int threads, double lambda, Transform[] explTrans, Transform[] respTrans, double[][][] init_weights, int seed ) {
		return getErrors_CV(xArray,yArray,W,innerCvList,kernel,bw,adaptive,eta,batchSize,opt,nrHidden,iterations,patience,threads,lambda, explTrans, respTrans, init_weights, seed, false);		
	}
	
	public static List<List<Double>> getErrors_CV(
			List<double[]> xArray, List<double[]> yArray, 
			DoubleMatrix W, 
			List<Entry<List<Integer>, List<Integer>>> innerCvList, 
			GWKernel kernel, double[] bw, boolean adaptive, 
			double[] eta, int batchSize, Optimizer opt, int[] nrHidden, 
			int iterations, int patience, 
			int threads, double lambda, Transform[] explTrans, Transform[] respTrans, double[][][] init_weights, int seed, boolean fast_cv ) {
				
		ExecutorService innerEs = Executors.newFixedThreadPool((int) threads);
		List<Future<List<Double>>> futures = new ArrayList<Future<List<Double>>>();		
			
		final AtomicInteger max_its = new AtomicInteger(iterations);
		
		for ( Entry<List<Integer>, List<Integer>> innerCvEntry : innerCvList ) {
						
			futures.add(innerEs.submit(new Callable<List<Double>>() {
				@Override
				public List<Double> call() throws Exception {
					List<Integer> trainIdx = innerCvEntry.getKey();
					List<Integer> testIdx = innerCvEntry.getValue();
					
					List<double[]> xTrain = DataUtils.subset_rows(xArray,trainIdx);
					List<double[]> yTrain = DataUtils.subset_rows(yArray,trainIdx);
										
					List<double[]> xTest = DataUtils.subset_rows(xArray,testIdx);
					List<double[]> yTest = DataUtils.subset_rows(yArray,testIdx);
													
					DoubleMatrix W_train_test = W.get( DataUtils.toIntArray(trainIdx), DataUtils.toIntArray(testIdx)); // train to test
					
					if( fast_cv ) { 
						//assert false;
						List<Double> errors = buildGWANN(xTrain, yTrain, xTest, yTest, W_train_test, nrHidden, eta, opt, batchSize, max_its.get(), patience, kernel, bw, adaptive, lambda, explTrans, respTrans, init_weights, seed).errors;
						max_its.accumulateAndGet(errors.size(), Math::min);
						return errors;	
					} else {
						
						/*{
							double[] nbw = new double[9];
							for( int j = 0; j < nbw.length; j++ )
								nbw[j] = bw[0];
							List<Double> errors = buildGWANN(xTrain, yTrain, xTest, yTest, W_train_test, nrHidden, eta, opt, batchSize, iterations, patience, kernel, nbw, adaptive, lambda, explTrans, respTrans, null, seed).errors;
							log.debug("nbw: "+errors.subList(0, 20));
						}*/
						List<Double> errors = buildGWANN(xTrain, yTrain, xTest, yTest, W_train_test, nrHidden, eta, opt, batchSize, iterations, patience, kernel, bw, adaptive, lambda, explTrans, respTrans, init_weights, seed).errors;
						//log.debug(" bw "+errors.subList(0, 20));
						
						//System.exit(1);
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
								
	public static double[] getParamsWithGridSearch(int minRadius, int maxRadius, int steps,
			List<double[]> xArray, List<double[]> yArray, DoubleMatrix W, List<Entry<List<Integer>, List<Integer>>> innerCvList, GWKernel kernel, 
			boolean adaptive, double[] eta, int batchSize,	Optimizer opt,  
			int[] nrHidden, int iterations, int patience, int threads, double lambda, Transform[] expTrans, Transform[] respTrans, double[][][] init_weights, int seed ) {
		
		if( !adaptive )
			throw new RuntimeException("Not implemented yet");
				
		double[] bestF = null;
		int bestBw = -1;
		for( int i = (int)minRadius; i <= maxRadius; i+=(maxRadius-minRadius)/steps ) {
			log.debug("bw test: "+i);
			double[] f = NNetUtils.getErrorParameters( getErrors_CV(xArray, yArray, W, innerCvList, kernel, new double[] {i}, adaptive, eta, batchSize, opt, nrHidden, iterations, patience, threads, lambda, expTrans, respTrans, init_weights, seed), error_param.argmin_mean );
			if( bestF == null || f[0] < bestF[0] ) { 
				bestF = f;
				bestBw = i;
			}
		}
		return new double[] { bestF[0], bestBw, bestF[1]};
	}
	
	@Deprecated
	public static double[] getParamsWithGoldenSection(double minRadius, double maxRadius, 
			List<double[]> samples, DoubleMatrix W, List<Entry<List<Integer>, List<Integer>>> innerCvList, int[] fa, int ta, GWKernel kernel, boolean adaptive, double[] eta, int batchSize, 
			Optimizer opt,  
			int[] nrHidden, int iterations, int patience, int threads, double a, Transform[] expTrans, Transform[] respTrans, double[][][] init_weights, int seed ) {
		
		List<double[]> xTrain = new ArrayList<>();
		List<double[]> yTrain = new ArrayList<>();
		for (double[] d : samples ) {
			xTrain.add(DataUtils.strip(d, fa));
			yTrain.add( new double[] { d[ta] } );
		}
		return getParamsWithGoldenSection(minRadius,maxRadius,xTrain, yTrain, W, innerCvList, kernel, adaptive, eta, batchSize, opt, nrHidden, iterations, patience, threads, a, expTrans, respTrans, init_weights, seed);
	}
	
	@Deprecated
	public static double[] getParamsWithGridSearch(int minRadius, int maxRadius, int steps, 
			List<double[]> samples, DoubleMatrix W, List<Entry<List<Integer>, List<Integer>>> innerCvList, int[] fa, int ta, GWKernel kernel, boolean adaptive, double[] eta, int batchSize, 
			Optimizer opt, 
			int[] nrHidden, int iterations, int patience, int threads, double a, Transform[] expTrans, Transform[] respTrans, double[][][] init_weights, int seed ) {
		
		List<double[]> xTrain = new ArrayList<>();
		List<double[]> yTrain = new ArrayList<>();
		for (double[] d : samples ) {
			xTrain.add(DataUtils.strip(d, fa));
			yTrain.add( new double[] { d[ta] } );
		}
		return getParamsWithGridSearch(minRadius, maxRadius, steps, xTrain, yTrain, W, innerCvList, kernel, adaptive, eta, batchSize, opt, nrHidden, iterations, patience, threads, a, expTrans, respTrans, init_weights, seed);
	}
	
	protected static double[][][] calculateSpatiallyNormalizedGradient(List<double[][][]> gradient) {
				
	    if (gradient == null || gradient.isEmpty()) {
	        return new double[0][][];
	    }

	    double[][][] first = gradient.get(0);
	    double[][][] sum = new double[first.length][][];
	    for (int i = 0; i < first.length; i++) {
	        sum[i] = new double[first[i].length][];
	        for (int j = 0; j < first[i].length; j++)
	            sum[i][j] = new double[first[i][j].length];
	    }

	    // sum the per-sample gradients (sw is already inside each one)
	    for (double[][][] g : gradient)
	        for (int i = 0; i < sum.length; i++)
	            for (int j = 0; j < sum[i].length; j++) {
	                double[] s = sum[i][j];
	                double[] gr = g[i][j];
	                for (int k = 0; k < s.length; k++)
	                    s[k] += gr[k];
	            }

	    // one global divisor for every entry
	    double inv = 1.0 / gradient.size();
	    for (double[][] layer : sum)
	        for (double[] row : layer)
	            for (int k = 0; k < row.length; k++)
	                row[k] *= inv;

	    return sum;
	}
}
