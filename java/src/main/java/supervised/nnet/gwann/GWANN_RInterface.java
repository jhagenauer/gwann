package supervised.nnet.gwann;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Map.Entry;

import org.jblas.DoubleMatrix;

import supervised.SupervisedUtils;
import supervised.nnet.NNet.Optimizer;
import supervised.nnet.NNetUtils;
import supervised.nnet.ReturnObject;
import utils.GWUtils.GWKernel;
import utils.Normalizer.Transform;

public class GWANN_RInterface {
		
	public static Return_R run(
			double[][] xArray_train, double[] yArray_train, double[][] W_train,
			double[][] xArray_pred, /*double[] yArray_pred,*/ double[][] W_train_pred,
			boolean norm,
			double nrHidden, double batchSize, String optim, double eta_, boolean linOut, 
			String krnl, double bw_, boolean adaptive, 
			String bwSearch, double bwMin, double bwMax, double steps_,
			double iterations, 
			double cv_max_iterations,double cv_patience, double cv_folds, double cv_repeats,
			double permutations,
			double threads) {
		
		assert 
			xArray_train.length == W_train.length &  
			W_train.length == W_train[0].length & // quadratic
			W_train_pred.length == xArray_train.length & 
			W_train_pred[0].length == xArray_pred.length; 
		
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
		final int steps = (int)steps_ < 0 ? 10 : (int)steps_;
		double[] eta = new double[] { eta_, eta_};
		
		Transform[] respTrans = norm ? new Transform[] {Transform.zScore} : new Transform[] {};
		Transform[] explTrans = norm ? new Transform[] {Transform.zScore} : new Transform[] {};
				
		DoubleMatrix W = new DoubleMatrix(W_train);
		List<Entry<List<Integer>, List<Integer>>> innerCvList = SupervisedUtils.getKFoldCVList( (int)cv_folds, (int)cv_repeats, xArray_train.length, seed);
		
		List<double[]> xTrain_list = Arrays.asList(xArray_train);
		List<Double> yTrain_list = new ArrayList<>();
		for( double d : yArray_train )
			yTrain_list.add(d);	
		
		List<double[]> xPred_list = Arrays.asList(xArray_pred);
		List<Double> yPred_list = new ArrayList<>();
		/*for( double d : yArray_pred )
			yPred_list.add(d);*/
		for( int i = 0; i < xArray_pred.length; i++ )
			yPred_list.add( Double.NaN );
						
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
						
		if( bw_ > 0 && iterations > 0 ) { // bw and its are given
			System.out.println("Iterations and bandwidth given. cv_max_iterations and cv_patience are ignored.");
			List<List<Double>> errors = GWANNUtils.getErrors_CV(xTrain_list, yTrain_list, W, innerCvList, kernel, bw_, adaptive, eta, (int)batchSize, opt, 0.0, new int[] {(int)nrHidden}, (int)iterations, (int)iterations, (int)threads, null, -1, explTrans, respTrans );
			double mean = 0;
			for( List<Double> e : errors )
				mean += e.get( (int)iterations-1 )/errors.size();			
			bestValError = mean;
			bestValBw = bw_;
			bestIts = (int)iterations;			
		} else if( bw_ > 0 && iterations < 0 ) {
			System.out.println("Pre-specified bandwidth...");
			List<List<Double>> errors = GWANNUtils.getErrors_CV(xTrain_list, yTrain_list, W, innerCvList, kernel, bw_, adaptive, eta, (int)batchSize, opt, 0.0, new int[] {(int)nrHidden}, (int)cv_max_iterations, (int)cv_patience, (int)threads, null, -1, explTrans, respTrans );
			double[] m = NNetUtils.getBestErrorParams(errors);
			
			bestValError = m[0];
			bestValBw = bw_;
			bestIts = (int)m[1]+1;
		} else if( ( bwSearch.equalsIgnoreCase("goldenSection") || bwSearch.equalsIgnoreCase("golden_section") ) && iterations < 0 ) { // determine best bw using golden section search 
			System.out.println("Golden section search...");
			double[] m = GWANNUtils.getParamsWithGoldenSection(min, max, xTrain_list, yTrain_list, W, innerCvList, kernel, adaptive, eta, (int)batchSize, opt, 0.0, new int[] {(int)nrHidden}, (int)cv_max_iterations, (int)cv_patience, (int)threads, null, -1, explTrans, respTrans);
						
			bestValError = m[0];
			bestValBw = m[1];
			bestIts = (int)m[2]+1;
		} else if( iterations < 0 ){ // determine best bw using grid search or local search routine 			
			System.out.println("Grid search...");
			
			List<Double> ll = new ArrayList<>();			
			for( double a = min; a <=max; a+= (max-min)/steps )
				if (adaptive)
					ll.add((double) Math.round(a));
				else
					ll.add(a);
					
			ll = new ArrayList<Double>( new HashSet<Double>(ll) ); // remove duplicates
			Collections.sort(ll);
			System.out.println("To test: "+ll);
			for (double bw : ll) {				
				List<List<Double>> errors = GWANNUtils.getErrors_CV(xTrain_list, yTrain_list, W, innerCvList, kernel, bw, adaptive, eta, (int)batchSize, opt, 0.0, new int[] {(int)nrHidden}, (int)cv_max_iterations, (int)cv_patience, (int)threads, null, -1, explTrans, respTrans);
				
				int max_it = 0;
				for( List<Double> e : errors)
					max_it = Math.max(e.size(),max_it);
				
				double[] mm = NNetUtils.getBestErrorParams(errors);
				if (mm[0] < bestValError) {
					bestValError = mm[0];
					bestIts = (int)mm[1]+1;
					bestValBw = bw;
				}
				System.out.println(bw+" "+Arrays.toString(mm)+","+max_it);
			}			
		} else 
			throw new RuntimeException("Combination of bandwith/iterations not implemented yet!");
		
		System.out.println("Cross-validation results for hyperparameter search (folds: "+cv_folds+", repeats: "+cv_repeats+"):");
		if( bw_ > 0 ) System.out.println("\tBandwidth: " + bestValBw);
		if( iterations > 0 ) System.out.println("\tIterations: " + bestIts);
		System.out.println("\tRMSE: " + bestValError);
		
		double[][][] imp = null;
		if( permutations > 0 ) { // importance
			System.out.println("Calculating feature importance...");
			ReturnObject bg = GWANNUtils.buildGWANN(
					xTrain_list, yTrain_list, W, 
					xTrain_list, yTrain_list, W, 
					new int[] { (int)nrHidden }, eta, opt, 0.0, (int)batchSize, bestIts, Integer.MAX_VALUE, kernel, bestValBw, adaptive, null, -1, explTrans, respTrans,0);
						
			double[][] preds = bg.prediction.toArray(new double[][] {});
									
			imp = new double[xArray_train[0].length][preds.length][preds[0].length];
			for( int i = 0; i < xArray_train[0].length; i++ ) { // for each variable
				System.out.println("Feature "+i);
				
				double[][] copy = Arrays.stream(xArray_train).map(double[]::clone).toArray(double[][]::new);
				List<Double> l = new ArrayList<>();
				for( int k = 0; k < copy.length; k++ )
					l.add(copy[k][i]);
				
				for( int j = 0; j < permutations; j++ ) {					
					Collections.shuffle(l);
					for( int k = 0; k < l.size(); k++ )
						copy[k][i] = l.get(k);
					
					double[][] preds_ = new double[copy.length][];
					for( int k = 0; k < copy.length; k++ )
						preds_[k] = bg.nnet.present(copy[k]);
					
					for( int k = 0; k < preds.length; k++ )
						for( int p = 0; p < preds[0].length; p++ )
							imp[i][k][p] += ( Math.pow(preds_[k][p] - yArray_train[k],2) - Math.pow(preds[k][p] - yArray_train[k],2) )/permutations;
				}			
			}
		}

		System.out.println("Building final model with bandwidth "+bestValBw+" and "+bestIts+" iterations...");					
		ReturnObject tg = GWANNUtils.buildGWANN(
				xTrain_list, yTrain_list, W, 
				xPred_list, yPred_list, new DoubleMatrix(W_train_pred), 
				new int[] { (int)nrHidden }, eta, opt, 0.0, (int)batchSize, bestIts, Integer.MAX_VALUE, kernel, bestValBw, adaptive, null, -1, explTrans, respTrans,0);
					
		Return_R ro = new Return_R();
		
		ro.predictions = tg.prediction_denormed.toArray( new double[][] {} );
		ro.importance = imp;
		ro.weights = tg.nnet.weights;
		//ro.rmse = bestValError;
		ro.its = bestIts;
		ro.bw = bestValBw;
		return ro;			
	}
}