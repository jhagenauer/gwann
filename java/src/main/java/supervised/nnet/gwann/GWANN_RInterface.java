package supervised.nnet.gwann;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Map.Entry;
import java.util.Set;

import org.jblas.DoubleMatrix;

import supervised.SupervisedUtils;
import supervised.nnet.NNet.Optimizer;
import supervised.nnet.ReturnObject;
import utils.GWRUtils.GWKernel;
import utils.Normalizer.Transform;

public class GWANN_RInterface {
		
	public static Return_R run(
			double[][] xArray_train, double[] yArray_train, double[][] W_train,
			double[][] xArray_pred, double[] yArray_pred, double[][] W_train_pred,
			boolean norm,
			double nrHidden, double batchSize, String optim, double eta_, boolean linOut, 
			String krnl, double bw_, boolean adaptive, 
			String bwSearch, double bwMin, double bwMax, double steps_,
			double maxIts, double patience, 
			double folds, double repeats,
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

		double eps = 1e-03; // oder lieber 4?
		int pow = 3;
		final int steps = (int)steps_ < 0 ? 10 : (int)steps_;
		double[] eta = new double[] { eta_, eta_};
		
		Transform[] respTrans = new Transform[] {};
		Transform[] explTrans = norm ? new Transform[] {Transform.zScore} : new Transform[] {};
				
		DoubleMatrix W = new DoubleMatrix(W_train);
		List<Entry<List<Integer>, List<Integer>>> innerCvList = SupervisedUtils.getKFoldCVList( (int)folds, (int)repeats, xArray_train.length, seed);
		
		List<double[]> xTrain_list = Arrays.asList(xArray_train);
		List<Double> yTrain_list = new ArrayList<>();
		for( double d : yArray_train )
			yTrain_list.add(d);	
		
		List<double[]> xPred_list = Arrays.asList(xArray_pred);
		List<Double> yPred_list = new ArrayList<>();
		for( double d : yArray_pred )
			yPred_list.add(d);		
						
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
		double prevBestValError = Double.POSITIVE_INFINITY;
				
		if( bw_ > 0 ) {
			System.out.println("Pre-specified bandwidth...");
			double[] m = GWANNUtils.getParamsCV(xTrain_list, yTrain_list, W, innerCvList, kernel, bw_, adaptive, eta, (int)batchSize, opt, 0.0, new int[] {(int)nrHidden}, (int)maxIts, (int)patience, (int)threads, null, explTrans, respTrans );
			
			bestValError = m[0];
			bestValBw = bw_;
			bestIts = (int)m[1];
		} else if( bwSearch.equalsIgnoreCase("goldenSection") || bwSearch.equalsIgnoreCase("golden_section") ) { // determine best bw using golden section search 
			System.out.println("Golden section search...");
			double[] m = GWANNUtils.getParamsWithGoldenSection(min, max, xTrain_list, yTrain_list, W, innerCvList, kernel, adaptive, eta, (int)batchSize, opt, 0.0, new int[] {(int)nrHidden}, (int)maxIts, (int)patience, (int)threads, null, explTrans, respTrans);
						
			bestValError = m[0];
			bestValBw = m[1];
			bestIts = (int)m[2];
		} else { // determine best bw using grid search or local search routine 
			
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
				
				if( bwSearch.equalsIgnoreCase("grid") ) {
					ll.clear();
					for( double a = min; a <=max; a+= (max-min)/steps )
						ll.add(a);
					System.out.println("Grid search...");
				} else
					System.out.println("Local search routine...");
				ll.removeAll(bwDone);
	
				System.out.println(bwShrinkFactor + ", current best bandwidth: " + bestValBw + ", RMSE:" + bestValError + ", bandwidths to test: " + ll);
				for (double bw : ll) {				
					double[] mm = GWANNUtils.getParamsCV(xTrain_list, yTrain_list, W, innerCvList, kernel, bestValBw, adaptive, eta, (int)batchSize, opt, 0.0, new int[] {(int)nrHidden}, (int)maxIts, (int)patience, (int)threads, null, explTrans, respTrans);
					if (mm[0] < bestValError) {
						bestValError = mm[0];
						bestIts = (int)mm[1];
						bestValBw = bw;
					}
					bwDone.add(bw);
					System.out.println(bw+" "+Arrays.toString(mm));
				}
				if (prevBestValError - bestValError < eps || ll.isEmpty() )
					break;
				prevBestValError = bestValError;
			}
		}

		System.out.println("Cross-validation results (folds: "+folds+", repeats: "+repeats+"):");
		System.out.println("Bandwidth: " + bestValBw);
		System.out.println("Iterations: " + bestIts);
		System.out.println("RMSE: " + bestValError);
		
		double[][][] imp = null;
		if( permutations > 0 ) { // importance
			System.out.println("Calculating feature importance...");
			ReturnObject bg = GWANNUtils.buildGWANN(
					xTrain_list, yTrain_list, W, 
					xTrain_list, yTrain_list, W, 
					new int[] { (int)nrHidden }, eta, opt, 0.0, (int)batchSize, bestIts, Integer.MAX_VALUE, kernel, bestValBw, adaptive, null, explTrans, respTrans);
						
			double[][] preds = bg.prediction.toArray(new double[][] {});
									
			imp = new double[xArray_train[0].length][preds.length][preds[0].length];
			for( int i = 0; i < xArray_train[0].length; i++ ) { // for each variable
				System.out.println("Feature "+i);
				
				for( int j = 0; j < permutations; j++ ) {
					
					double[][] copy = Arrays.stream(xArray_train).map(double[]::clone).toArray(double[][]::new);
					List<Double> l = new ArrayList<>();
					for( int k = 0; k < copy.length; k++ )
						l.add(copy[k][i]);
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
				new int[] { (int)nrHidden }, eta, opt, 0.0, (int)batchSize, bestIts, Integer.MAX_VALUE, kernel, bestValBw, adaptive, null, explTrans, respTrans);
					
		Return_R ro = new Return_R();
		ro.predictions = tg.prediction.toArray( new double[][] {} );
		ro.importance = imp;
		ro.weights = tg.nnet.weights;
		ro.rmse = bestValError;
		ro.its = bestIts;
		ro.bw = bestValBw;
		return ro;			
	}
}