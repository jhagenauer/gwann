package supervised.nnet;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Map.Entry;
import java.util.Random;

import supervised.SupervisedUtils;
import supervised.nnet.NNet.Optimizer;
import utils.ListNormalizer;
import utils.Normalizer.Transform;

public class NNet_RInterface {
		
	public static Return_R run(
			double[][] xArray_train, 
			double[] yArray_train, 
			
			double[][] xArray_pred,
			
			boolean norm,
			double nr_hidden, double batchSize, 
			String optim, 
			double eta_, 
			boolean linOut,						
			double iterations_,double cv_max_iterations,double cv_patience, double cv_folds_, double cv_repeats_,double permutations_,double threads, 
			int seed
		) {
				
		Optimizer opt;
		if( optim.equalsIgnoreCase("nesterov") )
			opt = Optimizer.Nesterov;
		else if( optim.equalsIgnoreCase("momentum"))
			opt = Optimizer.Momentum;
		else if( optim.equalsIgnoreCase("sgd"))
			opt = Optimizer.SGD;
		else if( optim.equalsIgnoreCase("adam_exp"))
			opt = Optimizer.Adam;
		else if( optim.equalsIgnoreCase("adam"))
			opt = Optimizer.Adam_ruder;
		else
			throw new RuntimeException("Unknown optimizer");

		int cv_folds = (int)cv_folds_, cv_repeats = (int)cv_repeats_, permutations = (int)permutations_, iterations = (int)iterations_;
		double[] eta = new double[] { eta_, eta_};
		
		Transform[] respTrans = norm ? new Transform[] {Transform.zScore} : new Transform[] {};
		Transform[] explTrans = norm ? new Transform[] {Transform.zScore} : new Transform[] {};
				
		List<Entry<List<Integer>, List<Integer>>> innerCvList = SupervisedUtils.getKFoldCVList( (int)cv_folds, (int)cv_repeats, xArray_train.length, new Random(seed) );
		
		List<double[]> xTrain_list = Arrays.asList(xArray_train);
		List<double[]> yTrain_list = new ArrayList<>();
		for( double d : yArray_train )
			yTrain_list.add( new double[] { d } );	
		
		List<double[]> xPred_list = Arrays.asList(xArray_pred);
		List<double[]> yPred_list = new ArrayList<>();
		/*for( double d : yArray_pred )
			yPred_list.add(d);*/
		for( int i = 0; i < xArray_pred.length; i++ )
			yPred_list.add( new double[] { Double.NaN } );
						
		int best_its = -1;			
		double bestValError = Double.POSITIVE_INFINITY;
		double lambda = 0.0;
						
		if( iterations > 0 ) { // bw given
			System.out.println("Iterations and bandwidth given. cv_max_iterations and cv_patience are ignored.");
			List<List<Double>> errors = NNetUtils.getErrors_CV(xTrain_list, yTrain_list, innerCvList, eta, (int)batchSize, opt, lambda, new int[] {(int)nr_hidden}, (int)iterations, (int)iterations, (int)threads, explTrans, respTrans);
						
			double mean = 0;
			for( List<Double> e : errors )
				mean += e.get( (int)iterations-1 )/errors.size();			
			bestValError = mean;
			best_its = (int)iterations;				
		
		} else {
			System.out.println("Pre-specified bandwidth...");
			List<List<Double>> errors = NNetUtils.getErrors_CV(xTrain_list, yTrain_list, innerCvList, eta, (int)batchSize, opt, lambda, new int[] {(int)nr_hidden}, (int)cv_max_iterations, (int)cv_patience, (int)threads, explTrans, respTrans);
			double[] m = NNetUtils.getBestErrorParams(errors);
			
			bestValError = m[0];
			best_its = (int)m[1]+1;
		} 
				
		System.out.println("Cross-validation results for hyperparameter search (folds: "+cv_folds+", repeats: "+cv_repeats+"):");
		if( iterations < 0 ) 
			System.out.println("* Iterations: " + best_its);
		System.out.println("* RMSE: " + bestValError);
		
		double[][][] imp = null;
		if( permutations > 0 ) { // importance
			System.out.println("Calculating feature importance...");
			ReturnObject bg = NNetUtils.buildNNet(
					xTrain_list, yTrain_list,  
					xTrain_list, yTrain_list,  
					new int[] { (int)nr_hidden }, eta, opt, 
					lambda, (int)batchSize, best_its, Integer.MAX_VALUE, 
					explTrans, respTrans);
						
			double[][] preds = bg.prediction.toArray(new double[][] {});
			
			List<double[]> xTrain_l = new ArrayList<>();
			for( double[] d : xTrain_list ) 
				xTrain_l.add(d);
			new ListNormalizer(explTrans, xTrain_l);
			
			List<double[]> yTrain_l = new ArrayList<>();
			for( double[] d : yTrain_list ) {
				double[] t = new double[yTrain_list.size()];
				for( int i = 0; i < t.length; i++ )
					t[i] = d[0];
			}
			ListNormalizer ln = new ListNormalizer(respTrans, yTrain_l);
									
			imp = new double[xArray_train[0].length][preds.length][preds[0].length];
			for( int i = 0; i < xArray_train[0].length; i++ ) { // for each variable
				System.out.println("Feature "+i);
												
				double[][] copy = new double[xTrain_l.size()][];
				for( int k = 0; k < copy.length; k++ )
					copy[k] = Arrays.copyOf(xTrain_l.get(k), xTrain_l.get(k).length);
								
				List<Double> l = new ArrayList<>();
				for( int k = 0; k < copy.length; k++ )
					l.add(copy[k][i]);
				
				for( int j = 0; j < permutations; j++ ) {	
					
					Collections.shuffle(l);
					for( int k = 0; k < l.size(); k++ )
						copy[k][i] = l.get(k);
					
					List<double[]> preds_ = new ArrayList<double[]>();
					for( double[] d : copy )
						preds_. add( bg.nnet.present( d ) );
					ln.denormalize(preds_);
					
					for( int k = 0; k < preds.length; k++ )
						for( int p = 0; p < preds[0].length; p++ )
							imp[i][k][p] += ( Math.pow(preds_.get(k)[p] - yArray_train[k],2) - Math.pow(preds[k][p] - yArray_train[k],2) )/permutations;
				}			
			}
		}

		System.out.println("Building final model with "+best_its+" iterations...");	
		long time = System.currentTimeMillis();
		ReturnObject tg = NNetUtils.buildNNet(
				xTrain_list, yTrain_list,  
				xPred_list, yPred_list,
				new int[] { (int)nr_hidden }, eta, opt, 
				lambda, (int)batchSize, best_its, Integer.MAX_VALUE, 
				explTrans, respTrans);
		
		double secs = (System.currentTimeMillis()-time)/1000;			
		Return_R rr = new Return_R();
		
		rr.predictions = tg.prediction.toArray( new double[][] {} );
		rr.importance = imp;
		rr.weights = tg.nnet.weights;
		rr.its = best_its;
		rr.bw = -1;
		rr.secs = secs;
		rr.ro = tg;
		return rr;			
	}
	
	public static double[][] predict( ReturnObject ro, double[][] xArray_pred ) {
		List<double[]> xPred_list = Arrays.asList(xArray_pred);
		List<double[]> l = ro.predict(xPred_list);
		
		double[][] d = new double[l.size()][];
		for( int i = 0; i < l.size(); i++ )
			d[i] = l.get(i);
		return d;
	}
}