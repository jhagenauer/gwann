package supervised;

import java.util.AbstractMap;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Map.Entry;
import java.util.Random;
import java.util.Set;

import org.apache.commons.math3.stat.correlation.PearsonsCorrelation;
import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.jblas.DoubleMatrix;
import org.jblas.Solve;
import org.jblas.exceptions.LapackException;

import supervised.nnet.NNet.Optimizer;
import utils.DataUtils;
import utils.MatrixNormalizer;

public class SupervisedUtils {

	private static Logger log = LogManager.getLogger(SupervisedUtils.class);
	
	public static List<Entry<List<Integer>, List<Integer>>> getKFoldCVList(int numFolds, int numRepeats, int numSamples ) {
		return getKFoldCVList(numFolds, numRepeats, numSamples, 0);
	}
		
	public static List<Integer> getIndicesWithNoNaN( List<double[]> samples, int ta ) {
		List<Integer> l = new ArrayList<>();
		for( int i = 0; i < samples.size(); i++ )
			if( !Double.isNaN( samples.get(i)[ta]  ) ) 
				l.add(i);
		return l;
	}


	public static List<Entry<List<Integer>, List<Integer>>> getKFoldCVList(int numFolds, int numRepeats, int numSamples, int seed ) {
		List<Integer> samplesIdx = new ArrayList<>();
		for (int i = 0; i < numSamples; i++)
			samplesIdx.add(i);
		return getKFoldCVList(numFolds, numRepeats, samplesIdx, seed);
	}
	
	public static List<Entry<List<Integer>, List<Integer>>> getKFoldCVList(int numFolds, int numRepeats, List<Integer> samplesIdx ) {
		return getKFoldCVList(numFolds, numRepeats, samplesIdx,0);
	}
	
	public static <T> List<Entry<List<T>, List<T>>> getKFoldCVList(int numFolds, int numRepeats, List<T> samples, int seed ) {
		Random r = new Random(seed);		
		List<Entry<List<T>, List<T>>> cvList = new ArrayList<Entry<List<T>, List<T>>>();
		for (int repeat = 0; repeat < numRepeats; repeat++) {
			
			List<T> l = new ArrayList<T>(samples);
			Collections.shuffle(l,r);
			
			double foldSize = (double)samples.size() / numFolds;
			for (int fold = 0; fold < numFolds; fold++) {
				int valStart = (int)Math.round(fold * foldSize);
				int valEnd = (int)Math.round( (fold + 1) * foldSize );
								
				List<T> val = new ArrayList<T>(l.subList( valStart, valEnd));
				List<T> train = new ArrayList<T>( l.subList(0, valStart ) );
				train.addAll( l.subList( valEnd, samples.size() ) );
				
				List<T> lt = new ArrayList<>(val);
				lt.retainAll(train);
				assert lt.isEmpty();
				
				cvList.add(new AbstractMap.SimpleEntry<List<T>, List<T>>(train, val));
			}
		}		
		return cvList;
	}
	
	@Deprecated
	public static List<Entry<List<Integer>, List<Integer>>> getCVList(int numTrainSamples, int numRepeats, int numSamples, int seed ) {
		List<Integer> samplesIdx = new ArrayList<>();
		for (int i = 0; i < numSamples; i++)
			samplesIdx.add(i);
		return getCVList(numTrainSamples, numRepeats, samplesIdx, seed);		
	}
	
	public static <T> List<Entry<List<T>, List<T>>> getCVList(int numTrainSamples, int numRepeats, List<T> samples, int seed ) {
		Random r = new Random(seed);				
		List<Entry<List<T>, List<T>>> cvList = new ArrayList<Entry<List<T>, List<T>>>();
		for (int repeat = 0; repeat < numRepeats; repeat++) {			
			List<T> train = new ArrayList<>(samples);			
			Collections.shuffle(train,r);
						
			List<T> val = new ArrayList<T>(train.subList(0, samples.size()-numTrainSamples));
			train.removeAll(val);
			cvList.add(new AbstractMap.SimpleEntry<List<T>, List<T>>(train, val));
		}		
		return cvList;
	}
	
	public static List<Entry<List<Integer>, List<Integer>>> getSplitCVList_sep(List<Integer> samples, List<Integer> subsamples_test, int size_test, int num_repeats, int seed ) {
		Random r = new Random(seed);				
		List<Entry<List<Integer>, List<Integer>>> cvList = new ArrayList<Entry<List<Integer>, List<Integer>>>();
		for (int repeat = 0; repeat < num_repeats; repeat++) {			
			List<Integer> train = new ArrayList<>(samples);			
			Collections.shuffle(train,r);
			
			List<Integer> test = new ArrayList<>(subsamples_test);
			Collections.shuffle(test,r);
			
			test = test.subList(0, size_test);
						
			train.removeAll(test);
			cvList.add(new AbstractMap.SimpleEntry<List<Integer>, List<Integer>>(train, test));
		}		
		return cvList;
	}
	
	public static double getRMSE(List<double[]> response, List<double[]> desired) {
		return Math.sqrt(getMSE(response, desired));
	}
	
	public static double getRMSE(List<Double> response, List<double[]> samples, int ta ) {
		return Math.sqrt(getMSE(response, samples, ta));
	}

	// Mean sum of squares
	@Deprecated
	public static double getMSE(List<double[]> response, List<double[]> desired) {
		if (response.size() != desired.size())
			throw new RuntimeException("response.size() != desired.size()");

		double mse = 0;
		for (int i = 0; i < response.size(); i++)
			mse += Math.pow(response.get(i)[0] - desired.get(i)[0], 2);
		return mse / response.size();
	}
	
	public static double getMSE(double[] response, double[] desired) {
		if (response.length != desired.length)
			throw new RuntimeException("response.length != desired.length");

		double mse = 0;
		for (int i = 0; i < response.length; i++)
			mse += Math.pow(response[i] - desired[i], 2);
		return mse / response.length;
	}
	
	public static double getRMSE(double[] response, double[] desired ) {
		return Math.sqrt(getMSE(response,desired));
	}
	
	public static double getMSE(List<Double> response, List<double[]> samples, int ta) {
		return getRSS(response, samples, ta) / response.size();
	}
	
	public static double getR2(List<Double> response, List<double[]> samples, int ta) {
		double[] r = new double[response.size()];
		for( int i = 0; i < response.size(); i++ )
			r[i] = response.get(i);
		
		double[] y = new double[samples.size()];
		for( int i = 0; i < samples.size(); i++ )
			y[i] = samples.get(i)[ta];
				
		return getR2(r,y);
	}
	
	public static double getR2(double[] response, double[] y ) {
		if (response.length != y.length)
			throw new RuntimeException("response size != samples size ("+response.length+"!="+y.length+")" );

		double ssr = 0;
		for (int i = 0; i < y.length; i++)
			ssr += Math.pow(y[i] - response[i], 2);
		
		double mean = 0;
		for (double d : y)
			mean += d;
		mean /= y.length;

		double varY = 0;
		for (int i = 0; i < y.length; i++)
			varY += Math.pow(y[i] - mean, 2);
		return 1.0 - ssr / varY;
	}
	
	public static double getAdjustedR2(double[] response, double[] y, int k ) {
		double r2 = getR2(response,y);
		int n = y.length;				
		return 1 - (1 - r2) * (n-1 ) / (n-k-1);
	}	
	
	public static double getR2_pearson(double[] response, double[] y ) {
		return Math.pow(getPearson(response, y), 2);
	}
	
	public static double getPearson(double[] response, double[] y ) {
		if (response.length != y.length)
			throw new RuntimeException("response size != samples size ("+response.length+"!="+y.length+")" );

		PearsonsCorrelation pc = new PearsonsCorrelation();
		return pc.correlation(response, y);
	}

	public static double getPearson(List<double[]> response, List<double[]> desired) {
		if (response.size() != desired.size())
			throw new RuntimeException();

		double meanDesired = 0;
		for (double[] d : desired)
			meanDesired += d[0];
		meanDesired /= desired.size();

		double meanResponse = 0;
		for (double[] d : response)
			meanResponse += d[0];
		meanResponse /= response.size();

		double a = 0;
		for (int i = 0; i < response.size(); i++)
			a += (response.get(i)[0] - meanResponse) * (desired.get(i)[0] - meanDesired);

		double b = 0;
		for (int i = 0; i < response.size(); i++)
			b += Math.pow(response.get(i)[0] - meanResponse, 2);
		b = Math.sqrt(b);

		double c = 0;
		for (int i = 0; i < desired.size(); i++)
			c += Math.pow(desired.get(i)[0] - meanDesired, 2);
		c = Math.sqrt(c);

		if (b == 0 || c == 0) // not sure about if this is ok
			return 0;

		return a / (b * c);
	}
	
	public static double getMultiLogLoss(List<Double> response, List<double[]> samples, int ta ) {
		List<double[]> desired = new ArrayList<double[]>();
		List<double[]> resp = new ArrayList<double[]>();
		for( int i = 0; i < samples.size(); i++ ) {
			resp.add( new double[]{ response.get(i) } );
			desired.add( new double[] { samples.get(i)[ta] } );
		}
		return getMultiLogLoss(resp, desired);
	}
	
	public static double getMultiLogLoss(List<double[]> response, List<double[]> desired ) {
		 double eps = Math.pow(10, -15);
		 double ll = 0;
		 for( int i = 0; i < response.size(); i++ ) 
			 for( int j = 0; j < response.get(i).length; j++ ) {
				 double a = desired.get(i)[j];
				 double p = Math.min(Math.max(eps, response.get(i)[j]), 1.0-eps);
				 ll += a*Math.log(p) + (1.0-a)*Math.log(1.0-p);
			 }
		 return ll * -1.0/desired.size();
	}

	public static double getAIC(double mse, double nrParams, int nrSamples) {
		return nrSamples * Math.log(mse) + 2 * nrParams;
	}
	
	public static double getAICc(double mse, double nrParams, int nrSamples) {
		if( nrSamples - nrParams - 1 <= 0 ) {
			log.error(nrSamples+","+nrParams);
			System.exit(1);
		}
		return getAIC(mse, nrParams, nrSamples) + (2.0 * nrParams * (nrParams + 1)) / (nrSamples - nrParams - 1);
	}
	
	// I don't know why, but that's how it is done in the GWMODEL package
	// dp.n*log(sigma.hat2) + dp.n*log(2*pi) +dp.n+tr.S
	public static double getAIC_GWMODEL(double mse, double nrParams, int nrSamples) {
		return nrSamples * Math.log(mse) + nrSamples * Math.log(2*Math.PI) + nrSamples + nrParams;
	}
		
	// I don't know why, but that's how it is done in the GWMODEL package
	// ##AICc = 	dev + 2.0 * (double)N * ( (double)MGlobal + 1.0) / ((double)N - (double)MGlobal - 2.0);
	// lm_AICc= dp.n*log(lm_RSS/dp.n)+dp.n*log(2*pi)+dp.n+2*dp.n*(var.n+1)/(dp.n-var.n-2)
	public static double getAICc_GWMODEL(double mse, double nrParams, int nrSamples) {
		if( nrSamples - nrParams - 2 <= 0 ) 
			throw new RuntimeException("too few samples! "+nrSamples+" "+nrParams);
		//return nrSamples * ( Math.log(mse) + Math.log(2*Math.PI) + 1 ) + (2.0 * nrSamples * ( nrParams + 1 ) ) / (nrSamples - nrParams - 2);
		return nrSamples * ( Math.log(mse) + Math.log(2*Math.PI) + ( nrSamples + nrParams) / (nrSamples - 2 - nrParams) );
	}

	public static double getBIC(double mse, double nrParams, int nrSamples) {
		return nrSamples * Math.log(mse) + nrParams * Math.log(nrSamples);
	}
	
	// TODO Check, not sure if correct
	public static double getAUC(List<double[]> props, List<double[]> desired ) {
		List<double[]> l = new ArrayList<>(props);
		Collections.sort(l, new Comparator<double[]>() {
			@Override
			public int compare(double[] o1, double[] o2) {
				return Double.compare(o1[0], o2[0]);
			}			
		});
		
		int posTotal = 0;
		for( double[] d : desired )
			if( d[0] == 1 )
				posTotal++;
		
		double auc = 0;
		double preP = 0;
		for( int i = 0; i < l.size(); i++ ) {
			double p = l.get(i)[0]; // <= p  is 0
			
			int posCor = 0;
			for( int j = i; j <= desired.size(); j++ )
				if( desired.get(j)[0] == 1 )
					posCor++;
			auc += (p-preP) * (double)posCor/posTotal; // width * height
			preP = p;
		}
		return auc;
	}
	
	public static void main(String[] args) {
		List<Entry<List<Integer>, List<Integer>>> cvList = getKFoldCVList(2,1,10,0);
		System.out.println(cvList);;
	}

	public static double getRSS(List<Double> response, List<double[]> samples, int ta) {
		if (response.size() != samples.size())
			throw new RuntimeException("response.size() != samples.size() "+response.size()+","+samples.size());

		double rss = 0;
		for (int i = 0; i < response.size(); i++)
			rss += Math.pow(response.get(i) - samples.get(i)[ta], 2);
		return rss;
	}

	public static int[] toIntArray(Collection<Integer> c) {
		int[] j = new int[c.size()];
		int i = 0;
		for (int l : c)
			j[i++] = l;
		return j;
	}
	
	public static double[] toDoubleArray(Collection<Double> c) {
		double[] j = new double[c.size()];
		int i = 0;
		for (double l : c)
			j[i++] = l;
		return j;
	}
	
	public static double getCost_CV(DoubleMatrix X, DoubleMatrix Y, List<Entry<List<Integer>, List<Integer>>> cv_list, double lambda ) {
		DescriptiveStatistics ds = new DescriptiveStatistics();		
		for (Entry<List<Integer>, List<Integer>> cv_entry : cv_list ) {
									
			List<Integer> outerTrainFinal = new ArrayList<>(cv_entry.getKey());
			List<Integer> outerTestFinal = new ArrayList<>(cv_entry.getValue());
							
			try {				
				int[] trainIdx = SupervisedUtils.toIntArray(outerTrainFinal);
				int[] testIdx = SupervisedUtils.toIntArray(outerTestFinal);
				
				DoubleMatrix XTrain = X.getRows(trainIdx);
				DoubleMatrix YTrain = Y.getRows(trainIdx);
				
				DoubleMatrix XTest = X.getRows(testIdx);
				DoubleMatrix YTest = Y.getRows(testIdx);
				double[] des = Arrays.copyOf( YTest.data, YTest.data.length);
				
				/*if( expTrans != null && expTrans.length > 0 ) {
					MatrixNormalizer mnx = new MatrixNormalizer(expTrans, XTrain, true);
					mnx.normalize(XTest);
				}*/
				
				MatrixNormalizer mny = null;
				/*if( respTrans != null && respTrans.length > 0 ) {
					mny = new MatrixNormalizer(respTrans, YTrain, false);
					mny.normalize(YTest);
				}*/
									
				DoubleMatrix Xt = XTrain.transpose();
				DoubleMatrix XtX = Xt.mmul(XTrain);
				DoubleMatrix beta;
				
				if( lambda > 0 ) { // ridge regression
					DoubleMatrix id = DoubleMatrix.eye(XtX.rows);			
					beta = Solve.solve(XtX.add(id.mul(lambda)), Xt.mmul(YTrain));
				} else 
					beta = Solve.solve(XtX, Xt.mmul(YTrain));
				
				
				DoubleMatrix P =  XTest.mmul(beta);
				if( mny != null )
					mny.denormalize(P);
				
				double rmse = SupervisedUtils.getRMSE(P.data, des);
				
				if( Double.isNaN(rmse) ) {

					synchronized (SupervisedUtils.class) {
						
						log.debug(X.rows+","+X.columns);
						log.debug("X: "+X);
						log.debug("beta: "+beta);
						log.debug( "p: "+Arrays.toString(P.data) );
						log.debug( "des: "+Arrays.toString(des) );

						System.exit(1);
					};					
				}
				ds.addValue(rmse);					
			} catch( LapackException e ) {		
				throw new RuntimeException(e.getMessage());
			}
		}					
		return ds.getMean();
	}
}