package utils;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.jblas.DoubleMatrix;

import dist.Dist;

public class GWRUtils {

	public static Map<double[], Double> getBandwidth(List<double[]> samples, Dist<double[]> dist, double bw, boolean adaptive) {
		Map<double[], Double> bandwidth = new HashMap<>();
		for (double[] a : samples) {
			if (!adaptive)
				bandwidth.put(a, bw);
			else {
				int k = (int) bw;
				List<double[]> s = new ArrayList<>(samples);
				Collections.sort(s, new Comparator<double[]>() {
					@Override
					public int compare(double[] o1, double[] o2) {
						return Double.compare(dist.dist(o1, a), dist.dist(o2, a));
					}
				});
				bandwidth.put(a, dist.dist(s.get(k - 1), a));
			}
		}
		return bandwidth;
	}

	public enum GWKernel {
		gaussian, bisquare, boxcar, exponential, tricube
	};

	public static double getKernelValue(GWKernel k, double dist, double bw) {	
		double w;
		if (k == GWKernel.gaussian)
			w = Math.exp(-0.5 * Math.pow(dist / bw, 2));
		else if (k == GWKernel.bisquare)
			w = dist <= bw ? Math.pow(1.0 - Math.pow(dist / bw, 2), 2) : 0;
		else if (k == GWKernel.boxcar)
			w = dist <= bw ? 1 : 0;
		else if( k == GWKernel.tricube )
			w = dist <= bw ? Math.pow(1.0 - Math.pow(dist / bw, 3), 3) : 0;
		else if( k == GWKernel.exponential )
			w = Math.exp(-dist / bw);
		else
			throw new RuntimeException("No valid kernel given");
		if (Double.isNaN(w))
			throw new RuntimeException("NaN kernel value " + dist + ", " + bw + ", " + w);
		return w;
	}
	
	public static DoubleMatrix getGWMean(DoubleMatrix X, DoubleMatrix W, GWKernel k, double bw) {						
		DoubleMatrix W_ = GWRUtils.applyKernel(W, bw, k);
		DoubleMatrix a = new DoubleMatrix(W_.rows,X.columns);
		
		for( int j = 0; j < W_.rows; j++ ) { // for each location
			for( int i = 0; i < X.rows; i++ ) {
				a.putRow(j, a.getRow(j).add( X.getRow(i).mul( W_.get(j,i) ) ) ); 
			}
			a.putRow(j, a.getRow(j).div(W_.getRow(j).sum() ) );
		}
		return a;
	}
		
	public static DoubleMatrix getGWSD( DoubleMatrix X, DoubleMatrix W, GWKernel k, double bw ) {
		DoubleMatrix mean = getGWMean(X, W, k, bw);
		DoubleMatrix W_ = GWRUtils.applyKernel(W, bw, k);
		
		DoubleMatrix a = new DoubleMatrix(W_.rows,X.columns);
		for( int j = 0; j < W_.rows; j++ ) {
			for( int i = 0; i < X.rows; i++ ) {
				DoubleMatrix s = mean.getRow(j).sub( X.getRow(i));
				s.muli(s).muli( W_.get(j,i));
				a.putRow(j, a.getRow(j).add(s) );
			}
			for( int i = 0; i < a.getRow(j).columns; i++ )
				a.put(j,i, Math.sqrt( a.get(j,i)/W_.getRow(j).sum() ) );
		}
		return a;
	}
	
	public static DoubleMatrix getKernelWeights(DoubleMatrix Wtrain, DoubleMatrix Wk, GWKernel kernel, int nb ) {
		DoubleMatrix kW = new DoubleMatrix(Wk.rows,Wk.columns);
		for (int i = 0; i < Wk.rows; i++) {
			double bw = Wk.getRow(i).sort().get(nb);
			double[] w = new double[Wk.columns];
			for (int j = 0; j < w.length; j++)
				w[j] = GWRUtils.getKernelValue(kernel, Wk.get(i, j), bw);
			kW.putRow(i,new DoubleMatrix(w));
		}
		return kW;
	}
	
	public static DoubleMatrix getKernelWeights( DoubleMatrix W, GWKernel kernel, double bw ) {
		DoubleMatrix kW = new DoubleMatrix(W.rows,W.columns);
		for (int i = 0; i < W.rows; i++) {
			double[] w = new double[W.columns];
			for (int j = 0; j < w.length; j++)
				w[j] = GWRUtils.getKernelValue(kernel, W.get(i,j), bw);
			kW.putRow(i, new DoubleMatrix(w));
		}
		return kW;
	}

	// ----------------------------------------------

	public static DoubleMatrix applyKernel(DoubleMatrix W, double bw, GWKernel kernel) {
		double[] d = new double[W.length];
		for (int i = 0; i < d.length; i++)
			d[i] = getKernelValue(kernel, W.get(i), bw);
		return new DoubleMatrix(W.rows,W.columns,d);
	}
}
