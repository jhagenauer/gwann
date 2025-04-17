package utils;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

import org.jblas.DoubleMatrix;

import dist.Dist;
import dist.EuclideanDist;

public class DataUtils {

	public static double[] strip(double[] d, int[] fa) {
		double[] nd = new double[fa.length];
		for (int j = 0; j < fa.length; j++)
			nd[j] = d[fa[j]];
		return nd;
	}

	public static int[] toIntArray(Collection<Integer> c) {
		int[] j = new int[c.size()];
		int i = 0;
		for (int l : c)
			j[i++] = l;
		return j;
	}
	
	public static DoubleMatrix getY(List<double[]> samples, int ta) {
		double[] y = new double[samples.size()];
		for (int i = 0; i < samples.size(); i++)
			y[i] = samples.get(i)[ta];
		return new DoubleMatrix(y);
	}	
	
	public static DoubleMatrix getX(List<double[]> samples, int[] fa, boolean addIntercept) {		
		double[][] x = new double[samples.size()][];
		for (int i = 0; i < samples.size(); i++) {
			double[] d = samples.get(i);
			
			x[i] = new double[fa.length + (addIntercept ? 1 : 0) ];
			x[i][x[i].length - 1] = 1.0; // gets overwritten if !addIntercept
			for (int j = 0; j < fa.length; j++) {
				x[i][j] = d[fa[j]];
			}
		}
				
		return new DoubleMatrix(x);
	}
	
	public static List<double[]> getStripped(List<double[]> a, int[] fa) {
		List<double[]> l = new ArrayList<>();
		for( double[] d : a )
			l.add( DataUtils.strip(d, fa) );
		return l;
	}
	
	public static DoubleMatrix getW(List<double[]> a, List<double[]> b, int[] ga ) {
		Dist<double[]> eDist = new EuclideanDist();
		DoubleMatrix W = new DoubleMatrix(a.size(), b.size());
		for (int i = 0; i < a.size(); i++)
			for (int j = 0; j < b.size(); j++)
				W.put(i, j, eDist.dist(
						new double[] { a.get(i)[ga[0]], a.get(i)[ga[1]] }, 
						new double[] { b.get(j)[ga[0]], b.get(j)[ga[1]] }
					));		
		return W;
	}
	
	public static <T> List<T> subset_row( List<T> l, int[] idx ) {
		List<T> r = new ArrayList<>();
		for( int i : idx )
			r.add(l.get(i));
		return r;
	}
	
	public static List<double[]> subset_row( List<double[]> l, List<Integer> li ) {
		return subset_row(l, toIntArray(li) );
	}
	
	public static List<double[]> subset_columns( List<double[]> x, int[] fa ) {
		List<double[]> r = new ArrayList<>();
		for( double[] d : x )
			r.add( DataUtils.strip(d, fa) );
		return r;
	}
}
