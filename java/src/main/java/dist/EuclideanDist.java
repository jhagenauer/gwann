package dist;

import java.util.Arrays;

public class EuclideanDist implements Dist<double[]> {
	private int[] idx;

	public EuclideanDist() {
		this.idx = null;
	}

	public EuclideanDist(int[] idx) {
		this.idx = idx;
	}

	@Override
	public double dist(double[] a, double[] b) {
		if( idx != null )
			return dist( a, 0, b, 0, idx );
		else
			return dist(a, 0, b, 0, a.length);
	}
	
	// shortcut
	public double dist(double[] a, double[] b, int[] idx ) {
		return dist(a,0,b,0,idx);
	}
	
	// shortcut
	public double dist(double[] a, int offsetA, double[] b, int offsetB) {
		if( idx != null )
			return dist(a,offsetA,b,offsetB, idx );
		else
			return dist(a,offsetA,b,offsetB,Math.min(a.length-offsetA, b.length-offsetB) );
	}
	
	public double dist(double[] a, int offsetA, double[] b, int offsetB, int[] idx ) {
		double dist = 0;
		for (int i : idx )
			dist += (a[i + offsetA] - b[i + offsetB])*(a[i + offsetA] - b[i + offsetB]);
		return Math.sqrt( dist );
	}

	public double dist(double[] a, int offsetA, double[] b, int offsetB, int length) {
		double dist = 0;
		for (int i = 0; i < length; i++)
			dist += (a[i + offsetA] - b[i + offsetB])*(a[i + offsetA] - b[i + offsetB]);
		return Math.sqrt( dist );
	}
	
	@Override 
	public String toString() {
		return Arrays.toString(idx);
	}
}
