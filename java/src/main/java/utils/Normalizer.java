package utils;

import java.util.List;

import org.apache.commons.math3.stat.descriptive.SummaryStatistics;

public class Normalizer {
	
	public enum Transform {
		pow2, log, sqrt, div, zScore, scale01, pca
	};
	
	int[] fa;
	Transform t;
	SummaryStatistics[] ds;
	
	public Normalizer( Transform t, List<double[]> samples ) {
		this(t,samples,null);
	}
	
	public Normalizer( Transform t, List<double[]> samples, int[] faa ) {
		if( faa == null ) {
			this.fa = new int[samples.get(0).length];
			for( int i = 0; i < this.fa.length; i++ )
				this.fa[i] = i;
		} else
			this.fa = faa;
		this.t = t;
		
		ds = new SummaryStatistics[fa.length];
		for (int i = 0; i < fa.length; i++)
			ds[i] = new SummaryStatistics();
		for (double[] d : samples)
			for (int i = 0; i < fa.length; i++)
				ds[i].addValue(d[fa[i]]);
		
		normalize(samples);
	}
	
	public void normalize(List<double[]> samples) {
		for (int i = 0; i < samples.size(); i++) {
			double[] d = samples.get(i);
			for (int j = 0; j < fa.length; j++) {
				if (t == Transform.zScore)
					d[fa[j]] = (d[fa[j]] - ds[j].getMean()) / ds[j].getStandardDeviation();
				else if (t == Transform.scale01)
					d[fa[j]] = (d[fa[j]] - ds[j].getMin()) / (ds[j].getMax() - ds[j].getMin());
				else
					throw new RuntimeException(t+" not supported!");
			}
		}
	}
}
