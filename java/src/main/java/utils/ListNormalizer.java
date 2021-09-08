package utils;

import java.util.List;

import org.apache.commons.math3.stat.descriptive.SummaryStatistics;

public class ListNormalizer extends Normalizer {
	
	int[] fa;
	Transform[] tt;
	SummaryStatistics[][] ds;
	
	public ListNormalizer( Transform t, List<double[]> samples ) {
		this( new Transform[] {t}, samples, null);
	}
	
	public ListNormalizer( Transform[] t, List<double[]> samples ) {
		this( t, samples, null );
	}
	
	public ListNormalizer( Transform[] ttt, List<double[]> samples, int[] faa ) {
		if( faa == null ) {
			this.fa = new int[samples.get(0).length];
			for( int i = 0; i < this.fa.length; i++ )
				this.fa[i] = i;
		} else
			this.fa = faa;
		this.tt = ttt;
		
		this.ds = new SummaryStatistics[tt.length][fa.length];
		for( int i = 0; i < tt.length; i++ ) {
			// calculate summary statistics before applying t
			for( int j = 0; j < fa.length; j++ ) {
				ds[i][j] = new SummaryStatistics();
				for( double[] d : samples )
					ds[i][j].addValue(d[j]);			
			}
			
			// normalize
			normalize(samples, i);			
		}
	}
	
	@Deprecated
	public ListNormalizer( Transform t, List<double[]> samples, int[] faa ) {
		this( new Transform[] {t},samples,faa);
	}
	
	public void normalize(List<double[]> samples) {	
		for( int i = 0; i < tt.length; i++ ) 
			normalize(samples, i );								
	}
	
	private void normalize(List<double[]> samples, int i ) {
		Transform t = tt[i];
		for( int j = 0; j < fa.length; j++ )
			for (double[] d : samples ) { 
				if (t == Transform.zScore) 
					d[fa[j]] = (d[fa[j]] - ds[i][j].getMean()) / ds[i][j].getStandardDeviation();
				else if (t == Transform.scale01)
					d[fa[j]] = (d[fa[j]] - ds[i][j].getMin()) / (ds[i][j].getMax() - ds[i][j].getMin());
				else if( t == Transform.sqrt )
					d[fa[j]] = Math.sqrt(d[fa[j]]);
				else if( t == Transform.log )
					d[fa[j]] = Math.log(d[fa[j]]);
				else if( t == Transform.log1 )
					d[fa[j]] = Math.log( d[fa[j]]+1.0 );
				else if( t == Transform.none )
					d[fa[j]] = d[fa[j]];					
				else
					throw new RuntimeException(t+" not supported!");							
			}		
	}
}
