package utils;

import java.util.List;

import org.apache.commons.math3.stat.descriptive.SummaryStatistics;

public class ListNormalizer extends Normalizer {
	
	int[] fa;
	
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
				
				assert !Double.isNaN( ds[i][j].getMean() );
				
			}
			
			// normalize
			for( int j = 0; j < fa.length; j++ )
				for (int k = 0; k < samples.size(); k++ )
					samples.get(k)[fa[j]] = normalize(samples.get(k)[fa[j]], i, j, false);			
		}
	}
	
	@Deprecated
	public ListNormalizer( Transform t, List<double[]> samples, int[] faa ) {
		this( new Transform[] {t},samples,faa);
	}
		
	public void normalize(List<double[]> samples ) {	
		for( int i = 0; i < tt.length; i++ ) 
			for( int j = 0; j < fa.length; j++ )
				for (int k = 0; k < samples.size(); k++ )
					samples.get(k)[fa[j]] = normalize(samples.get(k)[fa[j]], i, j, false);						
	}
	
	public void denormalize(List<double[]> samples ) {
		for( int i = tt.length-1; i >= 0 ; i-- ) 
			for( int j = 0; j < fa.length; j++ )
				for (int k = 0; k < samples.size(); k++ )
					samples.get(k)[fa[j]] = normalize(samples.get(k)[fa[j]], i, j, true);
	}
}
