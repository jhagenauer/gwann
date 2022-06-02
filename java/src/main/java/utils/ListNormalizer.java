package utils;

import java.util.List;

import org.apache.commons.math3.stat.descriptive.SummaryStatistics;

public class ListNormalizer extends Normalizer {
	
	int length = -1;
		
	public ListNormalizer( Transform[] ttt, List<double[]> samples ) {
		length = samples.get(0).length;
		this.tt = ttt;
		
		this.ds = new SummaryStatistics[tt.length][length];
		for( int i = 0; i < tt.length; i++ ) {
			
			// calculate summary statistics before applying t
			for( int j = 0; j < length; j++ ) {
				ds[i][j] = new SummaryStatistics();
				for( double[] d : samples )
					ds[i][j].addValue(d[j]);
				
				assert !Double.isNaN( ds[i][j].getMean() );
				
			}
			
			// normalize
			for( int j = 0; j < length; j++ )
				for (int k = 0; k < samples.size(); k++ )
					samples.get(k)[j] = normalize(samples.get(k)[j], i, j, false);			
		}
	}
			
	public void normalize(List<double[]> samples ) {	
		for( int i = 0; i < tt.length; i++ ) 
			for( int j = 0; j < length; j++ )
				for (int k = 0; k < samples.size(); k++ )
					samples.get(k)[j] = normalize(samples.get(k)[j], i, j, false);						
	}
		
	public void denormalize(List<double[]> samples ) {
		for( int i = tt.length-1; i >= 0 ; i-- ) 
			for( int j = 0; j < length; j++ )
				for (int k = 0; k < samples.size(); k++ )
					samples.get(k)[j] = normalize(samples.get(k)[j], i, j, true);
	}
	
	// denormalize column fa[j] of sample d
	public void denormalize(double[] d, int j ) {
		for( int i = tt.length-1; i >= 0 ; i-- ) 
			d[j] = normalize(d[j], i, j, true);
	}
}
