package utils;

import java.util.Arrays;
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
				
				/*if( Double.isNaN(ds[i][j].getMean())) {
					System.out.println(Arrays.toString(ttt)+","+i+","+j+","+samples.get(0).length);
					for( int k = 0; k < samples.size(); k++ ) {
						double[] d = samples.get(k);
						System.out.println(k+":"+d[j]);
					}
					System.exit(1);
				}*/							
				assert !Double.isNaN( ds[i][j].getMean() ) : "NaN. Transform index: " + i + ", var index :" + j+", size: "+ds[i][j].getN()+", var: "+ds[i][j].getVariance();				
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
	public void denormalize(double[] d, int col ) {
		for( int i = tt.length-1; i >= 0 ; i-- ) 
			d[col] = normalize(d[col], i, col, true);
	}
}
