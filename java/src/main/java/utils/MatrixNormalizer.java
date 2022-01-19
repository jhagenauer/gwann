package utils;

import org.apache.commons.math3.stat.descriptive.SummaryStatistics;
import org.jblas.DoubleMatrix;

public class MatrixNormalizer extends Normalizer {
	
	public MatrixNormalizer( Transform[] tt, DoubleMatrix X, boolean lastIc ) {
		this.tt = tt;
		
		this.ds = new SummaryStatistics[tt.length][X.columns - (lastIc ? 1 : 0)];
		for( int i = 0; i < tt.length; i++ ) {
			// calculate summary statistics before applying t
			for( int j = 0; j < ds[i].length; j++ ) {
				ds[i][j] = new SummaryStatistics();
				for( int k = 0; k < X.rows; k++  )
					ds[i][j].addValue(X.get(k,j));			
			}
			
			// normalize, affects subsequent summary statistics
			for( int j = 0; j < ds[i].length; j++ )
				for (int k = 0; k < X.rows; k++ ) 
					X.put(k,j,normalize(X.get(k,j), i, j, false));				
		}
	}
	
	public void normalize(DoubleMatrix X) {	
		for( int i = 0; i < tt.length; i++ ) 
			for( int j = 0; j < ds[i].length; j++ )
				for (int k = 0; k < X.rows; k++ ) 
					X.put(k,j,normalize(X.get(k,j), i, j, false));								
	}
	
	public void denormalize(DoubleMatrix X) {
		for( int i = tt.length-1; i >= 0 ; i-- ) 
			for( int j = 0; j < ds[i].length; j++ )
				for (int k = 0; k < X.rows; k++ ) 
					X.put(k,j,normalize(X.get(k,j), i, j, true));			
	}
}
