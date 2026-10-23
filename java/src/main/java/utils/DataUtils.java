package utils;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.jblas.DoubleMatrix;

import dist.Dist;
import dist.EuclideanDist;

public class DataUtils {
	
	private static Logger log = LogManager.getLogger(DataUtils.class);

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
	
	public static List<double[]> subset_rows( List<double[]> l, List<Integer> li ) {
		return subset_row(l, toIntArray(li) );
	}
	
	public static List<double[]> subset_columns( List<double[]> x, int[] fa ) {
		List<double[]> r = new ArrayList<>();
		for( double[] d : x )
			r.add( DataUtils.strip(d, fa) );
		return r;
	}

	public static void rowNormalizeMatrix(DoubleMatrix kW) {
	    for (int i = 0; i < kW.rows; i++) {
	        DoubleMatrix row = kW.getRow(i);
	        double sum = row.sum();
	
	        if( sum != 0.0 )
	        	kW.putRow(i, row.divi(sum)); 
	        else
	        	kW.putRow(i, row );
	    }
	}
	
	public static boolean warn_once = true;
	
	public static void columnNormalizeMatrix(DoubleMatrix kW) {
	    for (int i = 0; i < kW.columns; i++) {
	        DoubleMatrix column = kW.getColumn(i);
	        double sum = column.sum();
	
	        if (sum == 0.0 ) {
	        	if( warn_once ) {
	        		log.warn("Collumn " + i + " sums to zero; cannot normalize. Not further warnings!");
	        		warn_once = false;
	        	}
	        } else
	        	kW.putColumn(i, column.divi(sum)); // divi = in-place divide, avoids extra allocation
	    }
	}
	
	public static void saveToCSV(DoubleMatrix matrix, String filePath) {
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(filePath))) {
            for (int i = 0; i < matrix.rows; i++) {
                StringBuilder rowString = new StringBuilder();
                for (int j = 0; j < matrix.columns; j++) {
                    rowString.append(matrix.get(i, j));
                    
                    // Add comma if it's not the last element in the row
                    if (j < matrix.columns - 1) {
                        rowString.append(",");
                    }
                }
                writer.write(rowString.toString());
                writer.newLine();
            }
            System.out.println("Matrix successfully saved to: " + filePath);
        } catch (IOException e) {
            System.err.println("Error writing matrix to CSV: " + e.getMessage());
        }
	}
}
