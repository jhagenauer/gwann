package utils;

import java.util.List;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.SingularValueDecomposition;
import org.apache.commons.math3.stat.descriptive.SummaryStatistics;

public class DataUtils {

	public static double[] strip(double[] d, int[] fa) {
		double[] nd = new double[fa.length];
		for (int j = 0; j < fa.length; j++)
			nd[j] = d[fa[j]];
		return nd;
	}
}
