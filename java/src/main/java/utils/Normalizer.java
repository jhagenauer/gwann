package utils;

import org.apache.commons.math3.stat.descriptive.SummaryStatistics;

public abstract class Normalizer {

	public enum Transform {
		pow2, log, log1, sqrt, div, zScore, scale01, pca, none
	}

	Transform[] tt;
	SummaryStatistics[][] ds;

	protected double normalize(double d, int i, int col, boolean inv) {
		Transform t = tt[i];

		double nd;
		if (t == Transform.zScore) {
			double sd = ds[i][col].getStandardDeviation();
			assert !Double.isNaN(sd);
			
			if (!inv)
				nd = (d - ds[i][col].getMean()) / sd;
			else
				nd = d * sd + ds[i][col].getMean();
		} else if (t == Transform.scale01)
			if (!inv)
				nd = (d - ds[i][col].getMin()) / (ds[i][col].getMax() - ds[i][col].getMin());
			else
				nd = d * (ds[i][col].getMax() - ds[i][col].getMin()) + ds[i][col].getMin();
		else if (t == Transform.sqrt)
			if (!inv)
				nd = Math.sqrt(d);
			else
				nd = Math.pow(d, 2);
		else if (t == Transform.log)
			if (!inv)
				nd = Math.log(d);
			else
				nd = Math.exp(d);
		else if (t == Transform.log1)
			if (!inv)
				nd = Math.log(d + 1.0);
			else
				nd =  Math.exp(d) - 1;
		else if (t == Transform.none)
			nd = d;
		else
			throw new RuntimeException(t + " not supported!");
		return nd;
	}
}
