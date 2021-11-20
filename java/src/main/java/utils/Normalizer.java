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

		if (t == Transform.zScore)
			if (!inv)
				return (d - ds[i][col].getMean()) / ds[i][col].getStandardDeviation();
			else
				return d * ds[i][col].getStandardDeviation() + ds[i][col].getMean();
		else if (t == Transform.scale01)
			if (!inv)
				return (d - ds[i][col].getMin()) / (ds[i][col].getMax() - ds[i][col].getMin());
			else
				return d * (ds[i][col].getMax() - ds[i][col].getMin()) + ds[i][col].getMin();
		else if (t == Transform.sqrt)
			if (!inv)
				return Math.sqrt(d);
			else
				return Math.pow(d, 2);
		else if (t == Transform.log)
			if (!inv)
				return Math.log(d);
			else
				return Math.exp(d);
		else if (t == Transform.log1)
			if (!inv)
				return Math.log(d + 1.0);
			else
				return Math.exp(d) - 1;
		else if (t == Transform.none)
			return d;
		else
			throw new RuntimeException(t + " not supported!");
	}
}
