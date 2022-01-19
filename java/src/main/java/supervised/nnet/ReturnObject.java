package supervised.nnet;

import java.util.List;

public class ReturnObject {
	public List<Double> errors;
	public NNet nnet;
	public double rmse;
	public double r2;
	public List<double[]> prediction;
	public List<double[]> prediction_denormed;
}
