package supervised.nnet;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import utils.ListNormalizer;

public class ReturnObject {
	public List<Double> errors;
	public NNet nnet;
	public double rmse;
	public double r2;
	public List<double[]> prediction;
	public ListNormalizer ln_x, ln_y;
	
	public List<double[]> predict( List<double[]> x ) {
		
		List<double[]> nx = new ArrayList<>();
		for( double[] d : x )
			nx.add( Arrays.copyOf(d, d.length));
		ln_x.normalize(nx);
		
		List<double[]> response = new ArrayList<>();
		for( double[] d : nx )
			response.add( nnet.present(d) );
		
		ln_y.denormalize(response);
		return(response);		
	}
}
