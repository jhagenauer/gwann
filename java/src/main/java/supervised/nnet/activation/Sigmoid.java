package supervised.nnet.activation;

public class Sigmoid implements Function { // aka logistic function
	@Override
	public double f(double x) {
		/*if( x > 16 )
			return 1.0;
		else if( x < -16 )
			return 0.0;
		else*/
			return 1.0/(1.0+Math.exp(-x));
	}
	
	@Override
	public double fDevFOut(double fout ) {
		return fout * (1.0 - fout);
	}
	
	@Override
	public double fDev(double x ) {
		double fxi = f(x);
		return fxi * (1.0 - fxi);
	}
}
