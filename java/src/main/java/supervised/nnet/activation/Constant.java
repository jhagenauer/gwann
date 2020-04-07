package supervised.nnet.activation;

public class Constant implements Function {
	private double c;
	public Constant( double x ) {
		this.c = x;
	}
	
	@Override
	public double f(double x) {
		return this.c;
	}

	@Override
	public double fDevFOut(double fOut) {
		return 0;
	}
	
	@Override
	public double fDev(double f) {
		return 0;
	}
}
