package supervised.nnet.activation;

public class Linear implements Function {
	@Override
	public double f(double x) {
		return x;
	}

	@Override
	public double fDevFOut(double fOut) {
		return 1.0;
	}
	
	public double fDev(double x ) {
		return 1.0;
	}

}
