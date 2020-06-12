package supervised.nnet.activation;

public class ReLu implements Function {
	@Override
	public double f(double x) {
		return x < 0 ? 0 : x;
	}
	
	@Override
	public double fDevFOut(double fOut ) {
		return fOut < 0 ? 0 : 1;
	}
	
	@Override
	public double fDev(double f) {
		throw new RuntimeException("Not implemented yet!");
	}
}
