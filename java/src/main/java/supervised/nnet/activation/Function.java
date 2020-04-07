package supervised.nnet.activation;

public interface Function {
	public double f(double x);
	public double fDevFOut(double x);
	public double fDev(double d);
}
