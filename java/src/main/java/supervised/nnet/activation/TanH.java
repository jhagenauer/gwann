package supervised.nnet.activation;

import org.apache.commons.math3.util.FastMath;

public class TanH implements Function {
	@Override
	public double f(double x) {
		return FastMath.tanh(x);
	}

	@Override
	public double fDevFOut(double fOut) {
		return 1.0 - fOut*fOut;
	}
	
	@Override
	public double fDev(double f) {
		double g = FastMath.tanh(f);
		return 1 - g*g;
	}
}
