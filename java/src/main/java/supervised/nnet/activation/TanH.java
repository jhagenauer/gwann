package supervised.nnet.activation;

import org.apache.commons.math3.util.FastMath;

public class TanH implements Function {
	@Override
	public double f(double x) {
		//return Math.tanh(x); // RMSE: 1.2726902074544277, Took: 51.49s
		return FastMath.tanh(x); // RMSE: 1.2677143077504636, Took: 38.617s
	}

	@Override
	public double fDevFOut(double fOut) {
		//return 1.0 - Math.pow(fOut, 2);
		return 1.0 - fOut*fOut;
	}
	
	@Override
	public double fDev(double f) {
		return 1 - Math.pow( FastMath.tanh(f), 2);
	}
}
