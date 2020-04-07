package supervised;

public interface SupervisedNet {
	public double[] present( double[] x );
	public double[] getResponse( double[] x, double[] neuron );
	public void train( double t, double[] x, double[] desired ); //Problem: RBF is not time-dependend
}
