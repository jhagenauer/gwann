package dist;

public class ConstantDist<T> implements Dist<T> {
	private double d;
	public ConstantDist(double d ) {
		this.d = d;
	}
	
	@Override
	public double dist(T a, T b) {
		return d;
	}
}
