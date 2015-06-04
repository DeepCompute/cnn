package info.hb.cnn.utils;

import info.hb.cnn.core.Layer.Size;

import java.io.Serializable;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Random;
import java.util.Set;

public class MathUtils {

	public interface Operator extends Serializable {
		public double process(double value);
	}

	public static final Operator one_value = new Operator() {

		private static final long serialVersionUID = 3752139491940330714L;

		@Override
		public double process(double value) {
			return 1 - value;
		}

	};

	public static final Operator digmod = new Operator() {

		private static final long serialVersionUID = -1952718905019847589L;

		@Override
		public double process(double value) {
			return 1 / (1 + Math.pow(Math.E, -value));
		}

	};

	interface OperatorOnTwo extends Serializable {
		public double process(double a, double b);
	}

	public static final OperatorOnTwo plus = new OperatorOnTwo() {

		private static final long serialVersionUID = -6298144029766839945L;

		@Override
		public double process(double a, double b) {
			return a + b;
		}

	};

	public static OperatorOnTwo multiply = new OperatorOnTwo() {

		private static final long serialVersionUID = -7053767821858820698L;

		@Override
		public double process(double a, double b) {
			return a * b;
		}

	};

	public static OperatorOnTwo minus = new OperatorOnTwo() {

		private static final long serialVersionUID = 7346065545555093912L;

		@Override
		public double process(double a, double b) {
			return a - b;
		}

	};

	public static void printMatrix(double[][] matrix) {
		for (int i = 0; i < matrix.length; i++) {
			String line = Arrays.toString(matrix[i]);
			line = line.replaceAll(", ", "\t");
			System.out.println(line);
		}
		System.out.println();
	}

	public static double[][] rot180(double[][] matrix) {
		matrix = cloneMatrix(matrix);
		int m = matrix.length;
		int n = matrix[0].length;
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n / 2; j++) {
				double tmp = matrix[i][j];
				matrix[i][j] = matrix[i][n - 1 - j];
				matrix[i][n - 1 - j] = tmp;
			}
		}
		for (int j = 0; j < n; j++) {
			for (int i = 0; i < m / 2; i++) {
				double tmp = matrix[i][j];
				matrix[i][j] = matrix[m - 1 - i][j];
				matrix[m - 1 - i][j] = tmp;
			}
		}
		return matrix;
	}

	private static Random r = new Random(2);

	public static double[][] randomMatrix(int x, int y, boolean b) {
		double[][] matrix = new double[x][y];
		//		int tag = 1;
		for (int i = 0; i < x; i++) {
			for (int j = 0; j < y; j++) {
				// [-0.05,0.05)
				matrix[i][j] = (r.nextDouble() - 0.05) / 10;
				//				matrix[i][j] = tag * 0.5;
				//				if (b)
				//					matrix[i][j] *= 1.0*(i + j + 2) / (i + 1) / (j + 1);
				//				tag *= -1;
			}
		}
		// printMatrix(matrix);
		return matrix;
	}

	public static double[] randomArray(int len) {
		double[] data = new double[len];
		for (int i = 0; i < len; i++) {
			// data[i] = r.nextDouble() / 10 - 0.05;
			data[i] = 0;
		}
		return data;
	}

	public static int[] randomPerm(int size, int batchSize) {
		Set<Integer> set = new HashSet<Integer>();
		while (set.size() < batchSize) {
			set.add(r.nextInt(size));
		}
		int[] randPerm = new int[batchSize];
		int i = 0;
		for (Integer value : set)
			randPerm[i++] = value;
		return randPerm;
	}

	public static double[][] cloneMatrix(final double[][] matrix) {
		final int m = matrix.length;
		int n = matrix[0].length;
		final double[][] outMatrix = new double[m][n];

		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				outMatrix[i][j] = matrix[i][j];
			}
		}
		return outMatrix;
	}

	public static double[][] matrixOp(final double[][] ma, Operator operator) {
		final int m = ma.length;
		int n = ma[0].length;
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				ma[i][j] = operator.process(ma[i][j]);
			}
		}
		return ma;
	}

	public static double[][] matrixOp(final double[][] ma, final double[][] mb, final Operator operatorA,
			final Operator operatorB, OperatorOnTwo operator) {
		final int m = ma.length;
		int n = ma[0].length;
		if (m != mb.length || n != mb[0].length)
			throw new RuntimeException("ma.length:" + ma.length + "  mb.length:" + mb.length);

		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				double a = ma[i][j];
				if (operatorA != null)
					a = operatorA.process(a);
				double b = mb[i][j];
				if (operatorB != null)
					b = operatorB.process(b);
				mb[i][j] = operator.process(a, b);
			}
		}
		return mb;
	}

	public static double[][] kronecker(final double[][] matrix, final Size scale) {
		final int m = matrix.length;
		int n = matrix[0].length;
		final double[][] outMatrix = new double[m * scale.x][n * scale.y];

		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				for (int ki = i * scale.x; ki < (i + 1) * scale.x; ki++) {
					for (int kj = j * scale.y; kj < (j + 1) * scale.y; kj++) {
						outMatrix[ki][kj] = matrix[i][j];
					}
				}
			}
		}
		return outMatrix;
	}

	public static double[][] scaleMatrix(final double[][] matrix, final Size scale) {
		int m = matrix.length;
		int n = matrix[0].length;
		final int sm = m / scale.x;
		final int sn = n / scale.y;
		final double[][] outMatrix = new double[sm][sn];
		if (sm * scale.x != m || sn * scale.y != n)
			throw new RuntimeException("scale matrix");
		final int size = scale.x * scale.y;
		for (int i = 0; i < sm; i++) {
			for (int j = 0; j < sn; j++) {
				double sum = 0.0;
				for (int si = i * scale.x; si < (i + 1) * scale.x; si++) {
					for (int sj = j * scale.y; sj < (j + 1) * scale.y; sj++) {
						sum += matrix[si][sj];
					}
				}
				outMatrix[i][j] = sum / size;
			}
		}
		return outMatrix;
	}

	public static double[][] convnFull(double[][] matrix, final double[][] kernel) {
		int m = matrix.length;
		int n = matrix[0].length;
		final int km = kernel.length;
		final int kn = kernel[0].length;
		final double[][] extendMatrix = new double[m + 2 * (km - 1)][n + 2 * (kn - 1)];
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++)
				extendMatrix[i + km - 1][j + kn - 1] = matrix[i][j];
		}
		return convnValid(extendMatrix, kernel);
	}

	public static double[][] convnValid(final double[][] matrix, double[][] kernel) {
		//		kernel = rot180(kernel);
		int m = matrix.length;
		int n = matrix[0].length;
		final int km = kernel.length;
		final int kn = kernel[0].length;
		int kns = n - kn + 1;
		final int kms = m - km + 1;
		final double[][] outMatrix = new double[kms][kns];

		for (int i = 0; i < kms; i++) {
			for (int j = 0; j < kns; j++) {
				double sum = 0.0;
				for (int ki = 0; ki < km; ki++) {
					for (int kj = 0; kj < kn; kj++)
						sum += matrix[i + ki][j + kj] * kernel[ki][kj];
				}
				outMatrix[i][j] = sum;

			}
		}
		return outMatrix;
	}

	public static double[][] convnValid(final double[][][][] matrix, int mapNoX, double[][][][] kernel, int mapNoY) {
		int m = matrix.length;
		int n = matrix[0][mapNoX].length;
		int h = matrix[0][mapNoX][0].length;
		int km = kernel.length;
		int kn = kernel[0][mapNoY].length;
		int kh = kernel[0][mapNoY][0].length;
		int kms = m - km + 1;
		int kns = n - kn + 1;
		int khs = h - kh + 1;
		if (matrix.length != kernel.length)
			throw new RuntimeException("length");
		final double[][][] outMatrix = new double[kms][kns][khs];
		for (int i = 0; i < kms; i++) {
			for (int j = 0; j < kns; j++)
				for (int k = 0; k < khs; k++) {
					double sum = 0.0;
					for (int ki = 0; ki < km; ki++) {
						for (int kj = 0; kj < kn; kj++)
							for (int kk = 0; kk < kh; kk++) {
								sum += matrix[i + ki][mapNoX][j + kj][k + kk] * kernel[ki][mapNoY][kj][kk];
							}
					}
					outMatrix[i][j][k] = sum;
				}
		}
		return outMatrix[0];
	}

	public static double sigmod(double x) {
		return 1 / (1 + Math.pow(Math.E, -x));
	}

	public static double sum(double[][] error) {
		int m = error.length;
		int n = error[0].length;
		double sum = 0.0;
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				sum += error[i][j];
			}
		}
		return sum;
	}

	public static double[][] sum(double[][][][] errors, int j) {
		int m = errors[0][j].length;
		int n = errors[0][j][0].length;
		double[][] result = new double[m][n];
		for (int mi = 0; mi < m; mi++) {
			for (int nj = 0; nj < n; nj++) {
				double sum = 0;
				for (int i = 0; i < errors.length; i++)
					sum += errors[i][j][mi][nj];
				result[mi][nj] = sum;
			}
		}
		return result;
	}

	public static int binaryArray2int(double[] array) {
		int[] d = new int[array.length];
		for (int i = 0; i < d.length; i++) {
			if (array[i] >= 0.500000001)
				d[i] = 1;
			else
				d[i] = 0;
		}
		String s = Arrays.toString(d);
		String binary = s.substring(1, s.length() - 1).replace(", ", "");
		int data = Integer.parseInt(binary, 2);
		return data;

	}

	public static int getMaxIndex(double[] out) {
		double max = out[0];
		int index = 0;
		for (int i = 1; i < out.length; i++)
			if (out[i] > max) {
				max = out[i];
				index = i;
			}
		return index;
	}

	public static String fomart(double[] data) {
		StringBuilder sb = new StringBuilder("[");
		for (double each : data)
			sb.append(String.format("%4f,", each));
		sb.append("]");
		return sb.toString();
	}

}
