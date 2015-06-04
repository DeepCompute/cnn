package info.hb.cnn.utils;

import static org.junit.Assert.assertEquals;
import info.hb.cnn.core.Layer.Size;
import info.hb.cnn.utils.MathUtils.Operator;
import info.hb.cnn.utils.TimedTest.TestTask;

import org.junit.Test;

public class MathUtilsTest {

	@Test
	public void testTimedConvn() {
		new TimedTest(new TestTask() {

			@Override
			public void process() {
				testConvn();
			}

		}, 5).test();
		ConcurentRunner.stop();
	}

	@Test
	public void testTimedScaleMatrix() {
		new TimedTest(new TestTask() {

			@Override
			public void process() {
				testScaleMatrix();
			}

		}, 5).test();
		ConcurentRunner.stop();
	}

	@Test
	public void testTimedKronecker() {
		new TimedTest(new TestTask() {

			@Override
			public void process() {
				testKronecker();
			}

		}, 5).test();
		ConcurentRunner.stop();
	}

	@Test
	public void testTimedMatrixProduct() {
		new TimedTest(new TestTask() {

			@Override
			public void process() {
				testMatrixProduct();
			}

		}, 5).test();
		ConcurentRunner.stop();
	}

	@Test
	public void testTimedCloneMatrix() {
		new TimedTest(new TestTask() {

			@Override
			public void process() {
				testCloneMatrix();
			}

		}, 5).test();
		ConcurentRunner.stop();
	}

	@Test
	public void testM() {
		assertEquals(0.6743346010828868d, MathUtils.sigmod(0.727855957917715), 0.0d);
	}

	private static void testConvn() {
		int count = 1;
		double[][] m = new double[5][5];
		for (int i = 0; i < m.length; i++)
			for (int j = 0; j < m[0].length; j++)
				m[i][j] = count++;
		double[][] k = new double[3][3];
		for (int i = 0; i < k.length; i++)
			for (int j = 0; j < k[0].length; j++)
				k[i][j] = 1;
		double[][] out;
		// out= convnValid(m, k);
		MathUtils.printMatrix(m);
		out = MathUtils.convnFull(m, k);
		MathUtils.printMatrix(out);
		// System.out.println();
		// out = convnFull(m, Util.rot180(k));
		// Util.printMatrix(out);

	}

	private static void testScaleMatrix() {
		int count = 1;
		double[][] m = new double[16][16];
		for (int i = 0; i < m.length; i++)
			for (int j = 0; j < m[0].length; j++)
				m[i][j] = count++;
		double[][] out = MathUtils.scaleMatrix(m, new Size(2, 2));
		MathUtils.printMatrix(m);
		MathUtils.printMatrix(out);
	}

	private static void testKronecker() {
		int count = 1;
		double[][] m = new double[5][5];
		for (int i = 0; i < m.length; i++)
			for (int j = 0; j < m[0].length; j++)
				m[i][j] = count++;
		double[][] out = MathUtils.kronecker(m, new Size(2, 2));
		MathUtils.printMatrix(m);
		System.out.println();
		MathUtils.printMatrix(out);
	}

	private static void testMatrixProduct() {
		int count = 1;
		double[][] m = new double[5][5];
		for (int i = 0; i < m.length; i++)
			for (int j = 0; j < m[0].length; j++)
				m[i][j] = count++;
		double[][] k = new double[5][5];
		for (int i = 0; i < k.length; i++)
			for (int j = 0; j < k[0].length; j++)
				k[i][j] = j;

		MathUtils.printMatrix(m);
		MathUtils.printMatrix(k);
		double[][] out = MathUtils.matrixOp(m, k, new Operator() {

			private static final long serialVersionUID = -680712567166604573L;

			@Override
			public double process(double value) {
				return value - 1;
			}

		}, new Operator() {

			private static final long serialVersionUID = -6335660830579545544L;

			@Override
			public double process(double value) {

				return -1 * value;
			}

		}, MathUtils.multiply);
		MathUtils.printMatrix(out);
	}

	private static void testCloneMatrix() {
		int count = 1;
		double[][] m = new double[5][5];
		for (int i = 0; i < m.length; i++)
			for (int j = 0; j < m[0].length; j++)
				m[i][j] = count++;
		double[][] out = MathUtils.cloneMatrix(m);
		MathUtils.printMatrix(m);

		MathUtils.printMatrix(out);
	}

	public static void testRot180() {
		double[][] matrix = { { 1, 2, 3, 4 }, { 4, 5, 6, 7 }, { 7, 8, 9, 10 } };
		MathUtils.printMatrix(matrix);
		MathUtils.rot180(matrix);
		System.out.println();
		MathUtils.printMatrix(matrix);
	}

}
