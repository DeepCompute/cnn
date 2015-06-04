package info.hb.cnn.utils;

import info.hb.cnn.utils.TimedTest.TestTask;

import java.util.Locale;

import org.junit.Test;

public class TestArray {

	@Test
	public void testOrigin() {
		String a = "aAdfa_";
		System.out.println(a.toUpperCase(Locale.CHINA));
		double[][] d = new double[3][];
		d[0] = new double[] { 1, 2, 3 };
		d[1] = new double[] { 3, 4, 5, 6 };
		System.out.println(d[1][3]);
		final ArrayModel t = new ArrayModel(10000, 1000);
		new TimedTest(new TestTask() {

			@Override
			public void process() {
				t.useOrigin();
			}

		}, 1).test();
	}

	@Test
	public void testFunc() {
		String a = "aAdfa_";
		System.out.println(a.toUpperCase(Locale.CHINA));
		double[][] d = new double[3][];
		d[0] = new double[] { 1, 2, 3 };
		d[1] = new double[] { 3, 4, 5, 6 };
		System.out.println(d[1][3]);
		final ArrayModel t = new ArrayModel(10000, 1000);
		new TimedTest(new TestTask() {

			@Override
			public void process() {
				t.useFunc();
			}

		}, 1).test();
	}

	public static class ArrayModel {

		double[][] data;

		public ArrayModel(int m, int n) {
			data = new double[m][n];
		}

		public void set(int x, int y, double value) {
			data[x][y] = value;
		}

		public void useOrigin() {
			for (int i = 0; i < data.length; i++)
				for (int j = 0; j < data[0].length; j++)
					data[i][j] = i * j;
		}

		public void useFunc() {
			for (int i = 0; i < data.length; i++)
				for (int j = 0; j < data[0].length; j++)
					set(i, j, i * j);
		}

	}

}
