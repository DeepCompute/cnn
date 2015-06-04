package info.hb.cnn.data;

import info.hb.cnn.data.DataSet.Record;

import java.util.Arrays;

import org.junit.Test;

public class DataSetTest {

	@Test
	public void testDataSet() {
		DataSet dataSet = new DataSet();
		dataSet.setLableIndex(10);
		Record r = dataSet.new Record(new double[] { 3, 2, 2, 5, 4, 5, 3, 11, 3, 12, 1 });
		int[] encode = r.getEncodeTarget(4);

		System.out.println(r.getLabel());
		System.out.println(Arrays.toString(encode));
	}

}
