package info.hb.cnn.data;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class DataSet {

	private static Logger logger = LoggerFactory.getLogger(DataSet.class);

	private List<Record> records;

	private int lableIndex;

	private double maxLable = -1;

	public DataSet(int classIndex) {
		this.lableIndex = classIndex;
		records = new ArrayList<Record>();
	}

	public DataSet(List<double[]> datas) {
		this();
		for (double[] data : datas) {
			append(new Record(data));
		}
	}

	public DataSet() {
		this.lableIndex = -1;
		records = new ArrayList<Record>();
	}

	public int size() {
		return records.size();
	}

	public void setLableIndex(int lableIndex) {
		this.lableIndex = lableIndex;
	}

	public int getLableIndex() {
		return lableIndex;
	}

	public void append(Record record) {
		records.add(record);
	}

	public void clear() {
		records.clear();
	}

	public void append(double[] attrs, Double lable) {
		records.add(new Record(attrs, lable));
	}

	public Iterator<Record> iter() {
		return records.iterator();
	}

	public double[] getAttrs(int index) {
		return records.get(index).getAttrs();
	}

	public Double getLable(int index) {
		return records.get(index).getLabel();
	}

	public static DataSet load(String filePath, String tag, int lableIndex) {
		DataSet dataset = new DataSet();
		dataset.lableIndex = lableIndex;
		File file = new File(filePath);
		try {
			BufferedReader in = new BufferedReader(new FileReader(file));
			String line;
			while ((line = in.readLine()) != null) {
				String[] datas = line.split(tag);
				if (datas.length == 0)
					continue;
				double[] data = new double[datas.length];
				for (int i = 0; i < datas.length; i++)
					data[i] = Double.parseDouble(datas[i]);
				Record record = dataset.new Record(data);
				dataset.append(record);
			}
			in.close();

		} catch (IOException e) {
			e.printStackTrace();
			return null;
		}
		logger.info("DataSet size:{}", dataset.size());
		return dataset;
	}

	public class Record {

		private double[] attrs;
		private Double label;

		private Record(double[] attrs, Double lable) {
			this.attrs = attrs;
			this.label = lable;
		}

		public Record(double[] data) {
			if (lableIndex == -1)
				attrs = data;
			else {
				label = data[lableIndex];
				if (label > maxLable)
					maxLable = label;
				if (lableIndex == 0)
					attrs = Arrays.copyOfRange(data, 1, data.length);
				else
					attrs = Arrays.copyOfRange(data, 0, data.length - 1);
			}
		}

		public double[] getAttrs() {
			return attrs;
		}

		@Override
		public String toString() {
			StringBuilder sb = new StringBuilder();
			sb.append("attrs:");
			sb.append(Arrays.toString(attrs));
			sb.append("lable:");
			sb.append(label);
			return sb.toString();
		}

		public Double getLabel() {
			if (lableIndex == -1)
				return null;
			return label;
		}

		public int[] getEncodeTarget(int n) {
			String binary = Integer.toBinaryString(label.intValue());
			byte[] bytes = binary.getBytes();
			int[] encode = new int[n];
			int j = n;
			for (int i = bytes.length - 1; i >= 0; i--)
				encode[--j] = bytes[i] - '0';

			return encode;
		}

		public double[] getDoubleEncodeTarget(int n) {
			String binary = Integer.toBinaryString(label.intValue());
			byte[] bytes = binary.getBytes();
			double[] encode = new double[n];
			int j = n;
			for (int i = bytes.length - 1; i >= 0; i--)
				encode[--j] = bytes[i] - '0';

			return encode;
		}

	}

	public Record getRecord(int index) {
		return records.get(index);
	}

}
