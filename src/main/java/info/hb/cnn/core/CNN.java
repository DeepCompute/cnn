package info.hb.cnn.core;

import info.hb.cnn.core.Layer.Size;
import info.hb.cnn.data.DataSet;
import info.hb.cnn.data.DataSet.Record;
import info.hb.cnn.utils.ConcurentRunner.TaskManager;
import info.hb.cnn.utils.MathUtils;
import info.hb.cnn.utils.MathUtils.Operator;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.PrintWriter;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.concurrent.atomic.AtomicBoolean;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class CNN implements Serializable {

	private static final long serialVersionUID = 337920299147929932L;

	private static Logger logger = LoggerFactory.getLogger(CNN.class);

	private static double ALPHA = 0.85;
	protected static final double LAMBDA = 0;
	private List<Layer> layers;
	private int layerNum;

	private int batchSize;
	private Operator divide_batchSize;

	private Operator multiply_alpha;

	private Operator multiply_lambda;

	public CNN(LayerBuilder layerBuilder, final int batchSize) {
		layers = layerBuilder.mLayers;
		layerNum = layers.size();
		this.batchSize = batchSize;
		setup(batchSize);
		initPerator();
	}

	private void initPerator() {

		divide_batchSize = new Operator() {

			private static final long serialVersionUID = 7424011281732651055L;

			@Override
			public double process(double value) {
				return value / batchSize;
			}

		};

		multiply_alpha = new Operator() {

			private static final long serialVersionUID = 5761368499808006552L;

			@Override
			public double process(double value) {

				return value * ALPHA;
			}

		};

		multiply_lambda = new Operator() {

			private static final long serialVersionUID = 4499087728362870577L;

			@Override
			public double process(double value) {

				return value * (1 - LAMBDA * ALPHA);
			}

		};

	}

	public void train(DataSet trainset, int repeat) {

		new Lisenter().start();

		for (int t = 0; t < repeat && !stopTrain.get(); t++) {
			int epochsNum = trainset.size() / batchSize;
			if (trainset.size() % batchSize != 0)
				epochsNum++;
			logger.info("第{}次迭代，epochsNum: {}", t, epochsNum);
			int right = 0;
			int count = 0;
			for (int i = 0; i < epochsNum; i++) {
				int[] randPerm = MathUtils.randomPerm(trainset.size(), batchSize);
				Layer.prepareForNewBatch();

				for (int index : randPerm) {
					boolean isRight = train(trainset.getRecord(index));
					if (isRight)
						right++;
					count++;
					Layer.prepareForNewRecord();
				}

				updateParas();
				if (i % 50 == 0) {
					System.out.print("..");
					if (i + 50 > epochsNum)
						System.out.println();
				}
			}
			double p = 1.0 * right / count;
			if (t % 10 == 1 && p > 0.96) {
				ALPHA = 0.001 + ALPHA * 0.9;
				logger.info("设置 alpha = {}", ALPHA);
			}
			logger.info("计算精度： {}/{}={}.", right, count, p);
		}

	}

	private static AtomicBoolean stopTrain;

	static class Lisenter extends Thread {

		Lisenter() {
			setDaemon(true);
			stopTrain = new AtomicBoolean(false);
		}

		@Override
		public void run() {
			logger.info("输入&符号停止训练.");
			while (true) {
				try {
					int a = System.in.read();
					if (a == '&') {
						stopTrain.compareAndSet(false, true);
						break;
					}
				} catch (IOException e) {
					e.printStackTrace();
				}
			}
			System.out.println("监听停止.");
		}

	}

	@SuppressWarnings("unused")
	private double test(DataSet trainset) {
		Layer.prepareForNewBatch();
		Iterator<Record> iter = trainset.iter();
		int right = 0;
		while (iter.hasNext()) {
			Record record = iter.next();
			forward(record);
			Layer outputLayer = layers.get(layerNum - 1);
			int mapNum = outputLayer.getOutMapNum();
			double[] out = new double[mapNum];
			for (int m = 0; m < mapNum; m++) {
				double[][] outmap = outputLayer.getMap(m);
				out[m] = outmap[0][0];
			}
			if (record.getLabel().intValue() == MathUtils.getMaxIndex(out))
				right++;
		}
		double p = 1.0 * right / trainset.size();
		logger.info("计算精度为：\t{}", p + "");
		return p;
	}

	public void predict(DataSet testset, String fileName) {
		logger.info("开始预测 ...");
		try {
			//			int max = layers.get(layerNum - 1).getClassNum();
			PrintWriter writer = new PrintWriter(new File(fileName));
			Layer.prepareForNewBatch();
			Iterator<Record> iter = testset.iter();
			while (iter.hasNext()) {
				Record record = iter.next();
				forward(record);
				Layer outputLayer = layers.get(layerNum - 1);

				int mapNum = outputLayer.getOutMapNum();
				double[] out = new double[mapNum];
				for (int m = 0; m < mapNum; m++) {
					double[][] outmap = outputLayer.getMap(m);
					out[m] = outmap[0][0];
				}
				//				int lable = MathUtils.binaryArray2int(out);
				int lable = MathUtils.getMaxIndex(out);
				//				if (lable >= max)
				//					lable = lable - (1 << (out.length - 1));
				writer.write(lable + "\n");
			}
			writer.flush();
			writer.close();
		} catch (IOException e) {
			throw new RuntimeException(e);
		}
		logger.info("完成预测 ...");
	}

	@SuppressWarnings("unused")
	private boolean isSame(double[] output, double[] target) {
		boolean r = true;
		for (int i = 0; i < output.length; i++)
			if (Math.abs(output[i] - target[i]) > 0.5) {
				r = false;
				break;
			}

		return r;
	}

	private boolean train(Record record) {
		forward(record);
		boolean result = backPropagation(record);
		return result;
		// System.exit(0);
	}

	private boolean backPropagation(Record record) {
		boolean result = setOutLayerErrors(record);
		setHiddenLayerErrors();
		return result;
	}

	private void updateParas() {
		for (int l = 1; l < layerNum; l++) {
			Layer layer = layers.get(l);
			Layer lastLayer = layers.get(l - 1);
			switch (layer.getType()) {
			case conv:
			case output:
				updateKernels(layer, lastLayer);
				updateBias(layer, lastLayer);
				break;
			default:
				break;
			}
		}
	}

	private void updateBias(final Layer layer, Layer lastLayer) {
		final double[][][][] errors = layer.getErrors();
		int mapNum = layer.getOutMapNum();

		new TaskManager(mapNum) {
			@Override
			public void process(int start, int end) {
				for (int j = start; j < end; j++) {
					double[][] error = MathUtils.sum(errors, j);
					double deltaBias = MathUtils.sum(error) / batchSize;
					double bias = layer.getBias(j) + ALPHA * deltaBias;
					layer.setBias(j, bias);
				}
			}
		}.start();

	}

	private void updateKernels(final Layer layer, final Layer lastLayer) {

		int mapNum = layer.getOutMapNum();
		final int lastMapNum = lastLayer.getOutMapNum();

		new TaskManager(mapNum) {

			@Override
			public void process(int start, int end) {
				for (int j = start; j < end; j++) {
					for (int i = 0; i < lastMapNum; i++) {
						double[][] deltaKernel = null;
						for (int r = 0; r < batchSize; r++) {
							double[][] error = layer.getError(r, j);
							if (deltaKernel == null)
								deltaKernel = MathUtils.convnValid(lastLayer.getMap(r, i), error);
							else {
								deltaKernel = MathUtils.matrixOp(MathUtils.convnValid(lastLayer.getMap(r, i), error),
										deltaKernel, null, null, MathUtils.plus);
							}
						}

						deltaKernel = MathUtils.matrixOp(deltaKernel, divide_batchSize);
						double[][] kernel = layer.getKernel(i, j);
						deltaKernel = MathUtils.matrixOp(kernel, deltaKernel, multiply_lambda, multiply_alpha,
								MathUtils.plus);
						layer.setKernel(i, j, deltaKernel);
					}
				}

			}
		}.start();

	}

	private void setHiddenLayerErrors() {
		for (int l = layerNum - 2; l > 0; l--) {
			Layer layer = layers.get(l);
			Layer nextLayer = layers.get(l + 1);
			switch (layer.getType()) {
			case samp:
				setSampErrors(layer, nextLayer);
				break;
			case conv:
				setConvErrors(layer, nextLayer);
				break;
			default:
				break;
			}
		}
	}

	private void setSampErrors(final Layer layer, final Layer nextLayer) {

		int mapNum = layer.getOutMapNum();
		final int nextMapNum = nextLayer.getOutMapNum();

		new TaskManager(mapNum) {

			@Override
			public void process(int start, int end) {
				for (int i = start; i < end; i++) {
					double[][] sum = null;
					for (int j = 0; j < nextMapNum; j++) {
						double[][] nextError = nextLayer.getError(j);
						double[][] kernel = nextLayer.getKernel(i, j);
						if (sum == null)
							sum = MathUtils.convnFull(nextError, MathUtils.rot180(kernel));
						else
							sum = MathUtils.matrixOp(MathUtils.convnFull(nextError, MathUtils.rot180(kernel)), sum,
									null, null, MathUtils.plus);
					}
					layer.setError(i, sum);
				}
			}

		}.start();

	}

	private void setConvErrors(final Layer layer, final Layer nextLayer) {

		int mapNum = layer.getOutMapNum();

		new TaskManager(mapNum) {

			@Override
			public void process(int start, int end) {
				for (int m = start; m < end; m++) {
					Size scale = nextLayer.getScaleSize();
					double[][] nextError = nextLayer.getError(m);
					double[][] map = layer.getMap(m);
					double[][] outMatrix = MathUtils.matrixOp(map, MathUtils.cloneMatrix(map), null,
							MathUtils.one_value, MathUtils.multiply);
					outMatrix = MathUtils.matrixOp(outMatrix, MathUtils.kronecker(nextError, scale), null, null,
							MathUtils.multiply);
					layer.setError(m, outMatrix);
				}

			}

		}.start();

	}

	private boolean setOutLayerErrors(Record record) {

		Layer outputLayer = layers.get(layerNum - 1);
		int mapNum = outputLayer.getOutMapNum();
		// double[] target =
		// record.getDoubleEncodeTarget(mapNum);
		// double[] outmaps = new double[mapNum];
		// for (int m = 0; m < mapNum; m++) {
		// double[][] outmap = outputLayer.getMap(m);
		// double output = outmap[0][0];
		// outmaps[m] = output;
		// double errors = output * (1 - output) *
		// (target[m] - output);
		// outputLayer.setError(m, 0, 0, errors);
		// }
		// if (isSame(outmaps, target))
		// return true;
		// return false;

		double[] target = new double[mapNum];
		double[] outmaps = new double[mapNum];
		for (int m = 0; m < mapNum; m++) {
			double[][] outmap = outputLayer.getMap(m);
			outmaps[m] = outmap[0][0];

		}
		int lable = record.getLabel().intValue();
		target[lable] = 1;
		// Log.i(record.getLable() + "outmaps:" +
		// Util.fomart(outmaps)
		// + Arrays.toString(target));
		for (int m = 0; m < mapNum; m++) {
			outputLayer.setError(m, 0, 0, outmaps[m] * (1 - outmaps[m]) * (target[m] - outmaps[m]));
		}

		return lable == MathUtils.getMaxIndex(outmaps);
	}

	private void forward(Record record) {
		setInLayerOutput(record);
		for (int l = 1; l < layers.size(); l++) {
			Layer layer = layers.get(l);
			Layer lastLayer = layers.get(l - 1);
			switch (layer.getType()) {
			case conv:
				setConvOutput(layer, lastLayer);
				break;
			case samp:
				setSampOutput(layer, lastLayer);
				break;
			case output:
				setConvOutput(layer, lastLayer);
				break;
			default:
				break;
			}
		}
	}

	private void setInLayerOutput(Record record) {
		final Layer inputLayer = layers.get(0);
		final Size mapSize = inputLayer.getMapSize();
		final double[] attr = record.getAttrs();
		if (attr.length != mapSize.x * mapSize.y)
			throw new RuntimeException("map");
		for (int i = 0; i < mapSize.x; i++) {
			for (int j = 0; j < mapSize.y; j++) {
				// inputLayer.setMapValue(0, i, j, attr[mapSize.x * i + j]);
				inputLayer.setMapValue(0, i, j, attr[mapSize.y * i + j]);
			}
		}
	}

	private void setConvOutput(final Layer layer, final Layer lastLayer) {

		int mapNum = layer.getOutMapNum();
		final int lastMapNum = lastLayer.getOutMapNum();

		new TaskManager(mapNum) {

			@Override
			public void process(int start, int end) {
				for (int j = start; j < end; j++) {
					double[][] sum = null;
					for (int i = 0; i < lastMapNum; i++) {
						double[][] lastMap = lastLayer.getMap(i);
						double[][] kernel = layer.getKernel(i, j);
						if (sum == null)
							sum = MathUtils.convnValid(lastMap, kernel);
						else
							sum = MathUtils.matrixOp(MathUtils.convnValid(lastMap, kernel), sum, null, null,
									MathUtils.plus);
					}
					final double bias = layer.getBias(j);
					sum = MathUtils.matrixOp(sum, new Operator() {
						private static final long serialVersionUID = 2469461972825890810L;

						@Override
						public double process(double value) {
							return MathUtils.sigmod(value + bias);
						}

					});

					layer.setMapValue(j, sum);
				}
			}

		}.start();

	}

	private void setSampOutput(final Layer layer, final Layer lastLayer) {

		int lastMapNum = lastLayer.getOutMapNum();

		new TaskManager(lastMapNum) {

			@Override
			public void process(int start, int end) {
				for (int i = start; i < end; i++) {
					double[][] lastMap = lastLayer.getMap(i);
					Size scaleSize = layer.getScaleSize();
					double[][] sampMatrix = MathUtils.scaleMatrix(lastMap, scaleSize);
					layer.setMapValue(i, sampMatrix);
				}
			}

		}.start();

	}

	public void setup(int batchSize) {

		Layer inputLayer = layers.get(0);
		inputLayer.initOutmaps(batchSize);

		for (int i = 1; i < layers.size(); i++) {
			Layer layer = layers.get(i);
			Layer frontLayer = layers.get(i - 1);
			int frontMapNum = frontLayer.getOutMapNum();
			switch (layer.getType()) {
			case input:
				break;
			case conv:
				layer.setMapSize(frontLayer.getMapSize().subtract(layer.getKernelSize(), 1));
				layer.initKernel(frontMapNum);
				layer.initBias(frontMapNum);
				layer.initErros(batchSize);
				layer.initOutmaps(batchSize);
				break;
			case samp:
				layer.setOutMapNum(frontMapNum);
				layer.setMapSize(frontLayer.getMapSize().divide(layer.getScaleSize()));
				layer.initErros(batchSize);
				layer.initOutmaps(batchSize);
				break;
			case output:
				layer.initOutputKerkel(frontMapNum, frontLayer.getMapSize());
				layer.initBias(frontMapNum);
				layer.initErros(batchSize);
				layer.initOutmaps(batchSize);
				break;
			}
		}
	}

	public static class LayerBuilder {

		private List<Layer> mLayers;

		public LayerBuilder() {
			mLayers = new ArrayList<Layer>();
		}

		public LayerBuilder(Layer layer) {
			this();
			mLayers.add(layer);
		}

		public LayerBuilder addLayer(Layer layer) {
			mLayers.add(layer);
			return this;
		}

	}

	public void saveModel(String fileName) {
		try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(fileName));) {
			oos.writeObject(this);
			oos.flush();
		} catch (IOException e) {
			logger.error("IOException:{}", e);
			throw new RuntimeException(e);
		}
	}

	public static CNN loadModel(String fileName) {
		try (ObjectInputStream in = new ObjectInputStream(new FileInputStream(fileName));) {
			CNN cnn = (CNN) in.readObject();
			return cnn;
		} catch (IOException | ClassNotFoundException e) {
			logger.error("IOException or ClassNotFoundException:{}", e);
			throw new RuntimeException(e);
		}
	}

}
