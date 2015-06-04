package info.hb.ccn.main;

import info.hb.cnn.core.CNN;
import info.hb.cnn.core.CNN.LayerBuilder;
import info.hb.cnn.core.Layer;
import info.hb.cnn.core.Layer.Size;
import info.hb.cnn.data.DataSet;
import info.hb.cnn.utils.ConcurentRunner;

public class CNNSpeech {

	private static final String MODEL_NAME = "speech/model/model.cnn";

	private static final String TRAIN_DATA = "speech/train.format";

	private static final String TEST_DATA = "speech/test.format";

	private static final String TEST_PREDICT = "speech/test.predict";

	public static void main(String[] args) {

		System.err.println("训练阶段：");
		runTrain();
		System.err.println("测试阶段：");
		runTest();
		ConcurentRunner.stop();

	}

	public static void runTrain() {
		// 构建网络层次结构
		LayerBuilder builder = new LayerBuilder();
		builder.addLayer(Layer.buildInputLayer(new Size(6, 6)));
		builder.addLayer(Layer.buildConvLayer(2, new Size(3, 3)));
		builder.addLayer(Layer.buildSampLayer(new Size(2, 2)));
		//		builder.addLayer(Layer.buildConvLayer(3, new Size(3, 3)));
		//		builder.addLayer(Layer.buildSampLayer(new Size(2, 2)));
		builder.addLayer(Layer.buildOutputLayer(2));
		CNN cnn = new CNN(builder, 10);
		// 加载训练数据
		DataSet dataset = DataSet.load(TRAIN_DATA, ",", 36);
		// 开始训练模型
		cnn.train(dataset, 3);
		// 保存训练好的模型
		cnn.saveModel(MODEL_NAME);
		dataset.clear();
	}

	public static void runTest() {
		// 加载训练好的模型
		CNN cnn = CNN.loadModel(MODEL_NAME);
		// 加载测试数据
		DataSet testSet = DataSet.load(TEST_DATA, ",", -1);
		// 预测结果
		cnn.predict(testSet, TEST_PREDICT);
		testSet.clear();
	}

}
