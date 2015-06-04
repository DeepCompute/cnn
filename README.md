# JavaCNN

> 基于Java实现CNN

## 构建CNN

	LayerBuilder builder = new LayerBuilder();
	builder.addLayer(Layer.buildInputLayer(new Size(28, 28)));
	builder.addLayer(Layer.buildConvLayer(6, new Size(5, 5)));
	builder.addLayer(Layer.buildSampLayer(new Size(2, 2)));
	builder.addLayer(Layer.buildConvLayer(12, new Size(5, 5)));
	builder.addLayer(Layer.buildSampLayer(new Size(2, 2)));
	builder.addLayer(Layer.buildOutputLayer(10));
	CNN cnn = new CNN(builder, 50);
	
## 运行MNIST数据集
	
	String fileName = "data/train.format";
	Dataset dataset = Dataset.load(fileName, ",", 784);
	cnn.train(dataset, 100);
	Dataset testset = Dataset.load("data/test.format", ",", -1);
	cnn.predict(testset, "data/test.predict");

计算精度可以达到97.8%。
