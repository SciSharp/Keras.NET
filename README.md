# Keras.NET

Keras.NET is a high-level neural networks API, written in C# with Python Binding and capable of running on top of TensorFlow, CNTK, or Theano. It was developed with a focus on enabling fast experimentation. Being able to go from idea to result with the least possible delay is key to doing good research.

Use Keras if you need a deep learning library that:

Allows for easy and fast prototyping (through user friendliness, modularity, and extensibility).
Supports both convolutional networks and recurrent networks, as well as combinations of the two.
Runs seamlessly on CPU and GPU.

## Keras.NET is using:

* [Numpy.NET](https://github.com/SciSharp/Numpy.NET)
* [Python.Included](https://github.com/henon/Python.Included)

## Example with XOR sample

```csharp
//Load train data
NDarray x = np.array(new float[,] { { 0, 0 }, { 0, 1 }, { 1, 0 }, { 1, 1 } });
NDarray y = np.array(new float[] { 0, 1, 1, 0 });

//Build sequential model
var model = new Sequential();
model.Add(new Dense(32, activation: "relu", input_shape: new Shape(2)));
model.Add(new Dense(64, activation: "relu"));
model.Add(new Dense(1, activation: "sigmoid"));

//Compile and train
model.Compile(optimizer:"sgd", loss:"binary_crossentropy", metrics: new string[] { "accuracy" });
model.Fit(x, y, batch_size: 2, epochs: 1000, verbose: 1);

//Save model and weights
string json = model.ToJson();
File.WriteAllText("model.json", json);
model.SaveWeight("model.h5");

//Load model and weight
var loaded_model = Sequential.ModelFromJson(File.ReadAllText("model.json"));
loaded_model.LoadWeight("model.h5");
```

**Output:**

![](https://raw.githubusercontent.com/SciSharp/Keras.NET/master/Images/XOR_Output.PNG)

