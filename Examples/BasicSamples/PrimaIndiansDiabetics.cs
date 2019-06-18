using Keras;
using Keras.Layers;
using Keras.Models;
using Numpy;
using System;
using System.Collections.Generic;
using System.IO;
using System.Text;

namespace BasicSamples
{
    public class PrimaIndiansDiabetics
    {
        public static void Run()
        {
            //Load train data
            NDarray dataset = np.loadtxt(fname: "pima-indians-diabetes.data.csv", delimiter: ",");
            var X = dataset[":,0: 8"];
            var Y = dataset[":, 8"];

            //Build sequential model
            var model = new Sequential();
            model.Add(new Dense(12, input_dim: 8, kernel_initializer: "uniform", activation: "relu"));
            model.Add(new Dense(8, kernel_initializer: "uniform", activation: "relu"));
            model.Add(new Dense(1, activation: "sigmoid"));

            //Compile and train
            model.Compile(optimizer:"adam", loss:"binary_crossentropy", metrics: new string[] { "accuracy" });
            model.Fit(X, Y, batch_size: 10, epochs: 150, verbose: 1);

            //Evaluate model
            var scores = model.Evaluate(X, Y, verbose: 1);
            Console.WriteLine("Accuracy: {0}", scores[1] * 100);

            //Save model and weights
            string json = model.ToJson();
            File.WriteAllText("model.json", json);
            model.SaveWeight("model.h5");
            Console.WriteLine("Saved model to disk");
            //Load model and weight
            var loaded_model = Sequential.ModelFromJson(File.ReadAllText("model.json"));
            loaded_model.LoadWeight("model.h5");
            Console.WriteLine("Loaded model from disk");

            loaded_model.Compile(optimizer: "rmsprop", loss: "binary_crossentropy", metrics: new string[] { "accuracy" });
            scores = model.Evaluate(X, Y, verbose: 1);
            Console.WriteLine("Accuracy: {0}", scores[1] * 100);
        }
    }
}
