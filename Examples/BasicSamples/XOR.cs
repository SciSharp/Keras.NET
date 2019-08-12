using Keras;
using Keras.Layers;
using Keras.Models;
using Keras.Optimizers;
using Numpy;
using System;
using System.Collections.Generic;
using System.IO;
using System.Text;

namespace BasicSamples
{
    public class XOR
    {
        public static void Run()
        {
            //Load train data
            NDarray x = np.array(new float[,] { { 0, 0 }, { 0, 1 }, { 1, 0 }, { 1, 1 } });
            NDarray y = np.array(new float[] { 0, 1, 1, 0 });

            //Build sequential model
            var model = new Sequential();
            model.Add(new Dense(32, activation: "relu", input_shape: new Shape(2)));
            model.Add(new Dense(64, activation: "relu"));
            model.Add(new Dense(1, activation: "sigmoid"));

            //Compile and train
            model.Compile(optimizer: new Adam(), loss:"binary_crossentropy", metrics: new string[] { "accuracy" });
            var history = model.Fit(x, y, batch_size: 2, epochs: 10, verbose: 1);
            //var weights = model.GetWeights();
            //model.SetWeights(weights);
            var logs = history.HistoryLogs;
            //Save model and weights
            string json = model.ToJson();
            File.WriteAllText("model.json", json);
            model.SaveWeight("model.h5");

            //Load model and weight
            var loaded_model = Sequential.ModelFromJson(File.ReadAllText("model.json"));
            loaded_model.LoadWeight("model.h5");

        }
    }
}
