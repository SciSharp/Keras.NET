using Keras;
using Keras.Callbacks;
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
    public class EarlyStopExample
    {
        public static void Run()
        {
            //Load train data
            NDarray x = np.array(new float[,] { { 0, 0 }, { 0, 1 }, { 1, 0 }, { 1, 1 } });
            NDarray y = np.array(new float[] { 0, 1, 1, 0 });

            //Build sequential model
            var model = new Sequential();
            model.Add(new Dense(4, activation: "relu", input_shape: new Shape(2)));
            model.Add(new Dense(1));

            var stoploss = Callback.Custom("AccHistory", "AccHistory.py");

            //Compile and train
            model.Compile(optimizer: new SGD(), loss: "binary_crossentropy", metrics: new string[] { "accuracy" });
            var history = model.Fit(x, y, batch_size: 2, epochs: 10, verbose: 1, callbacks: new Callback[] { stoploss });
        }
    }
}
