using Keras;
using Keras.Layers;
using Keras.Models;
using Numpy;
using System;
using System.Collections.Generic;
using System.Text;

namespace BasicSamples
{
    public class XOR
    {
        public static void Run()
        {
            NDarray x = np.array(new float[,] { { 0, 0 }, { 0, 1 }, { 1, 0 }, { 1, 1 } });
            NDarray y = np.array(new float[] { 0, 1, 1, 0 });

            var model = new Sequential();
            model.Add(new Dense(32, activation: "relu", input_shape: new Shape(4)));
            model.Add(new Dense(64, activation: "relu"));
            model.Add(new Dense(1, activation: "sigmoid"));

            model.Compile("sgd", "binary_crossentropy", new string[] { "accuracy" });
            model.Fit(x, y, 2, 1000, 2);
        }
    }
}
