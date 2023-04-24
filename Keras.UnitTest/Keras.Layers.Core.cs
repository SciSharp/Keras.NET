using System.IO;
using Keras.Initializer;
using Keras.Layers;
using Keras.Models;
using Keras.Optimizers;
using Keras.Regularizers;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Newtonsoft.Json;
using NuGet.Frameworks;
using Numpy;

namespace Keras.UnitTest
{
    [TestClass]
    public class Keras_Layers_Core
    {
        [TestMethod]
        public void DenseTest()
        {
            Dense dense = new Dense(10, activation: "relu");
            var obj = dense.ToPython();
        }

        [TestMethod]
        public void Dense_CustomKRegularizerAndKInitParams()
        {
            NDarray x = np.array(new float[,] { { 1, 0 }, { 1, 1 }, { 1, 0 }, { 1, 1 } });
            NDarray y = np.array(new float[] { 0, 1, 1, 0 });

            var model = new Sequential();
            model.Add(new Dense(1, activation: "sigmoid", input_shape: new Shape(x.shape[1]),kernel_regularizer: new L1L2(1000,2000), kernel_initializer: new Constant(100)));

            var modelAsJson = JsonConvert.DeserializeObject<dynamic>(model.ToJson());

            Assert.AreEqual("Sequential",modelAsJson.class_name.Value);
            int i = 0;
            while (modelAsJson.config.layers[i].config.kernel_initializer == null && i < 3)
            {
                i++; 
            }
            Assert.AreEqual(100, modelAsJson.config.layers[i].config.kernel_initializer.config.value.Value);
            Assert.AreEqual("Constant", modelAsJson.config.layers[i].config.kernel_initializer.class_name.Value);

            Assert.AreEqual("L1L2", modelAsJson.config.layers[i].config.kernel_regularizer.class_name.Value);
            Assert.AreEqual(1000, modelAsJson.config.layers[i].config.kernel_regularizer.config.l1.Value);
            Assert.AreEqual(2000, modelAsJson.config.layers[i].config.kernel_regularizer.config.l2.Value);

            // Compile and train
            model.Compile(optimizer: new Adam(learning_rate: 0.001F), loss: "binary_crossentropy", metrics: new string[] { "accuracy" });
            model.Fit(x, y, batch_size: x.shape[0], epochs: 100, verbose: 0);
            Assert.AreEqual(2, model.GetWeights().Count);
        }

        [TestMethod]
        public void ActivationTest()
        {
            Activation act = new Activation("relu");
            var obj = act.ToPython();

            var act1 = Activations.Softmax(np.array<float>(1, 2, 3, 4).reshape(2, 2));
        }
    }
}