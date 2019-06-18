using Keras.Layers;
using Microsoft.VisualStudio.TestTools.UnitTesting;
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
        public void ActivationTest()
        {
            Activation act = new Activation("relu");
            var obj = act.ToPython();

            var act1 = Activations.Softmax(np.array<float>(1, 2, 3, 4).reshape(2, 2));
        }
    }
}
