using Keras.Layers;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace Keras.UnitTest
{
    [TestClass]
    public class Keras_Initializers
    {
        [TestMethod]
        public void Constant()
        {
            Constant constant = new Constant(1);
            var obj = constant.ToPython();
        }
    }
}
