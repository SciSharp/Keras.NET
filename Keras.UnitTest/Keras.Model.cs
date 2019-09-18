using System.IO;
using Keras.Models;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace Keras.UnitTest
{
    [TestClass]
    public class Keras_Model
    {
        [TestMethod]

        public void TestLoadModel()
        {
            var filename = "model.h5";
            Assert.IsTrue(File.Exists(filename));
            var model = BaseModel.LoadModel(filename);
            Assert.IsNotNull(model);
        }
    }
}