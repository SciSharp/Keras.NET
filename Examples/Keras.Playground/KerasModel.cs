using Keras.Models;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace Keras.Playground
{
    public class KerasModel
    {
        public static BaseModel MNIST = null;

        public static void LoadModels()
        {
            if (MNIST == null)
            {
                MNIST = Sequential.ModelFromJson(System.IO.File.ReadAllText("./MNIST/model.json"));
                MNIST.LoadWeight("./MNIST/weights.h5");
            }
        }
    }
}
