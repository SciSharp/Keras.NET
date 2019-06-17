using Keras.Layers;
using Numpy;
using System;
using System.Collections.Generic;
using System.Text;
using System.Linq;

namespace Keras.Models
{
    public class Model : BaseModel
    {
        public Model(BaseLayer[] inputs, BaseLayer[] outputs)
        {
            List<object> inputList = new List<object>();
            List<object> outputList = new List<object>();

            foreach (var item in inputs)
            {
                inputList.Add(item.GetPythonObject());
            }

            foreach (var item in outputs)
            {
                outputList.Add(item.GetPythonObject());
            }

            __self__ = Instance.self.models.Model(inputs: inputs, outputs: outputs);
        }
    }
}
