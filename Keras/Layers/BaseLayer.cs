using System;
using System.Collections.Generic;
using System.Text;
using System.Linq;
using Python.Runtime;

namespace Keras.Layers
{
    public class BaseLayer : Base
    {
        public BaseLayer()
        {

        }

        public BaseLayer(PyObject py)
        {
            PyInstance = py;
        }

        public BaseLayer Set(params BaseLayer[] inputs)
        {
            if (inputs.Length > 1)
                return new BaseLayer(PyInstance(inputs.Select(x => (x.PyInstance)).ToArray()));
            else
            {
                var b = new BaseLayer(PyInstance(inputs[0].PyInstance));
                b.Init();
                return b;
                //__self__.input = inputs[0].ToPython();
            }
        }
    }
}
