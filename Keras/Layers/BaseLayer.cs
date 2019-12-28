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
            if (inputs.Length == 1)
            {
                Dictionary<string, object> d = new Dictionary<string, object>();
                d.Add("inputs", inputs[0].ToPython());
                return new BaseLayer(InvokeMethod("__call__", d));
            }
            else
            {
                var t = inputs[0].PyInstance; // Backend.Cast(inputs[0].PyInstance);
                var b = new BaseLayer(t);
                b.Init();
                return b;
                //__self__.input = inputs[0].ToPython();
            }
        }
    }
}
