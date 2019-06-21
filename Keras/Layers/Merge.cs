using System;
using System.Collections.Generic;
using System.Text;
using System.Linq;
using Python.Runtime;

namespace Keras.Layers
{
    public class Merge: BaseLayer
    {
        internal PyObject merged = null;

        public override PyObject ToPython()
        {
            return merged;
        }
    }

    public class Add : Merge
    {
        public Add(params BaseLayer[] inputs)
        {
            //Parameters["inputs"] = inputs;
            merged = Instance.keras.layers.add(inputs: inputs.Select(x=>(x.ToPython())).ToArray());
        }
    }
}
