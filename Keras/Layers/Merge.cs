using System;
using System.Collections.Generic;
using System.Text;
using System.Linq;
using Python.Runtime;

namespace Keras.Layers
{
    public class Merge: BaseLayer
    {
        
    }

    public class Add : Merge
    {
        public Add(params BaseLayer[] inputs)
        {
            //Parameters["inputs"] = inputs;
            PyInstance = Instance.keras.layers.add(inputs: inputs.Select(x=>(x.PyInstance)).ToList());
        }
    }

    public class Concatenate : Merge
    {
        public Concatenate(params BaseLayer[] inputs)
        {
            //Parameters["inputs"] = inputs;
            PyInstance = Instance.keras.layers.concatenate(inputs.Select(x => (x.ToPython())).ToList());
        }
    }
}
