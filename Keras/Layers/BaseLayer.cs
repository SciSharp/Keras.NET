using System;
using System.Collections.Generic;
using System.Text;
using System.Linq;

namespace Keras.Layers
{
    public class BaseLayer : Base
    {
        public void Set(params BaseLayer[] inputs)
        {
            __self__.input = inputs.Select(x => (x.ToPython())).ToArray();
        }
    }
}
