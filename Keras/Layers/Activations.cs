using System;
using System.Collections.Generic;
using System.Text;

namespace Keras.Layers
{
    public partial class Activation : Base
    {
        public Activation(string act)
        {
            Parameters["activation"] = act;
            __self__ = Instance.self.layers.Activation;
        }
    }
}
