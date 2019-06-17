using Keras.Layers;
using System;
using System.Collections.Generic;
using System.Text;

namespace Keras.Models
{
    public class Sequential : BaseModel
    {
        public Sequential()
        {
            __self__ = Instance.self.models.Sequential();
        }

        public Sequential(BaseLayer[] layers) : this()
        {
            foreach (var item in layers)
            {
                Add(item);
            }
        }

        public void Add(BaseLayer layer)
        {
            __self__.add(layer: layer.ToPython());
        }
    }
}
