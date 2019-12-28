using Keras.Layers;
using Python.Runtime;
using System;
using System.Collections.Generic;
using System.Text;

namespace Keras.Models
{
    /// <summary>
    /// The Sequential model is a linear stack of layers. You can create a Sequential model by passing a list of layer instances to the constructor
    /// </summary>
    /// <seealso cref="Keras.Models.BaseModel" />
    public class Sequential : BaseModel
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="Sequential"/> class.
        /// </summary>
        internal Sequential(PyObject obj)
        {
            PyInstance = obj;
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="Sequential"/> class.
        /// </summary>
        public Sequential()
        {
            PyInstance = Instance.keras.models.Sequential();
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="Sequential"/> class.
        /// </summary>
        /// <param name="layers">The layers.</param>
        public Sequential(BaseLayer[] layers) : this()
        {
            foreach (var item in layers)
            {
                Add(item.PyInstance);
            }
        }

        /// <summary>
        /// You can also simply add layers via the .Add() method
        /// </summary>
        /// <param name="layer">The layer.</param>
        public void Add(BaseLayer layer)
        {
            PyInstance.add(layer: layer.PyInstance);
        }
    }
}
