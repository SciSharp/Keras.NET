using System;
using System.Collections.Generic;
using System.Text;
using Numpy.Models;
using Python.Runtime;

namespace Keras.Layers
{
    public class Input : BaseLayer
    {
        public Input(Shape shape)
        {
            this["shape"] = shape;
            __self__ = Instance.self.layers.Input;
        }
    }

    public class Dense : BaseLayer
    {
        public Dense(int units, string activation= "", bool use_bias= true, string kernel_initializer= "glorot_uniform", 
                    string bias_initializer= "zeros", string kernel_regularizer= "", string bias_regularizer= "", 
                    string activity_regularizer= "", string kernel_constraint= "", string bias_constraint= "")
        {
            this["units"] = units;
            this["activation"] = activation;
            this["use_bias"] = use_bias;
            this["kernel_initializer"] = kernel_initializer;
            this["bias_initializer"] = bias_initializer;
            this["kernel_regularizer"] = kernel_regularizer;
            this["bias_regularizer"] = bias_regularizer;
            this["activity_regularizer"] = activity_regularizer;
            this["kernel_constraint"] = kernel_constraint;
            this["bias_constraint"] = bias_constraint;
            __self__ = Instance.self.layers.Dense;
        }

    }
}
