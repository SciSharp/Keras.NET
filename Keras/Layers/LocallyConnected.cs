using Numpy.Models;
using System;
using System.Collections.Generic;
using System.Text;

namespace Keras.Layers
{
    public class LocallyConnected1D : BaseLayer
    {
        public LocallyConnected1D(int filters, int kernel_size, int strides = 1, string padding = "valid", string data_format = "channels_last",
                   int dilation_rate = 1, string activation = "", bool use_bias = true, string kernel_initializer = "glorot_uniform",
                   string bias_initializer = "zeros", string kernel_regularizer = "", string bias_regularizer = "",
                   string activity_regularizer = "", string kernel_constraint = "", string bias_constraint = "", Shape input_shape = null)
        {
            Parameters["filters"] = filters;
            Parameters["kernel_size"] = kernel_size;
            Parameters["strides"] = strides;
            Parameters["padding"] = padding;
            Parameters["data_format"] = data_format;
            Parameters["dilation_rate"] = dilation_rate;
            Parameters["activation"] = activation;
            Parameters["use_bias"] = use_bias;
            Parameters["kernel_initializer"] = kernel_initializer;
            Parameters["bias_initializer"] = bias_initializer;
            Parameters["kernel_regularizer"] = kernel_regularizer;
            Parameters["bias_regularizer"] = bias_regularizer;
            Parameters["activity_regularizer"] = activity_regularizer;
            Parameters["kernel_constraint"] = kernel_constraint;
            Parameters["bias_constraint"] = bias_constraint;
            Parameters["input_shape"] = input_shape;

            __self__ = Instance.self.layers.LocallyConnected1D;
        }
    }

    public class LocallyConnected2D : BaseLayer
    {
        public LocallyConnected2D(int filters, Tuple<int, int> kernel_size, Tuple<int, int> strides = null, string padding = "valid", string data_format = "channels_last",
                    Tuple<int, int> dilation_rate = null, string activation = "", bool use_bias = true, string kernel_initializer = "glorot_uniform",
                    string bias_initializer = "zeros", string kernel_regularizer = "", string bias_regularizer = "",
                    string activity_regularizer = "", string kernel_constraint = "", string bias_constraint = "", Shape input_shape = null)
        {
            Parameters["filters"] = filters;
            Parameters["kernel_size"] = (kernel_size.Item1, kernel_size.Item2);
            Parameters["strides"] = strides == null ? (1, 1) : (strides.Item1, strides.Item2);
            Parameters["padding"] = padding;
            Parameters["data_format"] = data_format;
            Parameters["dilation_rate"] = dilation_rate == null ? (1, 1) : (dilation_rate.Item1, dilation_rate.Item2); ;
            Parameters["activation"] = activation;
            Parameters["use_bias"] = use_bias;
            Parameters["kernel_initializer"] = kernel_initializer;
            Parameters["bias_initializer"] = bias_initializer;
            Parameters["kernel_regularizer"] = kernel_regularizer;
            Parameters["bias_regularizer"] = bias_regularizer;
            Parameters["activity_regularizer"] = activity_regularizer;
            Parameters["kernel_constraint"] = kernel_constraint;
            Parameters["bias_constraint"] = bias_constraint;
            Parameters["input_shape"] = input_shape;

            __self__ = Instance.self.layers.LocallyConnected2D;
        }
    }
}
