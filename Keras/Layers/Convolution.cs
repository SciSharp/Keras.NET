using Numpy.Models;
using System;
using System.Collections.Generic;
using System.Text;

namespace Keras.Layers
{
    public class Conv1D : BaseLayer
    {
        public Conv1D(int filters, int kernel_size, int strides = 1, string padding = "valid", string data_format = "channels_last",
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

            __self__ = Instance.self.layers.Conv1D;
        }
    }

    public class Conv2D : BaseLayer
    {
        public Conv2D(int filters, Tuple<int, int> kernel_size, Tuple<int, int> strides = null, string padding = "valid", string data_format = "channels_last",
                    Tuple<int, int> dilation_rate = null, string activation = "", bool use_bias = true, string kernel_initializer = "glorot_uniform",
                    string bias_initializer = "zeros", string kernel_regularizer = "", string bias_regularizer = "",
                    string activity_regularizer = "", string kernel_constraint = "", string bias_constraint = "", Shape input_shape = null)
        {
            Parameters["filters"] = filters;
            Parameters["kernel_size"] = new Shape(kernel_size.Item1, kernel_size.Item2); 
            Parameters["strides"] = strides == null ? new Shape(1, 1) : new Shape(strides.Item1, strides.Item2);
            Parameters["padding"] = padding;
            Parameters["data_format"] = data_format;
            Parameters["dilation_rate"] = dilation_rate == null ? new Shape(1, 1) : new Shape(dilation_rate.Item1, dilation_rate.Item2);
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

            __self__ = Instance.self.layers.Conv2D;
        }
    }

    public class Conv3D : BaseLayer
    {
        public Conv3D(int filters, Tuple<int, int, int> kernel_size, Tuple<int, int, int> strides = null, string padding = "valid", string data_format = "channels_last",
                    Tuple<int, int, int> dilation_rate = null, string activation = "", bool use_bias = true, string kernel_initializer = "glorot_uniform",
                    string bias_initializer = "zeros", string kernel_regularizer = "", string bias_regularizer = "",
                    string activity_regularizer = "", string kernel_constraint = "", string bias_constraint = "", Shape input_shape = null)
        {
            Parameters["filters"] = filters;
            Parameters["kernel_size"] = new Shape(kernel_size.Item1, kernel_size.Item2, kernel_size.Item3);
            Parameters["strides"] = strides == null ? new Shape(1, 1, 1) : new Shape(strides.Item1, strides.Item2, strides.Item3);
            Parameters["padding"] = padding;
            Parameters["data_format"] = data_format;
            Parameters["dilation_rate"] = dilation_rate == null ? new Shape(1, 1, 1) : new Shape(dilation_rate.Item1, dilation_rate.Item2, dilation_rate.Item3); ;
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

            __self__ = Instance.self.layers.Conv3D;
        }
    }

    public class SeparableConv1D : BaseLayer
    {
        public SeparableConv1D(int filters, int kernel_size, int strides = 1, string padding = "valid", string data_format = "channels_last",
                    int dilation_rate = 1, int depth_multiplier = 1, string activation = "", bool use_bias = true, string depthwise_initializer = "glorot_uniform",
                    string pointwise_initializer = "glorot_uniform", string bias_initializer = "zeros", string depthwise_regularizer = "", 
                    string pointwise_regularizer = "", string bias_regularizer = "", string activity_regularizer = "", string depthwise_constraint = "", 
                    string pointwise_constraint = "", string bias_constraint = "", Shape input_shape = null)
        {
            Parameters["filters"] = filters;
            Parameters["kernel_size"] = kernel_size;
            Parameters["strides"] = strides;
            Parameters["padding"] = padding;
            Parameters["data_format"] = data_format;
            Parameters["dilation_rate"] = dilation_rate;
            Parameters["depth_multiplier"] = depth_multiplier;
            Parameters["activation"] = activation;
            Parameters["use_bias"] = use_bias;
            Parameters["depthwise_initializer"] = depthwise_initializer;
            Parameters["pointwise_initializer"] = pointwise_initializer;
            Parameters["bias_initializer"] = bias_initializer;
            Parameters["depthwise_regularizer"] = depthwise_regularizer;
            Parameters["bias_initializer"] = bias_initializer;
            Parameters["pointwise_regularizer"] = pointwise_regularizer;
            Parameters["bias_regularizer"] = bias_regularizer;
            Parameters["activity_regularizer"] = activity_regularizer;
            Parameters["depthwise_constraint"] = depthwise_constraint;
            Parameters["pointwise_constraint"] = pointwise_constraint;
            Parameters["bias_constraint"] = bias_constraint;
            Parameters["input_shape"] = input_shape;

            __self__ = Instance.self.layers.SeparableConv1D;
        }
    }

    public class SeparableConv2D : BaseLayer
    {
        public SeparableConv2D(int filters, Tuple<int, int> kernel_size, Tuple<int, int> strides = null, string padding = "valid", string data_format = "channels_last",
                    Tuple<int, int> dilation_rate = null, int depth_multiplier = 1, string activation = "", bool use_bias = true, string depthwise_initializer = "glorot_uniform",
                    string pointwise_initializer = "glorot_uniform", string bias_initializer = "zeros", string depthwise_regularizer = "",
                    string pointwise_regularizer = "", string bias_regularizer = "", string activity_regularizer = "", string depthwise_constraint = "",
                    string pointwise_constraint = "", string bias_constraint = "", Shape input_shape = null)
        {
            Parameters["filters"] = filters;
            Parameters["kernel_size"] = new Shape(kernel_size.Item1, kernel_size.Item2);
            Parameters["strides"] = strides == null ? new Shape(1, 1) : new Shape(strides.Item1, strides.Item2);
            Parameters["padding"] = padding;
            Parameters["data_format"] = data_format;
            Parameters["dilation_rate"] = dilation_rate == null ? new Shape(1, 1) : new Shape(dilation_rate.Item1, dilation_rate.Item2); ;
            Parameters["depth_multiplier"] = depth_multiplier;
            Parameters["activation"] = activation;
            Parameters["use_bias"] = use_bias;
            Parameters["depthwise_initializer"] = depthwise_initializer;
            Parameters["pointwise_initializer"] = pointwise_initializer;
            Parameters["bias_initializer"] = bias_initializer;
            Parameters["depthwise_regularizer"] = depthwise_regularizer;
            Parameters["bias_initializer"] = bias_initializer;
            Parameters["pointwise_regularizer"] = pointwise_regularizer;
            Parameters["bias_regularizer"] = bias_regularizer;
            Parameters["activity_regularizer"] = activity_regularizer;
            Parameters["depthwise_constraint"] = depthwise_constraint;
            Parameters["pointwise_constraint"] = pointwise_constraint;
            Parameters["bias_constraint"] = bias_constraint;
            Parameters["input_shape"] = input_shape;

            __self__ = Instance.self.layers.SeparableConv2D;
        }
    }

    public class DepthwiseConv2D : BaseLayer
    {
        public DepthwiseConv2D(Tuple<int, int> kernel_size, Tuple<int, int> strides = null, string padding = "valid", 
                    int depth_multiplier = 1, string data_format = "", string activation = "", bool use_bias = true, string depthwise_initializer = "glorot_uniform",
                    string bias_initializer = "zeros", string depthwise_regularizer = "", string bias_regularizer = "", string activity_regularizer = "", 
                    string depthwise_constraint = "", string bias_constraint = "", Shape input_shape = null)
        {
            Parameters["kernel_size"] = new Shape(kernel_size.Item1, kernel_size.Item2);
            Parameters["strides"] = strides == null ? new Shape(1, 1) : new Shape(strides.Item1, strides.Item2);
            Parameters["padding"] = padding;
            Parameters["depth_multiplier"] = depth_multiplier;
            Parameters["data_format"] = data_format;
            Parameters["activation"] = activation;
            Parameters["use_bias"] = use_bias;
            Parameters["depthwise_initializer"] = depthwise_initializer;
            Parameters["bias_initializer"] = bias_initializer;
            Parameters["depthwise_regularizer"] = depthwise_regularizer;
            Parameters["bias_initializer"] = bias_initializer;
            Parameters["bias_regularizer"] = bias_regularizer;
            Parameters["activity_regularizer"] = activity_regularizer;
            Parameters["depthwise_constraint"] = depthwise_constraint;
            Parameters["bias_constraint"] = bias_constraint;
            Parameters["input_shape"] = input_shape;

            __self__ = Instance.self.layers.DepthwiseConv2D;
        }
    }

    public class Conv2DTranspose : BaseLayer
    {
        public Conv2DTranspose(int filters, Tuple<int, int> kernel_size, Tuple<int, int> strides = null, string padding = "valid", Tuple<int, int> output_padding = null,
                    string data_format = "channels_last", Tuple<int, int> dilation_rate = null, string activation = "", bool use_bias = true, string kernel_initializer = "glorot_uniform",
                    string bias_initializer = "zeros", string kernel_regularizer = "", string bias_regularizer = "",
                    string activity_regularizer = "", string kernel_constraint = "", string bias_constraint = "", Shape input_shape = null)
        {
            Parameters["filters"] = filters;
            Parameters["kernel_size"] = new Shape(kernel_size.Item1, kernel_size.Item2);
            Parameters["strides"] = strides == null ? new Shape(1, 1) : new Shape(strides.Item1, strides.Item2);
            Parameters["padding"] = padding;
            Parameters["output_padding"] = output_padding;
            Parameters["data_format"] = data_format;
            Parameters["dilation_rate"] = dilation_rate == null ? new Shape(1, 1) : new Shape(dilation_rate.Item1, dilation_rate.Item2); ;
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

            __self__ = Instance.self.layers.Conv2DTranspose;
        }
    }

    public class Conv3DTranspose : BaseLayer
    {
        public Conv3DTranspose(int filters, Tuple<int, int, int> kernel_size, Tuple<int, int, int> strides = null, string padding = "valid",
                    Tuple<int, int, int> output_padding = null, string data_format = "channels_last", Tuple<int, int, int> dilation_rate = null, 
                    string activation = "", bool use_bias = true, string kernel_initializer = "glorot_uniform",
                    string bias_initializer = "zeros", string kernel_regularizer = "", string bias_regularizer = "",
                    string activity_regularizer = "", string kernel_constraint = "", string bias_constraint = "", Shape input_shape = null)
        {
            Parameters["filters"] = filters;
            Parameters["kernel_size"] = new Shape(kernel_size.Item1, kernel_size.Item2, kernel_size.Item3);
            Parameters["strides"] = strides == null ? new Shape(1, 1, 1) : (strides.Item1, strides.Item2, strides.Item3);
            Parameters["padding"] = padding;
            Parameters["output_padding"] = output_padding;
            Parameters["data_format"] = data_format;
            Parameters["dilation_rate"] = dilation_rate == null ? new Shape(1, 1, 1) : new Shape(dilation_rate.Item1, dilation_rate.Item2, dilation_rate.Item3); ;
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

            __self__ = Instance.self.layers.Conv3DTranspose;
        }
    }

    public class Cropping1D : BaseLayer
    {
        public Cropping1D(Tuple<int,int> cropping, Shape input_shape = null)
        {
            Parameters["cropping"] = cropping == null ? new Shape(1, 1) : new Shape(cropping.Item1, cropping.Item2);
            Parameters["input_shape"] = input_shape;
            __self__ = Instance.self.layers.Cropping1D;
        }
    }

    public class Cropping2D : BaseLayer
    {
        public Cropping2D(Tuple<Tuple<int,int>,Tuple<int, int>> cropping, string data_format = "", Shape input_shape = null)
        {
            Parameters["cropping"] = cropping == null ? new Shape[] { new Shape(1, 1), new Shape(1, 1) }
                                : new Shape[] { new Shape(cropping.Item1.Item1, cropping.Item1.Item2), new Shape(cropping.Item2.Item1, cropping.Item2.Item2) };
            Parameters["data_format"] = data_format;
            Parameters["input_shape"] = input_shape;
            __self__ = Instance.self.layers.Cropping2D;
        }
    }

    public class Cropping3D : BaseLayer
    {
        public Cropping3D(Tuple<Tuple<int, int>, Tuple<int, int>, Tuple<int, int>> cropping, string data_format = "", Shape input_shape = null)
        {
            Parameters["cropping"] = cropping == null ? new Shape[] { new Shape(1, 1), new Shape(1, 1), new Shape(1, 1) }
                                : new Shape[] { new Shape(cropping.Item1.Item1, cropping.Item1.Item2), new Shape(cropping.Item2.Item1, cropping.Item2.Item2), new Shape(cropping.Item3.Item1, cropping.Item3.Item2) };
            Parameters["data_format"] = data_format;
            Parameters["input_shape"] = input_shape;
            __self__ = Instance.self.layers.Cropping3D;
        }
    }

    public class UpSampling1D : BaseLayer
    {
        public UpSampling1D(int size = 2, Shape input_shape = null)
        {
            Parameters["size"] = size;
            Parameters["input_shape"] = input_shape;
            __self__ = Instance.self.layers.UpSampling1D;
        }
    }

    public class UpSampling2D : BaseLayer
    {
        public UpSampling2D(Tuple<int, int> size = null, string data_format = "", string interpolation = "nearest", Shape input_shape = null)
        {
            Parameters["size"] = size == null ? new Shape(2, 2) : new Shape(size.Item1, size.Item2);
            Parameters["data_format"] = data_format;
            Parameters["interpolation"] = interpolation;
            Parameters["input_shape"] = input_shape;
            __self__ = Instance.self.layers.UpSampling2D;
        }
    }

    public class UpSampling3D : BaseLayer
    {
        public UpSampling3D(Tuple<int, int, int> size = null, string data_format = "", Shape input_shape = null)
        {
            Parameters["size"] = size == null ? new Shape(2, 2, 2) : new Shape(size.Item1, size.Item2, size.Item3);
            Parameters["data_format"] = data_format;
            Parameters["input_shape"] = input_shape;
            __self__ = Instance.self.layers.UpSampling3D;
        }
    }

    public class ZeroPadding1D : BaseLayer
    {
        public ZeroPadding1D(int padding = 1, Shape input_shape = null)
        {
            Parameters["padding"] = padding;
            Parameters["input_shape"] = input_shape;
            __self__ = Instance.self.layers.ZeroPadding1D;
        }
    }

    public class ZeroPadding2D : BaseLayer
    {
        public ZeroPadding2D(Tuple<int, int> padding = null, string data_format = "", Shape input_shape = null)
        {
            Parameters["padding"] = padding == null ? new Shape(2, 2) : new Shape(padding.Item1, padding.Item2);
            Parameters["data_format"] = data_format;
            Parameters["input_shape"] = input_shape;
            __self__ = Instance.self.layers.ZeroPadding2D;
        }
    }

    public class ZeroPadding3D : BaseLayer
    {
        public ZeroPadding3D(Tuple<int, int, int> padding = null, string data_format = "", Shape input_shape = null)
        {
            Parameters["padding"] = padding == null ? new Shape(2, 2, 2) : new Shape(padding.Item1, padding.Item2, padding.Item3);
            Parameters["data_format"] = data_format;
            Parameters["input_shape"] = input_shape;
            __self__ = Instance.self.layers.ZeroPadding3D;
        }
    }
}
