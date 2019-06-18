using System;
using System.Collections.Generic;
using System.Text;
using Numpy;
using Numpy.Models;
using Python.Runtime;

namespace Keras.Layers
{
    public class Input : BaseLayer
    {
        public Input(Shape shape, Shape batch_shape, string name = "", string dtype = "float32", bool sparse = false, NDarray tensor = null)
        {
            this["shape"] = shape;
            this["batch_shape"] = batch_shape;
            this["name"] = name;
            this["dtype"] = dtype;
            this["sparse"] = sparse;
            this["tensor"] = tensor;

            __self__ = Instance.self.layers.Input;
        }
    }

    public class Dense : BaseLayer
    {
        public Dense(int units, int? input_dim = null, string activation= "", bool use_bias= true, string kernel_initializer= "glorot_uniform", 
                    string bias_initializer= "zeros", string kernel_regularizer= "", string bias_regularizer= "", 
                    string activity_regularizer= "", string kernel_constraint= "", string bias_constraint= "", Shape input_shape = null)
        {
            this["units"] = units;
            this["input_dim"] = input_dim;
            this["activation"] = activation;
            this["use_bias"] = use_bias;
            this["kernel_initializer"] = kernel_initializer;
            this["bias_initializer"] = bias_initializer;
            this["kernel_regularizer"] = kernel_regularizer;
            this["bias_regularizer"] = bias_regularizer;
            this["activity_regularizer"] = activity_regularizer;
            this["kernel_constraint"] = kernel_constraint;
            this["bias_constraint"] = bias_constraint;
            Parameters["input_shape"] = input_shape;
            __self__ = Instance.self.layers.Dense;
        }

    }

    public class Activation : BaseLayer
    {
        public Activation(string act, Shape input_shape = null)
        {
            Parameters["activation"] = act;
            Parameters["input_shape"] = input_shape;
            __self__ = Instance.self.layers.Activation;
        }
    }

    public class Dropout : BaseLayer
    {
        public Dropout(double rate, Shape noise_shape = null, int? seed = null)
        {
            Parameters["rate"] = rate;
            Parameters["noise_shape"] = noise_shape;
            Parameters["seed"] = seed;
            __self__ = Instance.self.layers.Dropout;
        }
    }

    public class Flatten : BaseLayer
    {
        public Flatten(string data_format = "channels_last")
        {
            Parameters["data_format"] = data_format;
            __self__ = Instance.self.layers.Flatten;
        }
    }

    public class Reshape : BaseLayer
    {
        public Reshape(Shape target_shape, Shape input_shape = null)
        {
            Parameters["target_shape"] = target_shape;
            Parameters["input_shape"] = input_shape;
            __self__ = Instance.self.layers.Reshape;
        }
    }

    public class Permute : BaseLayer
    {
        public Permute(int dims, Shape input_shape = null)
        {
            Parameters["dims"] = dims;
            Parameters["input_shape"] = input_shape;
            __self__ = Instance.self.layers.Permute;
        }
    }

    public class RepeatVector : BaseLayer
    {
        public RepeatVector(int n, Shape input_shape = null)
        {
            Parameters["n"] = n;
            Parameters["input_shape"] = input_shape;
            __self__ = Instance.self.layers.RepeatVector;
        }
    }

    public class Lambda : BaseLayer
    {
        public Lambda(object function, Shape output_shape = null, NDarray mask = null, Dictionary<string, object> arguments = null, Shape input_shape = null)
        {
            Parameters["function"] = function;
            Parameters["output_shape"] = output_shape;
            Parameters["mask"] = mask;
            Parameters["arguments"] = arguments;
            Parameters["input_shape"] = input_shape;

            __self__ = Instance.self.layers.Lambda;
        }
    }

    public class ActivityRegularization : BaseLayer
    {
        public ActivityRegularization(float l1= 0.0f, float l2= 0.0f, Shape input_shape = null)
        {
            Parameters["l1"] = l1;
            Parameters["l2"] = l2;
            Parameters["input_shape"] = input_shape;
            __self__ = Instance.self.layers.ActivityRegularization;
        }
    }

    public class Masking : BaseLayer
    {
        public Masking(float mask_value = 0.0f)
        {
            Parameters["mask_value"] = mask_value;
            __self__ = Instance.self.layers.Masking;
        }
    }

    public class SpatialDropout1D : BaseLayer
    {
        public SpatialDropout1D(float rate, Shape input_shape = null)
        {
            Parameters["rate"] = rate;
            Parameters["input_shape"] = input_shape;
            __self__ = Instance.self.layers.SpatialDropout1D;
        }
    }

    public class SpatialDropout2D : BaseLayer
    {
        public SpatialDropout2D(float rate, string data_format = "", Shape input_shape = null)
        {
            Parameters["rate"] = rate;
            Parameters["input_shape"] = data_format;
            Parameters["input_shape"] = input_shape;
            __self__ = Instance.self.layers.SpatialDropout2D;
        }
    }

    public class SpatialDropout3D : BaseLayer
    {
        public SpatialDropout3D(float rate, string data_format = "", Shape input_shape = null)
        {
            Parameters["rate"] = rate;
            Parameters["input_shape"] = data_format;
            Parameters["input_shape"] = input_shape;
            __self__ = Instance.self.layers.SpatialDropout3D;
        }
    }
}

