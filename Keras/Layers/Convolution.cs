namespace Keras.Layers
{
    using System;

    /// <summary>
    /// 1D convolution layer (e.g. temporal convolution). 
    /// This layer creates a convolution kernel that is convolved with the layer input over a single spatial(or temporal) dimension to produce a tensor of outputs.If use_bias is True, a bias vector is created and added to the outputs.Finally, if activation is not None, it is applied to the outputs as well.
    /// When using this layer as the first layer in a model, provide an input_shape argument (tuple of integers or None, does not include the batch axis), e.g. input_shape=(10, 128) for time series sequences of 10 time steps with 128 features per step in data_format="channels_last", or (None, 128) for variable-length sequences with 128 features per step.
    /// </summary>
    /// <seealso cref="Keras.Layers.BaseLayer" />
    public class Conv1D : BaseLayer
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="Conv1D"/> class.
        /// </summary>
        /// <param name="filters">Integer, the dimensionality of the output space (i.e. the number of output filters in the convolution).</param>
        /// <param name="kernel_size">An integer or tuple/list of a single integer, specifying the length of the 1D convolution window.</param>
        /// <param name="strides">An integer or tuple/list of a single integer, specifying the stride length of the convolution. Specifying any stride value != 1 is incompatible with specifying any dilation_rate value != 1.</param>
        /// <param name="padding">One of "valid", "causal" or "same" (case-insensitive).  "valid" means "no padding". "same" results in padding the input such that the output has the same length as the original input.  "causal" results in causal (dilated) convolutions, e.g. output[t] does not depend on input[t + 1:]. A zero padding is used such that the output has the same length as the original input. Useful when modeling temporal data where the model should not violate the temporal order. See WaveNet: A Generative Model for Raw Audio, section 2.1.</param>
        /// <param name="data_format">A string, one of "channels_last" (default) or "channels_first". The ordering of the dimensions in the inputs.  "channels_last" corresponds to inputs with shape  (batch, steps, channels) (default format for temporal data in Keras) while "channels_first" corresponds to inputs with shape (batch, channels, steps).</param>
        /// <param name="dilation_rate">an integer or tuple/list of a single integer, specifying the dilation rate to use for dilated convolution. Currently, specifying any dilation_rate value != 1 is incompatible with specifying any strides value != 1.</param>
        /// <param name="activation">Activation function to use (see activations). If you don't specify anything, no activation is applied (ie. "linear" activation: a(x) = x).</param>
        /// <param name="use_bias">Boolean, whether the layer uses a bias vector.</param>
        /// <param name="kernel_initializer">Initializer for the kernel weights matrix (see initializers).</param>
        /// <param name="bias_initializer">Initializer for the bias vector (see initializers).</param>
        /// <param name="kernel_regularizer">Regularizer function applied to the kernel weights matrix (see regularizer).</param>
        /// <param name="bias_regularizer">Regularizer function applied to the bias vector (see regularizer).</param>
        /// <param name="activity_regularizer">Regularizer function applied to the output of the layer (its "activation"). (see regularizer).</param>
        /// <param name="kernel_constraint"> Constraint function applied to the kernel matrix (see constraints).</param>
        /// <param name="bias_constraint">Constraint function applied to the bias vector (see constraints).</param>
        /// <param name="input_shape">3D tensor with shape: (batch, steps, channels)</param>
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

            PyInstance = Instance.keras.layers.Conv1D;
            Init();
        }
    }

    /// <summary>
    /// 2D convolution layer (e.g. spatial convolution over images).
    /// This layer creates a convolution kernel that is convolved with the layer input to produce a tensor of outputs.If use_bias is True, a bias vector is created and added to the outputs.Finally, if activation is not None, it is applied to the outputs as well.
    /// When using this layer as the first layer in a model, provide the keyword argument input_shape (tuple of integers, does not include the batch axis), e.g. input_shape=(128, 128, 3) for 128x128 RGB pictures in data_format="channels_last".
    /// </summary>
    /// <seealso cref="Keras.Layers.BaseLayer" />
    public class Conv2D : BaseLayer
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="Conv2D"/> class.
        /// </summary>
        /// <param name="filters"> Integer, the dimensionality of the output space (i.e. the number of output filters in the convolution).</param>
        /// <param name="kernel_size"> An integer or tuple/list of 2 integers, specifying the height and width of the 2D convolution window. Can be a single integer to specify the same value for all spatial dimensions.</param>
        /// <param name="strides"> An integer or tuple/list of 2 integers, specifying the strides of the convolution along the height and width. Can be a single integer to specify the same value for all spatial dimensions. Specifying any stride value != 1 is incompatible with specifying any dilation_rate value != 1.</param>
        /// <param name="padding"> one of "valid" or "same" (case-insensitive). Note that "same" is slightly inconsistent across backends with strides != 1, as described here</param>
        /// <param name="data_format"> A string, one of "channels_last" or "channels_first". The ordering of the dimensions in the inputs. "channels_last" corresponds to inputs with shape (batch, height, width, channels) while "channels_first" corresponds to inputs with shape (batch, channels, height, width). It defaults to the image_data_format value found in your Keras config file at ~/.keras/keras.json. If you never set it, then it will be "channels_last".</param>
        /// <param name="dilation_rate"> an integer or tuple/list of 2 integers, specifying the dilation rate to use for dilated convolution. Can be a single integer to specify the same value for all spatial dimensions. Currently, specifying any dilation_rate value != 1 is incompatible with specifying any stride value != 1.</param>
        /// <param name="activation"> Activation function to use (see activations). If you don't specify anything, no activation is applied (ie. "linear" activation: a(x) = x).</param>
        /// <param name="use_bias"> Boolean, whether the layer uses a bias vector.</param>
        /// <param name="kernel_initializer"> Initializer for the kernel weights matrix (see initializers).</param>
        /// <param name="bias_initializer"> Initializer for the bias vector (see initializers).</param>
        /// <param name="kernel_regularizer"> Regularizer function applied to the kernel weights matrix (see regularizer).</param>
        /// <param name="bias_regularizer"> Regularizer function applied to the bias vector (see regularizer).</param>
        /// <param name="activity_regularizer"> Regularizer function applied to the output of the layer (its "activation"). (see regularizer).</param>
        /// <param name="kernel_constraint"> Constraint function applied to the kernel matrix (see constraints).</param>
        /// <param name="bias_constraint"> Constraint function applied to the bias vector (see constraints).</param>
        /// <param name="input_shape">4D tensor with shape: (batch, channels, rows, cols) if data_format is "channels_first" or 4D tensor with shape: (batch, rows, cols, channels) if data_format is "channels_last".</param>
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

            PyInstance = Instance.keras.layers.Conv2D;
            Init();
        }
    }

    /// <summary>
    /// 3D convolution layer (e.g. spatial convolution over volumes).
    /// This layer creates a convolution kernel that is convolved with the layer input to produce a tensor of outputs.If use_bias is True, a bias vector is created and added to the outputs.Finally, if activation is not None, it is applied to the outputs as well.
    ///  When using this layer as the first layer in a model, provide the keyword argument input_shape (tuple of integers, does not include the batch axis), e.g. input_shape=(128, 128, 128, 1) for 128x128x128 volumes with a single channel, in data_format="channels_last".
    /// </summary>
    /// <seealso cref="Keras.Layers.BaseLayer" />
    public class Conv3D : BaseLayer
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="Conv3D"/> class.
        /// </summary>
        /// <param name="filters"> Integer, the dimensionality of the output space (i.e. the number of output filters in the convolution).</param>
        /// <param name="kernel_size"> An integer or tuple/list of 3 integers, specifying the depth, height and width of the 3D convolution window. Can be a single integer to specify the same value for all spatial dimensions.</param>
        /// <param name="strides"> An integer or tuple/list of 3 integers, specifying the strides of the convolution along each spatial dimension. Can be a single integer to specify the same value for all spatial dimensions. Specifying any stride value != 1 is incompatible with specifying any dilation_ratevalue != 1.</param>
        /// <param name="padding"> one of "valid" or "same" (case-insensitive).</param>
        /// <param name="data_format"> A string, one of "channels_last" or "channels_first". The ordering of the dimensions in the inputs. "channels_last" corresponds to inputs with shape (batch, spatial_dim1, spatial_dim2, spatial_dim3, channels) while "channels_first" corresponds to inputs with shape (batch, channels, spatial_dim1, spatial_dim2, spatial_dim3). It defaults to the image_data_format value found in your Keras config file at ~/.keras/keras.json. If you never set it, then it will be "channels_last".</param>
        /// <param name="dilation_rate"> an integer or tuple/list of 3 integers, specifying the dilation rate to use for dilated convolution. Can be a single integer to specify the same value for all spatial dimensions. Currently, specifying any dilation_rate value != 1 is incompatible with specifying any stride value != 1.</param>
        /// <param name="activation"> Activation function to use (see activations). If you don't specify anything, no activation is applied (ie. "linear" activation: a(x) = x).</param>
        /// <param name="use_bias"> Boolean, whether the layer uses a bias vector.</param>
        /// <param name="kernel_initializer"> Initializer for the kernel weights matrix (see initializers).</param>
        /// <param name="bias_initializer"> Initializer for the bias vector (see initializers).</param>
        /// <param name="kernel_regularizer"> Regularizer function applied to the kernel weights matrix (see regularizer).</param>
        /// <param name="bias_regularizer"> Regularizer function applied to the bias vector (see regularizer).</param>
        /// <param name="activity_regularizer"> Regularizer function applied to the output of the layer (its "activation"). (see regularizer).</param>
        /// <param name="kernel_constraint"> Constraint function applied to the kernel matrix (see constraints).</param>
        /// <param name="bias_constraint"> Constraint function applied to the bias vector (see constraints).</param>
        /// <param name="bias_constraint"> Constraint function applied to the bias vector (see constraints).</param>
        /// <param name="depthwise_constraint"> Constraint function applied to the depthwise kernel matrix (see constraints).</param>
        /// <param name="pointwise_constraint"> Constraint function applied to the pointwise kernel matrix (see constraints).</param>
        /// <param name="bias_constraint"> Constraint function applied to the bias vector (see constraints).</param>
        /// <param name="input_shape">5D tensor with shape: (batch, channels, conv_dim1, conv_dim2, conv_dim3) if data_format is "channels_first" or 5D tensor with shape: (batch, conv_dim1, conv_dim2, conv_dim3, channels) if data_format is "channels_last".</param>
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

            PyInstance = Instance.keras.layers.Conv3D;
            Init();
        }
    }

    /// <summary>
    /// Depthwise separable 1D convolution.
    /// Separable convolutions consist in first performing a depthwise spatial convolution(which acts on each input channel separately) followed by a pointwise convolution which mixes together the resulting output channels.The depth_multiplier argument controls how many output channels are generated per input channel in the depthwise step.
    /// Intuitively, separable convolutions can be understood as a way to factorize a convolution kernel into two smaller kernels, or as an extreme version of an Inception block.
    /// </summary>
    /// <seealso cref="Keras.Layers.BaseLayer" />
    public class SeparableConv1D : BaseLayer
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="SeparableConv1D"/> class.
        /// </summary>
        /// <param name="filters"> Integer, the dimensionality of the output space (i.e. the number of output filters in the convolution).</param>
        /// <param name="kernel_size"> An integer or tuple/list of single integer, specifying the length of the 1D convolution window.</param>
        /// <param name="strides"> An integer or tuple/list of single integer, specifying the stride length of the convolution. Specifying any stride value != 1 is incompatible with specifying any dilation_rate value != 1.</param>
        /// <param name="padding"> one of "valid" or "same" (case-insensitive).</param>
        /// <param name="data_format"> A string, one of "channels_last" or "channels_first". The ordering of the dimensions in the inputs. "channels_last" corresponds to inputs with shape (batch, steps, channels) while "channels_first" corresponds to inputs with shape (batch, channels, steps).</param>
        /// <param name="dilation_rate"> An integer or tuple/list of a single integer, specifying the dilation rate to use for dilated convolution. Currently, specifying any dilation_rate value != 1 is incompatible with specifying any strides value != 1.</param>
        /// <param name="depth_multiplier"> The number of depthwise convolution output channels for each input channel. The total number of depthwise convolution output channels will be equal to filters_in * depth_multiplier.</param>
        /// <param name="activation"> Activation function to use (see activations). If you don't specify anything, no activation is applied (ie. "linear" activation: a(x) = x).</param>
        /// <param name="use_bias"> Boolean, whether the layer uses a bias vector.</param>
        /// <param name="depthwise_initializer"> Initializer for the depthwise kernel matrix (see initializers).</param>
        /// <param name="pointwise_initializer"> Initializer for the pointwise kernel matrix (see initializers).</param>
        /// <param name="bias_initializer"> Initializer for the bias vector (see initializers).</param>
        /// <param name="depthwise_regularizer"> Regularizer function applied to the depthwise kernel matrix (see regularizer).</param>
        /// <param name="pointwise_regularizer"> Regularizer function applied to the pointwise kernel matrix (see regularizer).</param>
        /// <param name="bias_regularizer"> Regularizer function applied to the bias vector (see regularizer).</param>
        /// <param name="activity_regularizer"> Regularizer function applied to the output of the layer (its "activation"). (see regularizer).</param>
        /// <param name="depthwise_constraint"> Constraint function applied to the depthwise kernel matrix (see constraints).</param>
        /// <param name="pointwise_constraint"> Constraint function applied to the pointwise kernel matrix (see constraints).</param>
        /// <param name="bias_constraint"> Constraint function applied to the bias vector (see constraints).</param>
        /// <param name="input_shape">3D tensor with shape: (batch, channels, steps) if data_format is "channels_first" or 3D tensor with shape: (batch, steps, channels) if data_format is "channels_last".</param>
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

            PyInstance = Instance.keras.layers.SeparableConv1D;
            Init();
        }
    }

    /// <summary>
    /// Depthwise separable 2D convolution.
    /// Separable convolutions consist in first performing a depthwise spatial convolution(which acts on each input channel separately) followed by a pointwise convolution which mixes together the resulting output channels.The depth_multiplier argument controls how many output channels are generated per input channel in the depthwise step.
    /// Intuitively, separable convolutions can be understood as a way to factorize a convolution kernel into two smaller kernels, or as an extreme version of an Inception block.
    /// </summary>
    /// <seealso cref="Keras.Layers.BaseLayer" />
    public class SeparableConv2D : BaseLayer
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="SeparableConv2D"/> class.
        /// </summary>
        /// <param name="filters"> Integer, the dimensionality of the output space (i.e. the number of output filters in the convolution).</param>
        /// <param name="kernel_size"> An integer or tuple/list of 2 integers, specifying the height and width of the 2D convolution window. Can be a single integer to specify the same value for all spatial dimensions.</param>
        /// <param name="strides"> An integer or tuple/list of 2 integers, specifying the strides of the convolution along the height and width. Can be a single integer to specify the same value for all spatial dimensions. Specifying any stride value != 1 is incompatible with specifying any dilation_rate value != 1.</param>
        /// <param name="padding"> one of "valid" or "same" (case-insensitive).</param>
        /// <param name="data_format"> A string, one of "channels_last" or "channels_first". The ordering of the dimensions in the inputs. "channels_last" corresponds to inputs with shape (batch, height, width, channels) while "channels_first" corresponds to inputs with shape (batch, channels, height, width). It defaults to the image_data_format value found in your Keras config file at ~/.keras/keras.json. If you never set it, then it will be "channels_last".</param>
        /// <param name="dilation_rate"> An integer or tuple/list of 2 integers, specifying the dilation rate to use for dilated convolution. Currently, specifying any dilation_rate value != 1 is incompatible with specifying any strides value != 1.</param>
        /// <param name="depth_multiplier"> The number of depthwise convolution output channels for each input channel. The total number of depthwise convolution output channels will be equal to filters_in * depth_multiplier.</param>
        /// <param name="activation"> Activation function to use (see activations). If you don't specify anything, no activation is applied (ie. "linear" activation: a(x) = x).</param>
        /// <param name="use_bias"> Boolean, whether the layer uses a bias vector.</param>
        /// <param name="depthwise_initializer"> Initializer for the depthwise kernel matrix (see initializers).</param>
        /// <param name="pointwise_initializer"> Initializer for the pointwise kernel matrix (see initializers).</param>
        /// <param name="bias_initializer"> Initializer for the bias vector (see initializers).</param>
        /// <param name="depthwise_regularizer"> Regularizer function applied to the depthwise kernel matrix (see regularizer).</param>
        /// <param name="pointwise_regularizer"> Regularizer function applied to the pointwise kernel matrix (see regularizer).</param>
        /// <param name="bias_regularizer"> Regularizer function applied to the bias vector (see regularizer).</param>
        /// <param name="activity_regularizer"> Regularizer function applied to the output of the layer (its "activation"). (see regularizer).</param>
        /// <param name="depthwise_constraint"> Constraint function applied to the depthwise kernel matrix (see constraints).</param>
        /// <param name="pointwise_constraint"> Constraint function applied to the pointwise kernel matrix (see constraints).</param>
        /// <param name="bias_constraint"> Constraint function applied to the bias vector (see constraints).</param>
        /// <param name="input_shape">4D tensor with shape: (batch, channels, rows, cols) if data_format is "channels_first" or 4D tensor with shape: (batch, rows, cols, channels) if data_format is "channels_last".</param>
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

            PyInstance = Instance.keras.layers.SeparableConv2D;
            Init();
        }
    }

    /// <summary>
    /// Depthwise separable 2D convolution.    
    /// Depthwise Separable convolutions consists in performing just the first step in a depthwise spatial convolution(which acts on each input channel separately). The depth_multiplier argument controls how many output channels are generated per input channel in the depthwise step.
    /// </summary>
    /// <seealso cref="Keras.Layers.BaseLayer" />
    public class DepthwiseConv2D : BaseLayer
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="DepthwiseConv2D"/> class.
        /// </summary>
        /// <param name="kernel_size"> An integer or tuple/list of 2 integers, specifying the height and width of the 2D convolution window. Can be a single integer to specify the same value for all spatial dimensions.</param>
        /// <param name="strides"> An integer or tuple/list of 2 integers, specifying the strides of the convolution along the height and width. Can be a single integer to specify the same value for all spatial dimensions. Specifying any stride value != 1 is incompatible with specifying any dilation_rate value != 1.</param>
        /// <param name="padding"> one of "valid" or "same" (case-insensitive).</param>
        /// <param name="depth_multiplier"> The number of depthwise convolution output channels for each input channel. The total number of depthwise convolution output channels will be equal to filters_in * depth_multiplier.</param>
        /// <param name="data_format"> A string, one of "channels_last" or "channels_first". The ordering of the dimensions in the inputs. "channels_last" corresponds to inputs with shape (batch, height, width, channels) while "channels_first" corresponds to inputs with shape (batch, channels, height, width). It defaults to the image_data_format value found in your Keras config file at ~/.keras/keras.json. If you never set it, then it will be 'channels_last'.</param>
        /// <param name="activation"> Activation function to use (see activations). If you don't specify anything, no activation is applied (ie. 'linear' activation: a(x) = x).</param>
        /// <param name="use_bias"> Boolean, whether the layer uses a bias vector.</param>
        /// <param name="depthwise_initializer"> Initializer for the depthwise kernel matrix (see initializers).</param>
        /// <param name="bias_initializer"> Initializer for the bias vector (see initializers).</param>
        /// <param name="depthwise_regularizer"> Regularizer function applied to the depthwise kernel matrix (see regularizer).</param>
        /// <param name="bias_regularizer"> Regularizer function applied to the bias vector (see regularizer).</param>
        /// <param name="activity_regularizer"> Regularizer function applied to the output of the layer (its 'activation'). (see regularizer).</param>
        /// <param name="depthwise_constraint"> Constraint function applied to the depthwise kernel matrix (see constraints).</param>
        /// <param name="bias_constraint"> Constraint function applied to the bias vector (see constraints).</param>
        /// <param name="bias_regularizer"> Regularizer function applied to the bias vector (see regularizer).</param>
        /// <param name="activity_regularizer"> Regularizer function applied to the output of the layer (its "activation"). (see regularizer).</param>
        /// <param name="depthwise_constraint"> Constraint function applied to the depthwise kernel matrix (see constraints).</param>
        /// <param name="pointwise_constraint"> Constraint function applied to the pointwise kernel matrix (see constraints).</param>
        /// <param name="bias_constraint"> Constraint function applied to the bias vector (see constraints).</param>
        /// <param name="input_shape">4D tensor with shape: (batch, channels, rows, cols) if data_format is "channels_first" or 4D tensor with shape: (batch, rows, cols, channels) if data_format is "channels_last".</param>
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

            PyInstance = Instance.keras.layers.DepthwiseConv2D;
            Init();
        }
    }

    /// <summary>
    /// Transposed convolution layer (sometimes called Deconvolution).
    /// The need for transposed convolutions generally arises from the desire to use a transformation going in the opposite direction of a normal convolution, i.e., from something that has the shape of the output of some convolution to something that has the shape of its input while maintaining a connectivity pattern that is compatible with said convolution.
    ///  When using this layer as the first layer in a model, provide the keyword argument input_shape (tuple of integers, does not include the batch axis), e.g. input_shape=(128, 128, 3) for 128x128 RGB pictures in data_format="channels_last".
    /// </summary>
    /// <seealso cref="Keras.Layers.BaseLayer" />
    public class Conv2DTranspose : BaseLayer
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="Conv2DTranspose"/> class.
        /// </summary>
        /// <param name="filters"> Integer, the dimensionality of the output space (i.e. the number of output filters in the convolution).</param>
        /// <param name="kernel_size"> An integer or tuple/list of 2 integers, specifying the height and width of the 2D convolution window. Can be a single integer to specify the same value for all spatial dimensions.</param>
        /// <param name="strides"> An integer or tuple/list of 2 integers, specifying the strides of the convolution along the height and width. Can be a single integer to specify the same value for all spatial dimensions. Specifying any stride value != 1 is incompatible with specifying any dilation_rate value != 1.</param>
        /// <param name="padding"> one of "valid" or "same" (case-insensitive).</param>
        /// <param name="output_padding"> An integer or tuple/list of 2 integers, specifying the amount of padding along the height and width of the output tensor. Can be a single integer to specify the same value for all spatial dimensions. The amount of output padding along a given dimension must be lower than the stride along that same dimension. If set to None (default), the output shape is inferred.</param>
        /// <param name="data_format"> A string, one of "channels_last" or "channels_first". The ordering of the dimensions in the inputs. "channels_last" corresponds to inputs with shape (batch, height, width, channels) while "channels_first" corresponds to inputs with shape (batch, channels, height, width). It defaults to the image_data_format value found in your Keras config file at ~/.keras/keras.json. If you never set it, then it will be "channels_last".</param>
        /// <param name="dilation_rate"> an integer or tuple/list of 2 integers, specifying the dilation rate to use for dilated convolution. Can be a single integer to specify the same value for all spatial dimensions. Currently, specifying any dilation_rate value != 1 is incompatible with specifying any stride value != 1.</param>
        /// <param name="activation"> Activation function to use (see activations). If you don't specify anything, no activation is applied (ie. "linear" activation: a(x) = x).</param>
        /// <param name="use_bias"> Boolean, whether the layer uses a bias vector.</param>
        /// <param name="kernel_initializer"> Initializer for the kernel weights matrix (see initializers).</param>
        /// <param name="bias_initializer"> Initializer for the bias vector (see initializers).</param>
        /// <param name="kernel_regularizer"> Regularizer function applied to the kernel weights matrix (see regularizer).</param>
        /// <param name="bias_regularizer"> Regularizer function applied to the bias vector (see regularizer).</param>
        /// <param name="activity_regularizer"> Regularizer function applied to the output of the layer (its "activation"). (see regularizer).</param>
        /// <param name="kernel_constraint"> Constraint function applied to the kernel matrix (see constraints).</param>
        /// <param name="bias_constraint"> Constraint function applied to the bias vector (see constraints).</param>
        /// <param name="depthwise_constraint"> Constraint function applied to the depthwise kernel matrix (see constraints).</param>
        /// <param name="pointwise_constraint"> Constraint function applied to the pointwise kernel matrix (see constraints).</param>
        /// <param name="bias_constraint"> Constraint function applied to the bias vector (see constraints).</param>
        /// <param name="input_shape">4D tensor with shape: (batch, channels, rows, cols) if data_format is "channels_first" or 4D tensor with shape: (batch, rows, cols, channels) if data_format is "channels_last".</param>
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

            PyInstance = Instance.keras.layers.Conv2DTranspose;
            Init();
        }
    }

    /// <summary>
    /// Transposed convolution layer (sometimes called Deconvolution).
    /// The need for transposed convolutions generally arises from the desire to use a transformation going in the opposite direction of a normal convolution, i.e., from something that has the shape of the output of some convolution to something that has the shape of its input while maintaining a connectivity pattern that is compatible with said convolution.
    ///   When using this layer as the first layer in a model, provide the keyword argument input_shape (tuple of integers, does not include the batch axis), e.g. input_shape=(128, 128, 128, 3) for a 128x128x128 volume with 3 channels if data_format="channels_last".
    /// </summary>
    /// <seealso cref="Keras.Layers.BaseLayer" />
    public class Conv3DTranspose : BaseLayer
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="Conv3DTranspose"/> class.
        /// </summary>
        /// <param name="filters"> Integer, the dimensionality of the output space (i.e. the number of output filters in the convolution).</param>
        /// <param name="kernel_size"> An integer or tuple/list of 3 integers, specifying the depth, height and width of the 3D convolution window. Can be a single integer to specify the same value for all spatial dimensions.</param>
        /// <param name="strides"> An integer or tuple/list of 3 integers, specifying the strides of the convolution along the depth, height and width. Can be a single integer to specify the same value for all spatial dimensions. Specifying any stride value != 1 is incompatible with specifying any dilation_ratevalue != 1.</param>
        /// <param name="padding"> one of "valid" or "same" (case-insensitive).</param>
        /// <param name="output_padding"> An integer or tuple/list of 3 integers, specifying the amount of padding along the depth, height, and width. Can be a single integer to specify the same value for all spatial dimensions. The amount of output padding along a given dimension must be lower than the stride along that same dimension. If set to None (default), the output shape is inferred.</param>
        /// <param name="data_format"> A string, one of "channels_last" or "channels_first". The ordering of the dimensions in the inputs. "channels_last" corresponds to inputs with shape (batch, depth, height, width, channels) while "channels_first" corresponds to inputs with shape (batch, channels, depth, height, width). It defaults to the image_data_format value found in your Keras config file at ~/.keras/keras.json. If you never set it, then it will be "channels_last".</param>
        /// <param name="dilation_rate"> an integer or tuple/list of 3 integers, specifying the dilation rate to use for dilated convolution. Can be a single integer to specify the same value for all spatial dimensions. Currently, specifying any dilation_rate value != 1 is incompatible with specifying any stride value != 1.</param>
        /// <param name="activation"> Activation function to use (see activations). If you don't specify anything, no activation is applied (ie. "linear" activation: a(x) = x).</param>
        /// <param name="use_bias"> Boolean, whether the layer uses a bias vector.</param>
        /// <param name="kernel_initializer"> Initializer for the kernel weights matrix (see initializers).</param>
        /// <param name="bias_initializer"> Initializer for the bias vector (see initializers).</param>
        /// <param name="kernel_regularizer"> Regularizer function applied to the kernel weights matrix (see regularizer).</param>
        /// <param name="bias_regularizer"> Regularizer function applied to the bias vector (see regularizer).</param>
        /// <param name="activity_regularizer"> Regularizer function applied to the output of the layer (its "activation"). (see regularizer).</param>
        /// <param name="kernel_constraint"> Constraint function applied to the kernel matrix (see constraints).</param>
        /// <param name="bias_constraint"> Constraint function applied to the bias vector (see constraints).</param>
        /// <param name="input_shape">5D tensor with shape: (batch, channels, depth, rows, cols) if data_format is "channels_first" or 5D tensor with shape: (batch, depth, rows, cols, channels) if data_format is "channels_last".</param>
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

            PyInstance = Instance.keras.layers.Conv3DTranspose;
            Init();
        }
    }

    /// <summary>
    /// Cropping layer for 1D input (e.g. temporal sequence).    It crops along the time dimension(axis 1).
    /// </summary>
    /// <seealso cref="Keras.Layers.BaseLayer" />
    public class Cropping1D : BaseLayer
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="Cropping1D"/> class.
        /// </summary>
        /// <param name="cropping">int or tuple of int (length 2) How many units should be trimmed off at the beginning and end of the cropping dimension (axis 1). If a single int is provided, the same value will be used for both.</param>
        /// <param name="input_shape">3D tensor with shape (batch, axis_to_crop, features)</param>
        public Cropping1D(Tuple<int,int> cropping, Shape input_shape = null)
        {
            Parameters["cropping"] = cropping == null ? new Shape(1, 1) : new Shape(cropping.Item1, cropping.Item2);
            Parameters["input_shape"] = input_shape;
            PyInstance = Instance.keras.layers.Cropping1D;
            Init();
        }
    }

    /// <summary>
    /// Cropping layer for 2D input (e.g. picture).    It crops along spatial dimensions, i.e.height and width.
    /// </summary>
    /// <seealso cref="Keras.Layers.BaseLayer" />
    public class Cropping2D : BaseLayer
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="Cropping2D"/> class.
        /// </summary>
        /// <param name="cropping">int, or tuple of 2 ints, or tuple of 2 tuples of 2 ints.</param>
        /// <param name="data_format">A string, one of "channels_last" or "channels_first". The ordering of the dimensions in the inputs.  "channels_last" corresponds to inputs with shape  (batch, height, width, channels) while "channels_first" corresponds to inputs with shape  (batch, channels, height, width). It defaults to the image_data_format value found in your Keras config file at ~/.keras/keras.json. If you never set it, then it will be "channels_last".</param>
        /// <param name="input_shape">4D tensor with shape: - If data_format is "channels_last":  (batch, rows, cols, channels) - If data_format is "channels_first":  (batch, channels, rows, cols)</param>
        public Cropping2D(Tuple<Tuple<int,int>,Tuple<int, int>> cropping, string data_format = "", Shape input_shape = null)
        {
            Parameters["cropping"] = cropping == null ? new Shape[] { new Shape(1, 1), new Shape(1, 1) }
                                : new Shape[] { new Shape(cropping.Item1.Item1, cropping.Item1.Item2), new Shape(cropping.Item2.Item1, cropping.Item2.Item2) };
            Parameters["data_format"] = data_format;
            Parameters["input_shape"] = input_shape;
            PyInstance = Instance.keras.layers.Cropping2D;
            Init();
        }
    }

    /// <summary>
    /// Cropping layer for 3D data (e.g. spatial or spatio-temporal).
    /// </summary>
    /// <seealso cref="Keras.Layers.BaseLayer" />
    public class Cropping3D : BaseLayer
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="Cropping3D"/> class.
        /// </summary>
        /// <param name="cropping">int, or tuple of 3 ints, or tuple of 3 tuples of 2 ints.</param>
        /// <param name="data_format">A string, one of "channels_last" or "channels_first". The ordering of the dimensions in the inputs.  "channels_last" corresponds to inputs with shape  (batch, spatial_dim1, spatial_dim2, spatial_dim3, channels) while "channels_first" corresponds to inputs with shape  (batch, channels, spatial_dim1, spatial_dim2, spatial_dim3). It defaults to the image_data_format value found in your Keras config file at ~/.keras/keras.json. If you never set it, then it will be "channels_last".</param>
        /// <param name="input_shape">5D tensor with shape: - If data_format is "channels_last":  (batch, first_axis_to_crop, second_axis_to_crop, third_axis_to_crop,        depth) - If data_format is "channels_first":  (batch, depth,        first_axis_to_crop, second_axis_to_crop, third_axis_to_crop)</param>
        public Cropping3D(Tuple<Tuple<int, int>, Tuple<int, int>, Tuple<int, int>> cropping, string data_format = "", Shape input_shape = null)
        {
            Parameters["cropping"] = cropping == null ? new Shape[] { new Shape(1, 1), new Shape(1, 1), new Shape(1, 1) }
                                : new Shape[] { new Shape(cropping.Item1.Item1, cropping.Item1.Item2), new Shape(cropping.Item2.Item1, cropping.Item2.Item2), new Shape(cropping.Item3.Item1, cropping.Item3.Item2) };
            Parameters["data_format"] = data_format;
            Parameters["input_shape"] = input_shape;
            PyInstance = Instance.keras.layers.Cropping3D;
            Init();
        }
    }

    /// <summary>
    /// Upsampling layer for 1D inputs.    Repeats each temporal step size times along the time axis.
    /// </summary>
    /// <seealso cref="Keras.Layers.BaseLayer" />
    public class UpSampling1D : BaseLayer
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="UpSampling1D"/> class.
        /// </summary>
        /// <param name="size">integer. Upsampling factor.</param>
        /// <param name="input_shape">3D tensor with shape: (batch, steps, features).</param>
        public UpSampling1D(int size = 2, Shape input_shape = null)
        {
            Parameters["size"] = size;
            Parameters["input_shape"] = input_shape;
            PyInstance = Instance.keras.layers.UpSampling1D;
            Init();
        }
    }

    /// <summary>
    /// Upsampling layer for 2D inputs.    Repeats the rows and columns of the data by size[0] and size[1] respectively.
    /// </summary>
    /// <seealso cref="Keras.Layers.BaseLayer" />
    public class UpSampling2D : BaseLayer
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="UpSampling2D"/> class.
        /// </summary>
        /// <param name="size">int, or tuple of 2 integers. The upsampling factors for rows and columns.</param>
        /// <param name="data_format">A string, one of "channels_last" or "channels_first". The ordering of the dimensions in the inputs.  "channels_last" corresponds to inputs with shape  (batch, height, width, channels) while "channels_first" corresponds to inputs with shape  (batch, channels, height, width). It defaults to the image_data_format value found in your Keras config file at ~/.keras/keras.json. If you never set it, then it will be "channels_last".</param>
        /// <param name="interpolation">A string, one of nearest or bilinear. Note that CNTK does not support yet the bilinear upscaling and that with Theano, only size=(2, 2) is possible.</param>
        /// <param name="input_shape">4D tensor with shape: - If data_format is "channels_last":  (batch, rows, cols, channels) - If data_format is "channels_first":  (batch, channels, rows, cols)</param>
        public UpSampling2D(Tuple<int, int> size = null, string data_format = "", string interpolation = "nearest", Shape input_shape = null)
        {
            Parameters["size"] = size == null ? new Shape(2, 2) : new Shape(size.Item1, size.Item2);
            Parameters["data_format"] = data_format;
            Parameters["interpolation"] = interpolation;
            Parameters["input_shape"] = input_shape;
            PyInstance = Instance.keras.layers.UpSampling2D;
            Init();
        }
    }

    /// <summary>
    /// Upsampling layer for 3D inputs.    Repeats the 1st, 2nd and 3rd dimensions of the data by size[0], size[1] and size[2] respectively.
    /// </summary>
    /// <seealso cref="Keras.Layers.BaseLayer" />
    public class UpSampling3D : BaseLayer
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="UpSampling3D"/> class.
        /// </summary>
        /// <param name="size">int, or tuple of 3 integers. The upsampling factors for dim1, dim2 and dim3.</param>
        /// <param name="data_format">A string, one of "channels_last" or "channels_first". The ordering of the dimensions in the inputs.  "channels_last" corresponds to inputs with shape  (batch, spatial_dim1, spatial_dim2, spatial_dim3, channels) while "channels_first" corresponds to inputs with shape  (batch, channels, spatial_dim1, spatial_dim2, spatial_dim3). It defaults to the image_data_format value found in your Keras config file at ~/.keras/keras.json. If you never set it, then it will be "channels_last".</param>
        /// <param name="input_shape">5D tensor with shape: - If data_format is "channels_last":  (batch, dim1, dim2, dim3, channels) - If data_format is "channels_first":  (batch, channels, dim1, dim2, dim3)</param>
        public UpSampling3D(Tuple<int, int, int> size = null, string data_format = "", Shape input_shape = null)
        {
            Parameters["size"] = size == null ? new Shape(2, 2, 2) : new Shape(size.Item1, size.Item2, size.Item3);
            Parameters["data_format"] = data_format;
            Parameters["input_shape"] = input_shape;
            PyInstance = Instance.keras.layers.UpSampling3D;
            Init();
        }
    }

    /// <summary>
    /// Zero-padding layer for 1D input (e.g. temporal sequence).
    /// </summary>
    /// <seealso cref="Keras.Layers.BaseLayer" />
    public class ZeroPadding1D : BaseLayer
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="ZeroPadding1D"/> class.
        /// </summary>
        /// <param name="padding"> int, or tuple of int (length 2), or dictionary.</param>
        /// <param name="input_shape">3D tensor with shape (batch, axis_to_pad, features)</param>
        public ZeroPadding1D(int padding = 1, Shape input_shape = null)
        {
            Parameters["padding"] = padding;
            Parameters["input_shape"] = input_shape;
            PyInstance = Instance.keras.layers.ZeroPadding1D;
            Init();
        }
    }

    /// <summary>
    /// Zero-padding layer for 2D input (e.g. picture).    This layer can add rows and columns of zeros at the top, bottom, left and right side of an image tensor.
    /// </summary>
    /// <seealso cref="Keras.Layers.BaseLayer" />
    public class ZeroPadding2D : BaseLayer
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="ZeroPadding2D"/> class.
        /// </summary>
        /// <param name="padding">int, or tuple of 2 ints, or tuple of 2 tuples of 2 ints.</param>
        /// <param name="data_format">A string, one of "channels_last" or "channels_first". The ordering of the dimensions in the inputs.  "channels_last" corresponds to inputs with shape  (batch, height, width, channels) while "channels_first" corresponds to inputs with shape  (batch, channels, height, width). It defaults to the image_data_format value found in your Keras config file at ~/.keras/keras.json. If you never set it, then it will be "channels_last".</param>
        /// <param name="input_shape">4D tensor with shape: - If data_format is "channels_last":  (batch, rows, cols, channels) - If data_format is "channels_first":  (batch, channels, rows, cols)</param>
        public ZeroPadding2D(Tuple<int, int> padding = null, string data_format = "", Shape input_shape = null)
        {
            Parameters["padding"] = padding == null ? new Shape(2, 2) : new Shape(padding.Item1, padding.Item2);
            Parameters["data_format"] = data_format;
            Parameters["input_shape"] = input_shape;
            PyInstance = Instance.keras.layers.ZeroPadding2D;
            Init();
        }
    }

    /// <summary>
    /// Zero-padding layer for 3D data (spatial or spatio-temporal).
    /// </summary>
    /// <seealso cref="Keras.Layers.BaseLayer" />
    public class ZeroPadding3D : BaseLayer
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="ZeroPadding3D"/> class.
        /// </summary>
        /// <param name="padding"> int, or tuple of 3 ints, or tuple of 3 tuples of 2 ints.</param>
        /// <param name="data_format">A string, one of "channels_last" or "channels_first". The ordering of the dimensions in the inputs.  "channels_last" corresponds to inputs with shape  (batch, spatial_dim1, spatial_dim2, spatial_dim3, channels) while "channels_first" corresponds to inputs with shape  (batch, channels, spatial_dim1, spatial_dim2, spatial_dim3). It defaults to the image_data_format value found in your Keras config file at ~/.keras/keras.json. If you never set it, then it will be "channels_last".</param>
        /// <param name="input_shape">5D tensor with shape: - If data_format is "channels_last":  (batch, first_axis_to_pad, second_axis_to_pad, third_axis_to_pad,        depth) - If data_format is "channels_first":  (batch, depth,        first_axis_to_pad, second_axis_to_pad, third_axis_to_pad)</param>
        public ZeroPadding3D(Tuple<int, int, int> padding = null, string data_format = "", Shape input_shape = null)
        {
            Parameters["padding"] = padding == null ? new Shape(2, 2, 2) : new Shape(padding.Item1, padding.Item2, padding.Item3);
            Parameters["data_format"] = data_format;
            Parameters["input_shape"] = input_shape;
            PyInstance = Instance.keras.layers.ZeroPadding3D;
            Init();
        }
    }
}
