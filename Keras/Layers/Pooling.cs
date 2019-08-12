namespace Keras.Layers
{
    using System;

    /// <summary>
    /// Max pooling operation for temporal data.
    /// </summary>
    /// <seealso cref="Keras.Layers.BaseLayer" />
    public class MaxPooling1D : BaseLayer
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="MaxPooling1D"/> class.
        /// </summary>
        /// <param name="pool_size"> Integer, size of the max pooling windows.</param>
        /// <param name="strides"> Integer, or None. Factor by which to downscale. E.g. 2 will halve the input. If None, it will default to pool_size.</param>
        /// <param name="padding"> One of "valid" or "same" (case-insensitive).</param>
        /// <param name="data_format"> A string, one of channels_last (default) or channels_first. The ordering of the dimensions in the inputs. channels_last corresponds to inputs with shape (batch, steps, features) while channels_first corresponds to inputs with shape (batch, features, steps).</param>
        public MaxPooling1D(int pool_size= 2, int? strides= null, string padding= "valid", string data_format= "channels_last")
        {
            Parameters["pool_size"] = pool_size;
            Parameters["strides"] = strides;
            Parameters["padding"] = padding;
            Parameters["data_format"] = data_format;
            PyInstance = Instance.keras.layers.MaxPooling1D;
            Init();
        }
    }

    /// <summary>
    /// Max pooling operation for spatial data.
    /// </summary>
    /// <seealso cref="Keras.Layers.BaseLayer" />
    public class MaxPooling2D : BaseLayer
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="MaxPooling2D"/> class.
        /// </summary>
        /// <param name="pool_size"> integer or tuple of 2 integers, factors by which to downscale (vertical, horizontal). (2, 2) will halve the input in both spatial dimension. If only one integer is specified, the same window length will be used for both dimensions.</param>
        /// <param name="strides"> Integer, tuple of 2 integers, or None. Strides values. If None, it will default to pool_size.</param>
        /// <param name="padding"> One of "valid" or "same" (case-insensitive).</param>
        /// <param name="data_format"> A string, one of channels_last (default) or channels_first. The ordering of the dimensions in the inputs. channels_last corresponds to inputs with shape (batch, height, width, channels) while channels_first corresponds to inputs with shape (batch, channels, height, width). It defaults to the image_data_format value found in your Keras config file at ~/.keras/keras.json. If you never set it, then it will be "channels_last".</param>

        public MaxPooling2D(Tuple<int,int> pool_size = null, Tuple<int, int> strides = null, string padding = "valid", string data_format = "channels_last")
        {
            Parameters["pool_size"] = pool_size == null ? new Shape(2, 2) : new Shape(pool_size.Item1, pool_size.Item2);
            if (strides != null)
                Parameters["strides"] = strides.ToValueTuple();

            Parameters["padding"] = padding;
            Parameters["data_format"] = data_format;
            PyInstance = Instance.keras.layers.MaxPooling2D;
            Init();
        }
    }

    /// <summary>
    /// Max pooling operation for 3D data (spatial or spatio-temporal).
    /// </summary>
    /// <seealso cref="Keras.Layers.BaseLayer" />
    public class MaxPooling3D : BaseLayer
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="MaxPooling3D"/> class.
        /// </summary>
        /// <param name="pool_size"> tuple of 3 integers, factors by which to downscale (dim1, dim2, dim3). (2, 2, 2) will halve the size of the 3D input in each dimension.</param>
        /// <param name="strides"> tuple of 3 integers, or None. Strides values.</param>
        /// <param name="padding"> One of "valid" or "same" (case-insensitive).</param>
        /// <param name="data_format"> A string, one of channels_last (default) or channels_first. The ordering of the dimensions in the inputs. channels_last corresponds to inputs with shape (batch, spatial_dim1, spatial_dim2, spatial_dim3, channels) while channels_first corresponds to inputs with shape(batch, channels, spatial_dim1, spatial_dim2, spatial_dim3). It defaults to the image_data_formatvalue found in your Keras config file at ~/.keras/keras.json. If you never set it, then it will be "channels_last".</param>
        public MaxPooling3D(Tuple<int, int, int> pool_size = null, Tuple<int, int, int> strides = null, string padding = "valid", string data_format = "channels_last")
        {
            Parameters["pool_size"] = pool_size == null ? new Shape(2, 2, 2) : new Shape(pool_size.Item1, pool_size.Item2, pool_size.Item3);
            if (strides != null)
                Parameters["strides"] = strides.ToValueTuple();

            Parameters["padding"] = padding;
            Parameters["data_format"] = data_format;
            PyInstance = Instance.keras.layers.MaxPooling3D;
            Init();
        }
    }

    /// <summary>
    /// Average pooling for temporal data.
    /// </summary>
    /// <seealso cref="Keras.Layers.BaseLayer" />
    public class AveragePooling1D : BaseLayer
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="AveragePooling1D"/> class.
        /// </summary>
        /// <param name="pool_size"> Integer, size of the max pooling windows.</param>
        /// <param name="strides"> Integer, or None. Factor by which to downscale. E.g. 2 will halve the input. If None, it will default to pool_size.</param>
        /// <param name="padding"> One of "valid" or "same" (case-insensitive).</param>
        /// <param name="data_format"> A string, one of channels_last (default) or channels_first. The ordering of the dimensions in the inputs. channels_last corresponds to inputs with shape (batch, steps, features) while channels_first corresponds to inputs with shape (batch, features, steps).</param>
        public AveragePooling1D(int pool_size = 2, int? strides = null, string padding = "valid", string data_format = "channels_last")
        {
            Parameters["pool_size"] = pool_size;
            Parameters["strides"] = strides;
            Parameters["padding"] = padding;
            Parameters["data_format"] = data_format;
            PyInstance = Instance.keras.layers.AveragePooling1D;
            Init();
        }
    }

    /// <summary>
    /// Average pooling operation for spatial data.
    /// </summary>
    /// <seealso cref="Keras.Layers.BaseLayer" />
    public class AveragePooling2D : BaseLayer
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="AveragePooling2D"/> class.
        /// </summary>
        /// <param name="pool_size"> integer or tuple of 2 integers, factors by which to downscale (vertical, horizontal). (2, 2) will halve the input in both spatial dimension. If only one integer is specified, the same window length will be used for both dimensions.</param>
        /// <param name="strides"> Integer, tuple of 2 integers, or None. Strides values. If None, it will default to pool_size.</param>
        /// <param name="padding"> One of "valid" or "same" (case-insensitive).</param>
        /// <param name="data_format"> A string, one of channels_last (default) or channels_first. The ordering of the dimensions in the inputs. channels_last corresponds to inputs with shape (batch, height, width, channels) while channels_first corresponds to inputs with shape (batch, channels, height, width). It defaults to the image_data_format value found in your Keras config file at ~/.keras/keras.json. If you never set it, then it will be "channels_last".</param>
        public AveragePooling2D(Tuple<int, int> pool_size = null, Tuple<int, int> strides = null, string padding = "valid", string data_format = "channels_last")
        {
            Parameters["pool_size"] = pool_size == null ? new Shape(2, 2) : new Shape(pool_size.Item1, pool_size.Item2);
            if (strides != null)
                Parameters["strides"] = strides.ToValueTuple();

            Parameters["padding"] = padding;
            Parameters["data_format"] = data_format;
            PyInstance = Instance.keras.layers.AveragePooling2D;
            Init();
        }
    }

    /// <summary>
    /// Average pooling operation for 3D data (spatial or spatio-temporal).
    /// </summary>
    /// <seealso cref="Keras.Layers.BaseLayer" />
    public class AveragePooling3D : BaseLayer
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="AveragePooling3D"/> class.
        /// </summary>
        /// <param name="pool_size"> tuple of 3 integers, factors by which to downscale (dim1, dim2, dim3). (2, 2, 2) will halve the size of the 3D input in each dimension.</param>
        /// <param name="strides"> tuple of 3 integers, or None. Strides values.</param>
        /// <param name="padding"> One of "valid" or "same" (case-insensitive).</param>
        /// <param name="data_format"> A string, one of channels_last (default) or channels_first. The ordering of the dimensions in the inputs. channels_last corresponds to inputs with shape (batch, spatial_dim1, spatial_dim2, spatial_dim3, channels) while channels_first corresponds to inputs with shape(batch, channels, spatial_dim1, spatial_dim2, spatial_dim3). It defaults to the image_data_formatvalue found in your Keras config file at ~/.keras/keras.json. If you never set it, then it will be "channels_last".</param>
        public AveragePooling3D(Tuple<int, int, int> pool_size = null, Tuple<int, int, int> strides = null, string padding = "valid", string data_format = "channels_last")
        {
            Parameters["pool_size"] = pool_size == null ? new Shape(2, 2, 2) : new Shape(pool_size.Item1, pool_size.Item2, pool_size.Item3);
            if (strides != null)
                Parameters["strides"] = strides.ToValueTuple();

            Parameters["padding"] = padding;
            Parameters["data_format"] = data_format;
            PyInstance = Instance.keras.layers.AveragePooling3D;
            Init();
        }
    }

    /// <summary>
    /// Global max pooling operation for temporal data.
    /// </summary>
    /// <seealso cref="Keras.Layers.BaseLayer" />
    public class GlobalMaxPooling1D : BaseLayer
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="GlobalMaxPooling1D"/> class.
        /// </summary>
        /// <param name="data_format">A string, one of channels_last (default) or channels_first. The ordering of the dimensions in the inputs.  channels_last corresponds to inputs with shape  (batch, steps, features) while channels_first corresponds to inputs with shape  (batch, features, steps).</param>
        public GlobalMaxPooling1D(string data_format = "channels_last")
        {
            Parameters["data_format"] = data_format;
            PyInstance = Instance.keras.layers.GlobalMaxPooling1D;
            Init();
        }
    }

    /// <summary>
    /// Global average pooling operation for temporal data.
    /// </summary>
    /// <seealso cref="Keras.Layers.BaseLayer" />
    public class GlobalAveragePooling1D : BaseLayer
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="GlobalAveragePooling1D"/> class.
        /// </summary>
        /// <param name="data_format">A string, one of channels_last (default) or channels_first. The ordering of the dimensions in the inputs.  channels_last corresponds to inputs with shape  (batch, steps, features) while channels_first corresponds to inputs with shape  (batch, features, steps).</param>
        public GlobalAveragePooling1D(string data_format = "channels_last")
        {
            Parameters["data_format"] = data_format;
            PyInstance = Instance.keras.layers.GlobalAveragePooling1D;
            Init();
        }
    }

    /// <summary>
    /// Global max pooling operation for spatial data.
    /// </summary>
    /// <seealso cref="Keras.Layers.BaseLayer" />
    public class GlobalMaxPooling2D : BaseLayer
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="GlobalMaxPooling2D"/> class.
        /// </summary>
        /// <param name="data_format">A string, one of channels_last (default) or channels_first. The ordering of the dimensions in the inputs.  channels_last corresponds to inputs with shape  (batch, steps, features) while channels_first corresponds to inputs with shape  (batch, features, steps).</param>
        public GlobalMaxPooling2D(string data_format = "channels_last")
        {
            Parameters["data_format"] = data_format;
            PyInstance = Instance.keras.layers.GlobalMaxPooling2D;
            Init();
        }
    }

    /// <summary>
    /// Global average pooling operation for spatial data.
    /// </summary>
    /// <seealso cref="Keras.Layers.BaseLayer" />
    public class GlobalAveragePooling2D : BaseLayer
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="GlobalAveragePooling2D"/> class.
        /// </summary>
        /// <param name="data_format">A string, one of channels_last (default) or channels_first. The ordering of the dimensions in the inputs.  channels_last corresponds to inputs with shape  (batch, steps, features) while channels_first corresponds to inputs with shape  (batch, features, steps).</param>
        public GlobalAveragePooling2D(string data_format = "channels_last")
        {
            Parameters["data_format"] = data_format;
            PyInstance = Instance.keras.layers.GlobalAveragePooling2D;
            Init();
        }
    }

    /// <summary>
    /// Global Max pooling operation for 3D data.
    /// </summary>
    /// <seealso cref="Keras.Layers.BaseLayer" />
    public class GlobalMaxPooling3D : BaseLayer
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="GlobalMaxPooling3D"/> class.
        /// </summary>
        /// <param name="data_format">A string, one of channels_last (default) or channels_first. The ordering of the dimensions in the inputs.  channels_last corresponds to inputs with shape  (batch, steps, features) while channels_first corresponds to inputs with shape  (batch, features, steps).</param>/// <param name="data_format">The data format.</param>
        public GlobalMaxPooling3D(string data_format = "channels_last")
        {
            Parameters["data_format"] = data_format;
            PyInstance = Instance.keras.layers.GlobalMaxPooling3D;
            Init();
        }
    }

    /// <summary>
    /// Global Average pooling operation for 3D data.
    /// </summary>
    /// <seealso cref="Keras.Layers.BaseLayer" />
    public class GlobalAveragePooling3D : BaseLayer
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="GlobalAveragePooling3D"/> class.
        /// </summary>
        /// <param name="data_format">A string, one of channels_last (default) or channels_first. The ordering of the dimensions in the inputs.  channels_last corresponds to inputs with shape  (batch, steps, features) while channels_first corresponds to inputs with shape  (batch, features, steps).</param>
        public GlobalAveragePooling3D(string data_format = "channels_last")
        {
            Parameters["data_format"] = data_format;
            PyInstance = Instance.keras.layers.GlobalAveragePooling3D;
            Init();
        }
    }
}
