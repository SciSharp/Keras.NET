using System;
using System.Collections.Generic;
using System.Text;

namespace Keras.Layers
{
    public class MaxPooling1D : BaseLayer
    {
        public MaxPooling1D(int pool_size= 2, int? strides= null, string padding= "valid", string data_format= "channels_last")
        {
            Parameters["pool_size"] = pool_size;
            Parameters["strides"] = strides;
            Parameters["padding"] = padding;
            Parameters["data_format"] = data_format;
            __self__ = Instance.self.layers.MaxPooling1D;
        }
    }

    public class MaxPooling2D : BaseLayer
    {
        public MaxPooling2D(Tuple<int,int> pool_size = null, Tuple<int, int> strides = null, string padding = "valid", string data_format = "channels_last")
        {
            Parameters["pool_size"] = pool_size == null ? (2, 2) : pool_size.ToValueTuple();
            if (strides != null)
                Parameters["strides"] = strides.ToValueTuple();

            Parameters["padding"] = padding;
            Parameters["data_format"] = data_format;
            __self__ = Instance.self.layers.MaxPooling2D;
        }
    }

    public class MaxPooling3D : BaseLayer
    {
        public MaxPooling3D(Tuple<int, int, int> pool_size = null, Tuple<int, int, int> strides = null, string padding = "valid", string data_format = "channels_last")
        {
            Parameters["pool_size"] = pool_size == null ? (2, 2, 2) : pool_size.ToValueTuple();
            if (strides != null)
                Parameters["strides"] = strides.ToValueTuple();

            Parameters["padding"] = padding;
            Parameters["data_format"] = data_format;
            __self__ = Instance.self.layers.MaxPooling3D;
        }
    }

    public class AveragePooling1D : BaseLayer
    {
        public AveragePooling1D(int pool_size = 2, int? strides = null, string padding = "valid", string data_format = "channels_last")
        {
            Parameters["pool_size"] = pool_size;
            Parameters["strides"] = strides;
            Parameters["padding"] = padding;
            Parameters["data_format"] = data_format;
            __self__ = Instance.self.layers.AveragePooling1D;
        }
    }

    public class AveragePooling2D : BaseLayer
    {
        public AveragePooling2D(Tuple<int, int> pool_size = null, Tuple<int, int> strides = null, string padding = "valid", string data_format = "channels_last")
        {
            Parameters["pool_size"] = pool_size == null ? (2, 2) : pool_size.ToValueTuple();
            if (strides != null)
                Parameters["strides"] = strides.ToValueTuple();

            Parameters["padding"] = padding;
            Parameters["data_format"] = data_format;
            __self__ = Instance.self.layers.AveragePooling2D;
        }
    }

    public class AveragePooling3D : BaseLayer
    {
        public AveragePooling3D(Tuple<int, int, int> pool_size = null, Tuple<int, int, int> strides = null, string padding = "valid", string data_format = "channels_last")
        {
            Parameters["pool_size"] = pool_size == null ? (2, 2, 2) : pool_size.ToValueTuple();
            if (strides != null)
                Parameters["strides"] = strides.ToValueTuple();

            Parameters["padding"] = padding;
            Parameters["data_format"] = data_format;
            __self__ = Instance.self.layers.AveragePooling3D;
        }
    }

    public class GlobalMaxPooling1D : BaseLayer
    {
        public GlobalMaxPooling1D(string data_format = "channels_last")
        {
            Parameters["data_format"] = data_format;
            __self__ = Instance.self.layers.GlobalMaxPooling1D;
        }
    }

    public class GlobalAveragePooling1D : BaseLayer
    {
        public GlobalAveragePooling1D(string data_format = "channels_last")
        {
            Parameters["data_format"] = data_format;
            __self__ = Instance.self.layers.GlobalAveragePooling1D;
        }
    }

    public class GlobalMaxPooling2D : BaseLayer
    {
        public GlobalMaxPooling2D(string data_format = "channels_last")
        {
            Parameters["data_format"] = data_format;
            __self__ = Instance.self.layers.GlobalMaxPooling2D;
        }
    }

    public class GlobalAveragePooling2D : BaseLayer
    {
        public GlobalAveragePooling2D(string data_format = "channels_last")
        {
            Parameters["data_format"] = data_format;
            __self__ = Instance.self.layers.GlobalAveragePooling2D;
        }
    }

    public class GlobalMaxPooling3D : BaseLayer
    {
        public GlobalMaxPooling3D(string data_format = "channels_last")
        {
            Parameters["data_format"] = data_format;
            __self__ = Instance.self.layers.GlobalMaxPooling3D;
        }
    }

    public class GlobalAveragePooling3D : BaseLayer
    {
        public GlobalAveragePooling3D(string data_format = "channels_last")
        {
            Parameters["data_format"] = data_format;
            __self__ = Instance.self.layers.GlobalAveragePooling3D;
        }
    }
}
