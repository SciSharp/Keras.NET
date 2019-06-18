using System;
using System.Collections.Generic;
using System.Text;

namespace Keras.Initializer
{
    public class Zeros : Base
    {
        public Zeros()
        {
            __self__ = Instance.self.initializers.Zeros;
        }
    }

    public class Ones : Base
    {
        public Ones()
        {
            __self__ = Instance.self.initializers.Zeros;
        }
    }

    public class Constant : Base
    {
        public Constant(float value = 0)
        {
            Parameters["value"] = value;
            __self__ = Instance.self.initializers.Constant;
        }
    }

    public class RandomNormal : Base
    {
        public RandomNormal(float mean= 0.0f, float stddev= 0.05f, int? seed= null)
        {
            Parameters["mean"] = mean;
            Parameters["stddev"] = stddev;
            Parameters["seed"] = seed;
            __self__ = Instance.self.initializers.RandomNormal;
        }
    }

    public class RandomUniform : Base
    {
        public RandomUniform(float minval = -0.05f, float maxval = 0.05f, int? seed = null)
        {
            Parameters["minval"] = minval;
            Parameters["maxval"] = maxval;
            Parameters["seed"] = seed;
            __self__ = Instance.self.initializers.RandomUniform;
        }
    }

    public class TruncatedNormal : Base
    {
        public TruncatedNormal(float mean = 0.0f, float stddev = 0.05f, int? seed = null)
        {
            Parameters["mean"] = mean;
            Parameters["stddev"] = stddev;
            Parameters["seed"] = seed;
            __self__ = Instance.self.initializers.TruncatedNormal;
        }
    }

    public class VarianceScaling : Base
    {
        public VarianceScaling(float scale = 1.0f, string mode = "fan_in", string distribution = "normal", int? seed = null)
        {
            Parameters["scale"] = scale;
            Parameters["mode"] = mode;
            Parameters["distribution"] = distribution;
            Parameters["seed"] = seed;
            __self__ = Instance.self.initializers.VarianceScaling;
        }
    }

    public class Orthogonal : Base
    {
        public Orthogonal(float gain = 1.0f, int? seed = null)
        {
            Parameters["gain"] = gain;
            Parameters["seed"] = seed;
            __self__ = Instance.self.initializers.Orthogonal;
        }
    }

    public class Identity : Base
    {
        public Identity(float gain = 1.0f)
        {
            Parameters["gain"] = gain;
            __self__ = Instance.self.initializers.Identity;
        }
    }

    public class LecunUniform : Base
    {
        public LecunUniform(int? seed = null)
        {
            Parameters["seed"] = seed;
            __self__ = Instance.self.initializers.lecun_uniform;
        }
    }

    public class GlorotNormal : Base
    {
        public GlorotNormal(int? seed = null)
        {
            Parameters["seed"] = seed;
            __self__ = Instance.self.initializers.glorot_normal;
        }
    }

    public class GlorotUniform : Base
    {
        public GlorotUniform(int? seed = null)
        {
            Parameters["seed"] = seed;
            __self__ = Instance.self.initializers.glorot_uniform;
        }
    }

    public class HeUniform : Base
    {
        public HeUniform(int? seed = null)
        {
            Parameters["seed"] = seed;
            __self__ = Instance.self.initializers.he_uniform;
        }
    }

    public class HeNormal : Base
    {
        public HeNormal(int? seed = null)
        {
            Parameters["seed"] = seed;
            __self__ = Instance.self.initializers.he_normal;
        }
    }

    public class LecunNormal : Base
    {
        public LecunNormal(int? seed = null)
        {
            Parameters["seed"] = seed;
            __self__ = Instance.self.initializers.lecun_normal;
        }
    }
}
