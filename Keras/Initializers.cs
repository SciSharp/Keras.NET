namespace Keras.Initializer
{
    /// <summary>
    /// Initializer that generates tensors initialized to 0.
    /// </summary>
    /// <seealso cref="Keras.Base" />
    public class Zeros : Base
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="Zeros"/> class.
        /// </summary>
        public Zeros()
        {
            PyInstance = Instance.keras.initializers.Zeros;
            Init();
        }
    }

    /// <summary>
    /// Initializer that generates tensors initialized to 1.
    /// </summary>
    /// <seealso cref="Keras.Base" />
    public class Ones : Base
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="Ones"/> class.
        /// </summary>
        public Ones()
        {
            PyInstance = Instance.keras.initializers.Zeros;
            Init();
        }
    }

    /// <summary>
    /// Initializer that generates tensors initialized to a constant value.
    /// </summary>
    /// <seealso cref="Keras.Base" />
    public class Constant : Base
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="Constant"/> class.
        /// </summary>
        /// <param name="value">float; the value of the generator tensors.</param>
        public Constant(float value = 0)
        {
            Parameters["value"] = value;
            PyInstance = Instance.keras.initializers.Constant;
            Init();
        }
    }

    /// <summary>
    /// Initializer that generates tensors with a normal distribution.
    /// </summary>
    /// <seealso cref="Keras.Base" />
    public class RandomNormal : Base
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="RandomNormal"/> class.
        /// </summary>
        /// <param name="mean">float. Mean of the random values to generate.</param>
        /// <param name="stddev">float.  Standard deviation of the random values to generate.</param>
        /// <param name="seed">The seed number for random generator</param>
        public RandomNormal(float mean= 0.0f, float stddev= 0.05f, int? seed= null)
        {
            Parameters["mean"] = mean;
            Parameters["stddev"] = stddev;
            Parameters["seed"] = seed;
            PyInstance = Instance.keras.initializers.RandomNormal;
            Init();
        }
    }

    /// <summary>
    /// Initializer that generates tensors with a uniform distribution.
    /// </summary>
    /// <seealso cref="Keras.Base" />
    public class RandomUniform : Base
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="RandomUniform"/> class.
        /// </summary>
        /// <param name="minval">float. Lower bound of the range of random values to generate.</param>
        /// <param name="maxval">float. Upper bound of the range of random values to generate. Defaults to 1 for float types.</param>
        /// <param name="seed">The seed number for random generator</param>
        public RandomUniform(float minval = -0.05f, float maxval = 0.05f, int? seed = null)
        {
            Parameters["minval"] = minval;
            Parameters["maxval"] = maxval;
            Parameters["seed"] = seed;
            PyInstance = Instance.keras.initializers.RandomUniform;
            Init();
        }
    }

    /// <summary>
    /// Initializer that generates a truncated normal distribution.
    /// These values are similar to values from a RandomNormal except that values more than two standard deviations from the mean are discarded and redrawn.This is the recommended initializer for neural network weights and filters.
    /// </summary>
    /// <seealso cref="Keras.Base" />
    public class TruncatedNormal : Base
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="TruncatedNormal"/> class.
        /// </summary>
        /// <param name="mean">float. Mean of the random values to generate.</param>
        /// <param name="stddev">float.  Standard deviation of the random values to generate..</param>
        /// <param name="seed">The seed number for random generator</param>
        public TruncatedNormal(float mean = 0.0f, float stddev = 0.05f, int? seed = null)
        {
            Parameters["mean"] = mean;
            Parameters["stddev"] = stddev;
            Parameters["seed"] = seed;
            PyInstance = Instance.keras.initializers.TruncatedNormal;
            Init();
        }
    }

    /// <summary>
    /// Initializer capable of adapting its scale to the shape of weights.
    /// With distribution = "normal", samples are drawn from a truncated normal distribution centered on zero, with stddev = sqrt(scale / n) where n is:
    /// number of input units in the weight tensor, if mode = "fan_in", 
    /// number of output units, if mode = "fan_out", 
    /// average of the numbers of input and output units, if mode = "fan_avg",
    ///  With distribution = "uniform", samples are drawn from a uniform distribution within[-limit, limit], with limit = sqrt(3 * scale / n).
    /// </summary>
    /// <seealso cref="Keras.Base" />
    public class VarianceScaling : Base
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="VarianceScaling"/> class.
        /// </summary>
        /// <param name="scale">Scaling factor (positive float).</param>
        /// <param name="mode"> One of "fan_in", "fan_out", "fan_avg".</param>
        /// <param name="distribution"> Random distribution to use. One of "normal", "uniform".</param>
        /// <param name="seed">The seed number for random generator</param>
        public VarianceScaling(float scale = 1.0f, string mode = "fan_in", string distribution = "normal", int? seed = null)
        {
            Parameters["scale"] = scale;
            Parameters["mode"] = mode;
            Parameters["distribution"] = distribution;
            Parameters["seed"] = seed;
            PyInstance = Instance.keras.initializers.VarianceScaling;
            Init();
        }
    }

    /// <summary>
    /// Initializer that generates a random orthogonal matrix.
    /// </summary>
    /// <seealso cref="Keras.Base" />
    public class Orthogonal : Base
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="Orthogonal"/> class.
        /// </summary>
        /// <param name="gain">Multiplicative factor to apply to the orthogonal matrix..</param>
        /// <param name="seed">The seed number for random generator</param>
        public Orthogonal(float gain = 1.0f, int? seed = null)
        {
            Parameters["gain"] = gain;
            Parameters["seed"] = seed;
            PyInstance = Instance.keras.initializers.Orthogonal;
            Init();
        }
    }

    /// <summary>
    /// Initializer that generates the identity matrix.
    /// Only use for 2D matrices.If the desired matrix is not square, it pads with zeros on the additional rows/columns
    /// </summary>
    /// <seealso cref="Keras.Base" />
    public class Identity : Base
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="Identity"/> class.
        /// </summary>
        /// <param name="gain">Multiplicative factor to apply to the identity matrix.</param>
        public Identity(float gain = 1.0f)
        {
            Parameters["gain"] = gain;
            PyInstance = Instance.keras.initializers.Identity;
            Init();
        }
    }

    /// <summary>
    /// LeCun uniform initializer.    It draws samples from a uniform distribution within[-limit, limit] where limit is sqrt(3 / fan_in) where fan_in is the number of input units in the weight tensor.
    /// </summary>
    /// <seealso cref="Keras.Base" />
    public class LecunUniform : Base
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="LecunUniform"/> class.
        /// </summary>
        /// <param name="seed">The seed number for random generator</param>
        public LecunUniform(int? seed = null)
        {
            Parameters["seed"] = seed;
            PyInstance = Instance.keras.initializers.lecun_uniform;
            Init();
        }
    }

    /// <summary>
    /// Glorot normal initializer, also called Xavier normal initializer.
    /// It draws samples from a truncated normal distribution centered on 0 with stddev = sqrt(2 / (fan_in + fan_out)) where fan_in is the number of input units in the weight tensor and fan_out is the number of output units in the weight tensor.
    /// </summary>
    /// <seealso cref="Keras.Base" />
    public class GlorotNormal : Base
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="GlorotNormal"/> class.
        /// </summary>
        /// <param name="seed">The seed number for random generator</param>
        public GlorotNormal(int? seed = null)
        {
            Parameters["seed"] = seed;
            PyInstance = Instance.keras.initializers.glorot_normal;
            Init();
        }
    }

    /// <summary>
    /// Glorot uniform initializer, also called Xavier uniform initializer.
    /// It draws samples from a uniform distribution within[-limit, limit] where limit is sqrt(6 / (fan_in + fan_out)) where fan_in is the number of input units in the weight tensor and fan_out is the number of output units in the weight tensor.
    /// </summary>
    /// <seealso cref="Keras.Base" />
    public class GlorotUniform : Base
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="GlorotUniform"/> class.
        /// </summary>
        /// <param name="seed">The seed number for random generator</param>
        public GlorotUniform(int? seed = null)
        {
            Parameters["seed"] = seed;
            PyInstance = Instance.keras.initializers.glorot_uniform;
            Init();
        }
    }

    /// <summary>
    /// He uniform variance scaling initializer.
    /// It draws samples from a uniform distribution within[-limit, limit] where limit is sqrt(6 / fan_in) where fan_in is the number of input units in the weight tensor.
    /// </summary>
    /// <seealso cref="Keras.Base" />
    public class HeUniform : Base
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="HeUniform"/> class.
        /// </summary>
        /// <param name="seed">The seed number for random generator</param>
        public HeUniform(int? seed = null)
        {
            Parameters["seed"] = seed;
            PyInstance = Instance.keras.initializers.he_uniform;
            Init();
        }
    }

    /// <summary>
    /// He normal initializer.
    /// It draws samples from a truncated normal distribution centered on 0 with stddev = sqrt(2 / fan_in) where fan_in is the number of input units in the weight tensor.
    /// </summary>
    /// <seealso cref="Keras.Base" />
    public class HeNormal : Base
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="HeNormal"/> class.
        /// </summary>
        /// <param name="seed">The seed number for random generator</param>
        public HeNormal(int? seed = null)
        {
            Parameters["seed"] = seed;
            PyInstance = Instance.keras.initializers.he_normal;
            Init();
        }
    }

    /// <summary>
    /// LeCun normal initializer.
    /// It draws samples from a truncated normal distribution centered on 0 with stddev = sqrt(1 / fan_in) where fan_in is the number of input units in the weight tensor.
    /// </summary>
    /// <seealso cref="Keras.Base" />
    public class LecunNormal : Base
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="LecunNormal"/> class.
        /// </summary>
        /// <param name="seed">The seed number for random generator</param>
        public LecunNormal(int? seed = null)
        {
            Parameters["seed"] = seed;
            PyInstance = Instance.keras.initializers.lecun_normal;
            Init();
        }
    }
}
