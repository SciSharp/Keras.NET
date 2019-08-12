namespace Keras.Constraints
{
    /// <summary>
    /// MaxNorm weight constraint.
    /// Constrains the weights incident to each hidden unit to have a norm less than or equal to a desired value.
    /// </summary>
    public class MaxNorm : Base
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="MaxNorm"/> class.
        /// </summary>
        /// <param name="max_value">the maximum norm for the incoming weights.</param>
        /// <param name="axis"> integer, axis along which to calculate weight norms. For instance, in a Dense layer the weight matrix has shape (input_dim, output_dim), set axis to 0 to constrain each weight vector of length (input_dim,). In a Conv2D layer with data_format="channels_last", the weight tensor has shape  (rows, cols, input_depth, output_depth), set axis to [0, 1, 2] to constrain the weights of each filter tensor of size  (rows, cols, input_depth).</param>
        public MaxNorm(float max_value= 2, int axis= 0)
        {
            Parameters["max_value"] = max_value;
            Parameters["axis"] = axis;

            PyInstance = keras.constraints.MaxNorm;
            Init();
        }
    }

    /// <summary>
    /// Constrains the weights to be non-negative.
    /// </summary>
    /// <seealso cref="Keras.Base" />
    public class NonNeg : Base
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="NonNeg" /> class.
        /// </summary>
        public NonNeg()
        {
            PyInstance = keras.constraints.NonNeg;
            Init();
        }
    }

    /// <summary>
    /// Constrains the weights incident to each hidden unit to have unit norm.
    /// </summary>
    /// <seealso cref="Keras.Base" />
    public class UnitNorm : Base
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="NonNeg" /> class.
        /// </summary>
        /// <param name="axis">integer, axis along which to calculate weight norms. For instance, in a Dense layer the weight matrix has shape (input_dim, output_dim), set axis to 0 to constrain each weight vector of length (input_dim,). In a Conv2D layer with data_format="channels_last", the weight tensor has shape  (rows, cols, input_depth, output_depth), set axis to [0, 1, 2] to constrain the weights of each filter tensor of size  (rows, cols, input_depth).</param>
        public UnitNorm(int axis = 0)
        {
            PyInstance = keras.constraints.NonNeg;
            Init();
        }
    }

    /// <summary>
    /// MinMaxNorm weight constraint.
    /// Constrains the weights incident to each hidden unit to have the norm between a lower bound and an upper bound.
    /// </summary>
    /// <seealso cref="Keras.Base" />
    public class MinMaxNorm : Base
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="MinMaxNorm" /> class.
        /// </summary>
        /// <param name="min_value">the minimum norm for the incoming weights.</param>
        /// <param name="max_value">the maximum norm for the incoming weights.</param>
        /// <param name="rate">rate for enforcing the constraint: weights will be rescaled to yield  (1 - rate) * norm + rate * norm.clip(min_value, max_value). Effectively, this means that rate=1.0 stands for strict enforcement of the constraint, while rate<1.0 means that weights will be rescaled at each step to slowly move towards a value inside the desired interval.</param>
        /// <param name="axis">integer, axis along which to calculate weight norms. For instance, in a Dense layer the weight matrix has shape (input_dim, output_dim), set axis to 0 to constrain each weight vector of length (input_dim,). In a Conv2D layer with data_format="channels_last", the weight tensor has shape  (rows, cols, input_depth, output_depth), set axis to [0, 1, 2] to constrain the weights of each filter tensor of size  (rows, cols, input_depth).</param>
        public MinMaxNorm(float min_value= 0.0f, float max_value= 1.0f, float rate= 1.0f, int axis = 0)
        {
            PyInstance = keras.constraints.NonNeg;
            Init();
        }
    }
}
