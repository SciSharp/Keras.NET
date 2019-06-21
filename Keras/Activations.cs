using Numpy;
using Python.Runtime;
using System;
using System.Collections.Generic;
using System.Text;

namespace Keras
{
    /// <summary>
    /// Activations can either be used through an Activation layer, or through the activation argument supported by all forward layers:
    /// </summary>
    /// <seealso cref="Keras.Base" />
    public partial class Activations : Base
    {
        static dynamic caller = Instance.keras.activations;

        /// <summary>
        /// Softmax activation function
        /// </summary>
        /// <param name="x">The input tensor.</param>
        /// <param name="axis">Integer, axis along which the softmax normalization is applied.</param>
        /// <returns>Tensor, output of softmax transformation.</returns>
        public static NDarray Softmax(NDarray x, int axis = -1)
        {
            Dictionary<string, object> parameters = new Dictionary<string, object>();
            parameters["x"] = x;
            parameters["axis"] = axis;
            return new NDarray(InvokeStaticMethod(caller, "softmax", parameters));
        }

        /// <summary>
        /// Rectified Linear Unit. With default values, it returns element-wise max(x, 0). Otherwise, it follows: f(x) = max_value for x >= max_value, f(x) = x for threshold <= x<max_value, f(x) = alpha* (x - threshold) otherwise.
        /// </summary>
        /// <param name="x"> Input tensor.</param>
        /// <param name="alpha">float. Slope of the negative part. Defaults to zero.</param>
        /// <param name="max_value">float. Saturation threshold.</param>
        /// <param name="threshold">float. Threshold value for thresholded activation.</param>
        /// <returns></returns>
        public static NDarray Relu(NDarray x, float alpha = 0.0f, float? max_value = null, float threshold = 0.0f)
        {
            Dictionary<string, object> parameters = new Dictionary<string, object>();
            parameters["x"] = x;
            parameters["alpha"] = alpha;
            parameters["max_value"] = max_value;
            parameters["threshold"] = threshold;

            return new NDarray(InvokeStaticMethod(caller, "relu", parameters));
        }

        /// <summary>
        /// Sigmoid activation function.
        /// </summary>
        /// <param name="x">Input tensor.</param>
        /// <returns>Tensor, output of sigmoid</returns>
        public static NDarray Sigmoid(NDarray x)
        {
            Dictionary<string, object> parameters = new Dictionary<string, object>();
            parameters["x"] = x;
            return new NDarray(InvokeStaticMethod(caller, "sigmoid", parameters));
        }

        /// <summary>
        /// Exponential linear unit. The exponential linear activation: x if x > 0 and alpha * (exp(x)-1) if x < 0.
        /// </summary>
        /// <param name="x">Input tensor.</param>
        /// <param name="alpha">A scalar, slope of negative section.</param>
        /// <returns>The exponential unit activation: elu(x, alpha).</returns>
        public static NDarray Elu(NDarray x, float alpha = 1.0f)
        {
            Dictionary<string, object> parameters = new Dictionary<string, object>();
            parameters["x"] = x;
            parameters["alpha"] = alpha;
            return new NDarray(InvokeStaticMethod(caller, "elu", parameters));
        }

        /// <summary>
        /// Scaled Exponential Linear Unit (SELU).
        /// SELU is equal to: scale* elu(x, alpha), where alpha and scale are predefined constants.The values of alpha and scale are chosen so that the mean and variance of the inputs are preserved between two consecutive layers as long as the weights are initialized correctly(see lecun_normal initialization) and the number of inputs is "large enough" (see references for more information).
        /// </summary>
        /// <param name="x">Input tensor.</param>
        /// <returns>The scaled exponential unit activation: scale * elu(x, alpha).</returns>
        public static NDarray Selu(NDarray x)
        {
            Dictionary<string, object> parameters = new Dictionary<string, object>();
            parameters["x"] = x;
            return new NDarray(InvokeStaticMethod(caller, "selu", parameters));
        }

        /// <summary>
        /// Softplus activation function. The softplus activation: log(exp(x) + 1).
        /// </summary>
        /// <param name="x">Input tensor.</param>
        /// <returns>Output tensor</returns>
        public static NDarray Softplus(NDarray x)
        {
            Dictionary<string, object> parameters = new Dictionary<string, object>();
            parameters["x"] = x;
            return new NDarray(InvokeStaticMethod(caller, "softplus", parameters));
        }

        /// <summary>
        /// The softsign activation: x / (abs(x) + 1).
        /// </summary>
        /// <param name="x">Input tensor.</param>
        /// <returns>Output tensor</returns>
        public static NDarray Softsign(NDarray x)
        {
            Dictionary<string, object> parameters = new Dictionary<string, object>();
            parameters["x"] = x;
            return new NDarray(InvokeStaticMethod(caller, "softsign", parameters));
        }

        /// <summary>
        /// Hyperbolic tangent activation function.
        /// </summary>
        /// <param name="x">Input tensor.</param>
        /// <returns>Output tensor</returns>
        public static NDarray Tanh(NDarray x)
        {
            Dictionary<string, object> parameters = new Dictionary<string, object>();
            parameters["x"] = x;
            return new NDarray(InvokeStaticMethod(caller, "tanh", parameters));
        }

        /// <summary>
        /// Hard sigmoid activation function.        Faster to compute than sigmoid activation.
        /// </summary>
        /// <param name="x">Input tensor.</param>
        /// <returns>Output tensor</returns>
        public static NDarray HardSigmoid(NDarray x)
        {
            Dictionary<string, object> parameters = new Dictionary<string, object>();
            parameters["x"] = x;
            return new NDarray(InvokeStaticMethod(caller, "hard_sigmoid", parameters));
        }

        /// <summary>
        /// Exponential (base e) activation function.
        /// </summary>
        /// <param name="x">Input tensor.</param>
        /// <returns>Output tensor</returns>
        public static NDarray Exponential(NDarray x)
        {
            Dictionary<string, object> parameters = new Dictionary<string, object>();
            parameters["x"] = x;
            return new NDarray(InvokeStaticMethod(caller, "exponential", parameters));
        }

        /// <summary>
        /// Linear (i.e. identity) activation function.
        /// </summary>
        /// <param name="x">Input tensor.</param>
        /// <returns>Output tensor</returns>
        public static NDarray Linear(NDarray x)
        {
            Dictionary<string, object> parameters = new Dictionary<string, object>();
            parameters["x"] = x;
            return new NDarray(InvokeStaticMethod(caller, "linear", parameters));
        }
    }
}
