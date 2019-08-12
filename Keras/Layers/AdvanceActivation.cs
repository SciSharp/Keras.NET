using System;
using System.Collections.Generic;
using System.Text;

namespace Keras.Layers
{
    /// <summary>
    /// Leaky version of a Rectified Linear Unit.
    /// It allows a small gradient when the unit is not active: f(x) = alpha* x for x< 0, f(x) = x for x >= 0.
    /// </summary>
    /// <seealso cref="Keras.Layers.BaseLayer" />
    public class LeakyReLU : BaseLayer
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="LeakyReLU"/> class.
        /// </summary>
        /// <param name="alpha">float >= 0. Negative slope coefficient.</param>
        public LeakyReLU(float alpha = 0.3f)
        {
            Parameters["alpha"] = alpha;

            PyInstance = Instance.keras.layers.LeakyReLU;
            Init();
        }
    }

    /// <summary>
    /// Parametric Rectified Linear Unit.
    /// It follows: f(x) = alpha* x for x< 0, f(x) = x for x >= 0, where alpha is a learned array with the same shape as x.
    /// </summary>
    /// <seealso cref="Keras.Layers.BaseLayer" />
    public class PReLU : BaseLayer
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="PReLU" /> class.
        /// </summary>
        /// <param name="alpha_initializer">initializer function for the weights.</param>
        /// <param name="alpha_regularizer">regularizer for the weights.</param>
        /// <param name="alpha_constraint">constraint for the weights.</param>
        /// <param name="shared_axes">the axes along which to share learnable parameters for the activation function. For example, if the incoming feature maps are from a 2D convolution with output shape (batch, height, width, channels), and you wish to share parameters across space so that each filter only has one set of parameters, set shared_axes=[1, 2].</param>
        public PReLU(string alpha_initializer= "zeros", string alpha_regularizer= "", string alpha_constraint= "", int[] shared_axes= null)
        {
            Parameters["alpha_initializer"] = alpha_initializer;
            Parameters["alpha_regularizer"] = alpha_regularizer;
            Parameters["alpha_constraint"] = alpha_constraint;
            Parameters["shared_axes"] = shared_axes;

            PyInstance = Instance.keras.layers.PReLU;
            Init();
        }
    }

    /// <summary>
    /// Exponential Linear Unit.
    /// It follows: f(x) =  alpha* (exp(x) - 1.) for x< 0, f(x) = x for x >= 0.
    /// </summary>
    /// <seealso cref="Keras.Layers.BaseLayer" />
    public class ELU : BaseLayer
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="ELU"/> class.
        /// </summary>
        /// <param name="alpha"> scale for the negative factor.</param>
        public ELU(float alpha = 1)
        {
            Parameters["alpha"] = alpha;

            PyInstance = Instance.keras.layers.LeakyReLU;
            Init();
        }
    }

    /// <summary>
    /// Thresholded Rectified Linear Unit.
    /// It follows: f(x) = x for x > theta, f(x) = 0 otherwise.
    /// </summary>
    /// <seealso cref="Keras.Layers.BaseLayer" />
    public class ThresholdedReLU : BaseLayer
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="ThresholdedReLU" /> class.
        /// </summary>
        /// <param name="theta">float >= 0. Threshold location of activation.</param>
        public ThresholdedReLU(float theta = 1)
        {
            Parameters["theta"] = theta;

            PyInstance = Instance.keras.layers.ThresholdedReLU;
            Init();
        }
    }

    /// <summary>
    /// Softmax activation function.
    /// </summary>
    /// <seealso cref="Keras.Layers.BaseLayer" />
    public class Softmax : BaseLayer
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="Softmax" /> class.
        /// </summary>
        /// <param name="axis"> Integer, axis along which the softmax normalization is applied.</param>
        public Softmax(int axis = -1)
        {
            Parameters["axis"] = axis;

            PyInstance = Instance.keras.layers.Softmax;
            Init();
        }
    }

    /// <summary>
    /// Rectified Linear Unit activation function.
    /// With default values, it returns element-wise max(x, 0). 
    /// Otherwise, it follows: f(x) = max_value for x >= max_value, f(x) = x for threshold &lt;= x<max_value, f(x) = negative_slope* (x - threshold) otherwise.
    /// </summary>
    /// <seealso cref="Keras.Layers.BaseLayer" />
    public class ReLU : BaseLayer
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="ReLU" /> class.
        /// </summary>
        /// <param name="max_value">float >= 0. Maximum activation value.</param>
        /// <param name="negative_slope">float >= 0. Negative slope coefficient.</param>
        /// <param name="threshold">float. Threshold value for thresholded activation.</param>
        public ReLU(float? max_value= null, float negative_slope= 0, float threshold= 0)
        {
            Parameters["max_value"] = max_value;
            Parameters["negative_slope"] = negative_slope;
            Parameters["threshold"] = threshold;

            PyInstance = Instance.keras.layers.ReLU;
            Init();
        }
    }
}
