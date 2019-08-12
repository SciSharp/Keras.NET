using Numpy;
using System;
using System.Collections.Generic;
using System.Text;

namespace Keras.Layers
{
    /// <summary>
    /// Apply additive zero-centered Gaussian noise.
    /// This is useful to mitigate overfitting(you could see it as a form of random data augmentation). 
    /// Gaussian Noise(GS) is a natural choice as corruption process for real valued inputs.
    /// As it is a regularization layer, it is only active at training time.
    /// </summary>
    /// <seealso cref="Keras.Layers.BaseLayer" />
    public class GaussianNoise : BaseLayer
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="GaussianNoise"/> class.
        /// </summary>
        /// <param name="stddev">float, standard deviation of the noise distribution.</param>
        public GaussianNoise(float stddev)
        {
            Parameters["stddev"] = stddev;
            PyInstance = Instance.keras.layers.GaussianNoise;
            Init();
        }
    }

    /// <summary>
    /// Apply multiplicative 1-centered Gaussian noise.
    /// As it is a regularization layer, it is only active at training time.
    /// </summary>
    /// <seealso cref="Keras.Layers.BaseLayer" />
    public class GaussianDropout : BaseLayer
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="GaussianDropout" /> class.
        /// </summary>
        /// <param name="rate"> float, drop probability (as with Dropout). The multiplicative noise will have standard deviation sqrt(rate / (1 - rate)).</param>
        public GaussianDropout(float rate)
        {
            Parameters["rate"] = rate;
            PyInstance = Instance.keras.layers.GaussianDropout;
            Init();
        }
    }

    /// <summary>
    /// Applies Alpha Dropout to the input.
    /// Alpha Dropout is a Dropout that keeps mean and variance of inputs to their original values, in order to ensure the self-normalizing property even after this dropout.
    /// Alpha Dropout fits well to Scaled Exponential Linear Units by randomly setting activations to the negative saturation value.
    /// </summary>
    /// <seealso cref="Keras.Layers.BaseLayer" />
    public class AlphaDropout : BaseLayer
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="AlphaDropout" /> class.
        /// </summary>
        /// <param name="rate">float, drop probability (as with Dropout). The multiplicative noise will have standard deviation sqrt(rate / (1 - rate)).</param>
        /// <param name="noise_shape">A 1-D Tensor of type int32, representing the shape for randomly generated keep/drop flags.</param>
        /// <param name="seed">A Python integer to use as random seed.</param>
        public AlphaDropout(float rate, NDarray<int> noise_shape = null, int? seed = null)
        {
            Parameters["rate"] = rate;
            Parameters["noise_shape"] = noise_shape;
            Parameters["seed"] = seed;
            PyInstance = Instance.keras.layers.AlphaDropout;
            Init();
        }
    }
}
