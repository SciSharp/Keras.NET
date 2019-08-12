using Numpy;
using System;
using System.Collections.Generic;
using System.Text;

namespace Keras.Layers
{
    /// <summary>
    /// This wrapper applies a layer to every temporal slice of an input.
    /// The input should be at least 3D, and the dimension of index one will be considered to be the temporal dimension.
    /// Consider a batch of 32 samples, where each sample is a sequence of 10 vectors of 16 dimensions.
    /// The batch input shape of the layer is then (32, 10, 16), and the input_shape, not including the samples dimension, is (10, 16).
    /// You can then use TimeDistributed to apply a Dense layer to each of the 10 timesteps, independently:
    /// </summary>
    /// <seealso cref="Keras.Layers.BaseLayer" />
    public class TimeDistributed : BaseLayer
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="TimeDistributed"/> class.
        /// </summary>
        /// <param name="layer">The layer instance.</param>
        public TimeDistributed(BaseLayer layer)
        {
            Parameters["layer"] = layer.PyInstance;

            PyInstance = Instance.keras.layers.TimeDistributed;
            Init();
        }
    }

    /// <summary>
    /// Bidirectional wrapper for RNNs.
    /// </summary>
    /// <seealso cref="Keras.Layers.BaseLayer" />
    public class Bidirectional : BaseLayer
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="Bidirectional" /> class.
        /// </summary>
        /// <param name="layer">The recurrent layer instance.</param>
        /// <param name="merge_mode">Mode by which outputs of the forward and backward RNNs will be combined. One of {'sum', 'mul', 'concat', 'ave', None}. If None, the outputs will not be combined, they will be returned as a list.</param>
        /// <param name="weights">Initial weights to load in the Bidirectional model.</param>
        public Bidirectional(BaseLayer layer, string merge_mode= "concat", NDarray weights= null)
        {
            Parameters["layer"] = layer.PyInstance;

            PyInstance = Instance.keras.layers.Bidirectional;
            Init();
        }
    }
}
