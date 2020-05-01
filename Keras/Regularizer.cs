using Numpy;
using Python.Runtime;
using System;
using System.Collections.Generic;
using System.Text;

namespace Keras.Regularizers
{
    /// <summary>
    /// Regularizers allow to apply penalties on layer parameters or layer activity during optimization. These penalties are incorporated in the loss function that the network optimizes.
    /// </summary>
    /// <seealso cref="Keras.Base" />
    public class L1L2 : Base
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="L1L2"/> class.
        /// </summary>
        /// <param name="l1">The l1.</param>
        /// <param name="l2">The l2.</param>
        public L1L2(float l1 = 0.01f, float l2 = 0.01f)
        {
            Parameters["l1"] = l1;
            Parameters["l2"] = l2;
            PyInstance = Instance.keras.regularizers.L1L2;
            Init();
        }
    }

    /// <summary>
    /// Regularizers allow to apply penalties on layer parameters or layer activity during optimization. These penalties are incorporated in the loss function that the network optimizes.
    /// </summary>
    /// <seealso cref="Keras.Base" />
    public class L1 : Base
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="L1"/> class.
        /// </summary>
        /// <param name="l1">The l1.</param>
        public L1(float l1 = 0.01f)
        {
            Parameters["l1"] = l1;
            PyInstance = Instance.keras.regularizers.l1;
            Init();
        }
    }

    /// <summary>
    /// Regularizers allow to apply penalties on layer parameters or layer activity during optimization. These penalties are incorporated in the loss function that the network optimizes.
    /// </summary>
    /// <seealso cref="Keras.Base" />
    public class L2 : Base
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="L2" /> class.
        /// </summary>
        /// <param name="l2">The l2.</param>
        public L2(float l2 = 0.01f)
        {
            Parameters["l2"] = l2;
            PyInstance = Instance.keras.regularizers.l2;
            Init();
        }
    }
}
