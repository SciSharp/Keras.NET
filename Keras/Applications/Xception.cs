using Keras.Models;
using Numpy;
using Python.Runtime;
using System;
using System.Collections.Generic;
using System.Text;

namespace Keras.Applications
{
    /// <summary>
    /// Xception V1 model, with weights pre-trained on ImageNet.
    /// On ImageNet, this model gets to a top-1 validation accuracy of 0.790 and a top-5 validation accuracy of 0.945.
    /// Note that this model only supports the data format 'channels_last' (height, width, channels).
    /// The default input size for this model is 299x299.
    /// </summary>
    /// <seealso cref="Keras.Applications.AppModelBase" />
    public class Xception : AppModelBase
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="Xception"/> class.
        /// </summary>
        private Xception() : base((PyObject)Instance.keras.applications.xception)
        {

        }

        /// <summary>
        /// Initializes a new instance of the <see cref="Xception"/> class.
        /// </summary>
        /// <param name="include_top">optional shape tuple, only to be specified if include_top is False (otherwise the input shape has to be (224, 224, 3) (with 'channels_last' data format) or (3, 224, 224) (with 'channels_first' data format) for NASNetMobile or (331, 331, 3) (with 'channels_last' data format) or (3, 331, 331) (with 'channels_first' data format) for NASNetLarge. It should have exactly 3 inputs channels, and width and height should be no smaller than 32. E.g. (200, 200, 3) would be one valid value.</param>
        /// <param name="weights">one of None (random initialization), 'imagenet' (pre-training on ImageNet), or the path to the weights file to be loaded.</param>
        /// <param name="input_tensor">optional Keras tensor (i.e. output of layers.Input()) to use as image input for the model.</param>
        /// <param name="input_shape">optional shape tuple, only to be specified if include_top is False (otherwise the input shape has to be (224, 224, 3) (with 'channels_last' data format) or (3, 224, 224) (with 'channels_first' data format). It should have exactly 3 inputs channels, and width and height should be no smaller than 32. E.g. (200, 200, 3) would be one valid value.</param>
        /// <param name="pooling">optional pooling mode for feature extraction when include_top is False.
        /// None means that the output of the model will be the 4D tensor output of the last convolutional layer.
        /// avg means that global average pooling will be applied to the output of the last convolutional layer, and thus the output of the model will be a 2D tensor.
        /// max means that global max pooling will be applied.</param>
        /// <param name="classes">optional number of classes to classify images into, only to be specified if include_top is True, and if no weights argument is specified.</param>
        public Xception(bool include_top = true, string weights = "imagenet", NDarray input_tensor = null,
                                    Shape input_shape = null, string pooling = "None", int classes = 1000)
            : this()
        {
            Parameters["include_top"] = include_top;
            Parameters["weights"] = weights;
            Parameters["input_tensor"] = input_tensor;
            Parameters["input_shape"] = input_shape;
            Parameters["pooling"] = pooling;
            Parameters["classes"] = classes;

            PyInstance = caller.Xception;
            Init();
        }
    }
}
