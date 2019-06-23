using Keras.Models;
using Numpy;
using Python.Runtime;
using System;
using System.Collections.Generic;
using System.Text;

namespace Keras.Applications
{
    /// <summary>
    /// MobileNet model, with weights pre-trained on ImageNet.
    /// Note that this model only supports the data format 'channels_last' (height, width, channels).
    /// The default input size for this model is 224x224.
    /// </summary>
    /// <seealso cref="Keras.Base" />
    public class MobileNet : Base
    {
        private static dynamic caller = Instance.keras.applications;

        /// <summary>
        /// Gets the model.
        /// </summary>
        /// <param name="input_shape">optional shape tuple, only to be specified if include_top is False (otherwise the input shape has to be (224, 224, 3) (with 'channels_last' data format) or (3, 224, 224) (with 'channels_first' data format). It should have exactly 3 inputs channels, and width and height should be no smaller than 32. E.g. (200, 200, 3) would be one valid value.</param>
        /// <param name="alpha">controls the width of the network. If alpha &lt; 1.0, proportionally decreases the number of filters in each layer. If alpha &gt; 1.0, proportionally increases the number of filters in each layer.</param>
        /// <param name="depth_multiplier">depth multiplier for depthwise convolution (also called the resolution multiplier)</param>
        /// <param name="dropout">The dropout rate.</param>
        /// <param name="include_top">optional shape tuple, only to be specified if include_top is False (otherwise the input shape has to be (224, 224, 3) (with 'channels_last' data format) or (3, 224, 224) (with 'channels_first' data format) for NASNetMobile or (331, 331, 3) (with 'channels_last' data format) or (3, 331, 331) (with 'channels_first' data format) for NASNetLarge. It should have exactly 3 inputs channels, and width and height should be no smaller than 32. E.g. (200, 200, 3) would be one valid value.</param>
        /// <param name="weights">one of None (random initialization), 'imagenet' (pre-training on ImageNet), or the path to the weights file to be loaded.</param>
        /// <param name="input_tensor">optional Keras tensor (i.e. output of layers.Input()) to use as image input for the model.</param>
        /// <param name="pooling">optional pooling mode for feature extraction when include_top is False.
        /// None means that the output of the model will be the 4D tensor output of the last convolutional layer.
        /// avg means that global average pooling will be applied to the output of the last convolutional layer, and thus the output of the model will be a 2D tensor.
        /// max means that global max pooling will be applied.</param>
        /// <param name="classes">optional number of classes to classify images into, only to be specified if include_top is True, and if no weights argument is specified.</param>
        /// <returns>
        /// A Keras model instance.
        /// </returns>
        public static Model GetModel(Shape input_shape = null, float alpha = 1, int depth_multiplier=1, float dropout = 1e-3f, bool include_top = true, string weights = "imagenet", 
                                    NDarray input_tensor = null, string pooling = "None", int classes = 1000)
        {
            Dictionary<string, object> parameters = new Dictionary<string, object>();
            parameters["input_shape"] = input_shape;
            parameters["alpha"] = alpha;
            parameters["depth_multiplier"] = depth_multiplier;
            parameters["dropout"] = dropout;
            parameters["include_top"] = include_top;
            parameters["weights"] = weights;
            parameters["input_tensor"] = input_tensor;
            parameters["pooling"] = pooling;
            parameters["classes"] = classes;

            return new Model(InvokeStaticMethod(caller, "MobileNet", parameters));
        }
    }
}
