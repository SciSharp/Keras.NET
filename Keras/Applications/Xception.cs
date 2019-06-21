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
    /// <seealso cref="Keras.Base" />
    public class Xception : Base
    {
        private static dynamic caller = Instance.keras.applications.xception;

        /// <summary>
        /// Gets the model.
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
        /// <returns>A Keras model instance.</returns>
        public static Model GetModel(bool include_top = true, string weights = "imagenet", NDarray input_tensor = null,
                                    Shape input_shape = null, string pooling = "None", int classes = 1000)
        {
            Dictionary<string, object> parameters = new Dictionary<string, object>();
            parameters["include_top"] = include_top;
            parameters["weights"] = weights;
            parameters["input_tensor"] = input_tensor;
            parameters["input_shape"] = input_shape;
            parameters["pooling"] = pooling;
            parameters["classes"] = classes;

            return new Model(InvokeStaticMethod(caller, "Xception", parameters));
        }

        /// <summary>
        /// Decodes the predictions.
        /// </summary>
        /// <param name="preds">The preds.</param>
        /// <param name="top">The top.</param>
        /// <returns></returns>
        public ImageNetPrediction[] DecodePredictions(NDarray preds, int top = 3)
        {
            Dictionary<string, object> parameters = new Dictionary<string, object>();
            parameters["preds"] = preds;
            parameters["top"] = top;
            return ((PyObject)InvokeStaticMethod(caller, "decode_predictions", parameters)).As<ImageNetPrediction[]>();
        }

        /// <summary>
        /// Preprocesses the input.
        /// </summary>
        /// <param name="x">The input tensor.</param>
        /// <returns></returns>
        public NDarray PreprocessInput(NDarray x)
        {
            Dictionary<string, object> parameters = new Dictionary<string, object>();
            parameters["x"] = x;
            return new NDarray((PyObject)InvokeStaticMethod(caller, "preprocess_input", parameters));
        }
    }
}
