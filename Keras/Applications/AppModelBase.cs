using Keras.Helper;
using Keras.Models;
using Numpy;
using Python.Runtime;
using System;
using System.Collections.Generic;
using System.Dynamic;
using System.Globalization;
using System.Text;

namespace Keras.Applications
{
    /// <summary>
    /// Model base class for keras applications
    /// </summary>
    /// <seealso cref="Keras.Models.Model" />
    public class AppModelBase : Model
    {
        internal dynamic caller;

        /// <summary>
        /// Initializes a new instance of the <see cref="AppModelBase"/> class.
        /// </summary>
        /// <param name="_caller">The caller.</param>
        public AppModelBase(dynamic _caller)
        {
            caller = _caller;
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
            var predobj = (PyObject)InvokeStaticMethod(caller, "decode_predictions", parameters);
            var d = predobj.ToString();
            var list = TupleSolver.TupleToList<object>(predobj[0]);
            List<ImageNetPrediction> predictions = new List<ImageNetPrediction>();
            for (int i = 0; i < list.Length; i = i++)
            {
                ImageNetPrediction pred = new ImageNetPrediction()
                {
                    WordID = list[i].ToString(),
                    Word = list[i + 1].ToString(),
                    PredictedValue = Convert.ToSingle(list[i + 2].ToString(),CultureInfo.InvariantCulture),
                };

                i = i + 3;

                predictions.Add(pred);
            }

            return predictions.ToArray();
        }

        /// <summary>
        /// Preprocesses the input.
        /// </summary>
        /// <param name="x">The input tensor.</param>
        /// <returns></returns>
        public NDarray PreprocessInput(NDarray x, string data_format = "channels_last")
        {
            Dictionary<string, object> parameters = new Dictionary<string, object>();
            parameters["x"] = x;
            parameters["data_format"] = data_format;
            //Parameters["mode"] = mode;
            return new NDarray((PyObject)InvokeStaticMethod(caller, "preprocess_input", parameters));
        }
    }
}
