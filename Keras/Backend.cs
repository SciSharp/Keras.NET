using Numpy;
using Python.Runtime;
using System;
using System.Collections.Generic;
using System.Text;

namespace Keras
{
    /// <summary>
    /// Keras is a model-level library, providing high-level building blocks for developing deep learning models. 
    /// It does not handle low-level operations such as tensor products, convolutions and so on itself. Instead, it relies on a specialized, 
    /// well optimized tensor manipulation library to do so, serving as the "backend engine" of Keras. 
    /// Rather than picking one single tensor library and making the implementation of Keras tied to that library, Keras handles the problem in a modular way, 
    /// and several different backend engines can be plugged seamlessly into Keras.
    /// </summary>
    /// <seealso cref="Keras.Base" />
    public class Backend : Base
    {
        static dynamic caller = Instance.keras.backend;

        /// <summary>
        /// Publicly accessible method for determining the current backend.
        /// </summary>
        /// <returns>String, the name of the backend Keras is currently using.</returns>
        public static string GetBackend()
        {
            return caller.backend().ToString();
        }

        /// <summary>
        /// Returns the value of the fuzz factor used in numeric expressions.
        /// </summary>
        /// <returns></returns>
        public static float Epsilon()
        {
            return (float)caller.epsilon();
        }

        /// <summary>
        /// Sets the value of the fuzz factor used in numeric expressions.
        /// </summary>
        /// <param name="e">float. New value of epsilon.</param>
        public void SetEpsilon(float e)
        {
            caller.set_epsilon(e: e);
        }

        /// <summary>
        /// Returns the default float type, as a string. (e.g. 'float16', 'float32', 'float64').
        /// </summary>
        /// <returns>String, the current default float type.</returns>
        public string FloatX()
        {
            return caller.floatx().ToString();
        }

        /// <summary>
        /// Sets the default float type.
        /// </summary>
        /// <param name="floatx">String, 'float16', 'float32', or 'float64'.</param>
        public void SetFloatX(string floatx)
        {
            caller.set_floatx(floatx: floatx);
        }

        /// <summary>
        /// Cast a Numpy array to the default Keras float type.
        /// </summary>
        /// <param name="x">The x, Numpy array..</param>
        /// <returns>The same Numpy array, cast to its new type.</returns>
        public NDarray CastToFloatX(NDarray x)
        {
            return new NDarray((PyObject)caller.cast_to_floatx(x: x));
        }

        /// <summary>
        /// Returns the default image data format convention.
        /// </summary>
        /// <returns>A string, either 'channels_first' or 'channels_last'</returns>
        public static string ImageDataFormat()
        {
            return caller.image_data_format().ToString();
        }

        /// <summary>
        /// Turns off Tensorflow Eager Execution for TF 1.15
        /// </summary>
        /// <returns></returns>
        public static void DisableEager()
        {

            Instance.tensorflow.compat.v1.disable_eager_execution();
        }

        /// <summary>
        /// Sets the image data format.
        /// </summary>
        /// <param name="data_format">data_format: string. 'channels_first' or 'channels_last'.</param>
        public static void SetImageDataFormat(string data_format)
        {
            caller.set_image_data_format(data_format: data_format);
        }

        /// <summary>
        /// Get the uid for the default graph.
        /// </summary>
        /// <param name="prefix">An optional prefix of the graph.</param>
        /// <returns>A unique identifier for the graph.</returns>
        public static string GetUid(string prefix = "")
        {
            return caller.get_uid(prefix: prefix).ToString();
        }

        /// <summary>
        /// Resets graph identifiers.
        /// </summary>
        public static void ResetUids()
        {
            caller.reset_uids();
        }

        /// <summary>
        /// Destroys the current TF graph and creates a new one.Useful to avoid clutter from old models / layers.
        /// </summary>
        public static void ClearSession()
        {
            caller.clear_session();
        }

        /// <summary>
        /// Determines whether the specified tensor is sparse.
        /// </summary>
        /// <param name="tensor">The tensor.</param>
        /// <returns>
        ///   <c>true</c> if the specified tensor is sparse; otherwise, <c>false</c>.
        /// </returns>
        public static bool IsSparse(NDarray tensor)
        {
            return (bool)caller.is_sparse(tensor: tensor);
        }

        /// <summary>
        /// Converts a sparse tensor into a dense tensor and returns it.
        /// </summary>
        /// <param name="tensor">The tensor potentially sparse.</param>
        /// <returns>A dense tensor.</returns>
        public static NDarray ToDense(NDarray tensor)
        {
            return new NDarray(caller.to_dense(tensor: tensor).ToPython());
        }

        public static PyObject Cast(PyObject x, string dtype = "float32")
        {
            return (PyObject)caller.cast(x: x, dtype: dtype);
        }
    }
}
