using Numpy;
using System;
using System.Collections.Generic;
using System.Text;

namespace Keras
{
    public class Losses : Base
    {
        static dynamic caller = Instance.keras.losses;

        public static NDarray MeanSquaredError(NDarray y_true, NDarray y_pred)
        {
            Dictionary<string, object> parameters = new Dictionary<string, object>();
            parameters["y_true"] = y_true;
            parameters["y_pred"] = y_pred;

            return new NDarray(InvokeStaticMethod(caller, "mean_squared_error", parameters));
        }

        public static NDarray MeanAbsoluteError(NDarray y_true, NDarray y_pred)
        {
            Dictionary<string, object> parameters = new Dictionary<string, object>();
            parameters["y_true"] = y_true;
            parameters["y_pred"] = y_pred;

            return new NDarray(InvokeStaticMethod(caller, "mean_absolute_error", parameters));
        }

        public static NDarray MeanAbsolutePercentageError(NDarray y_true, NDarray y_pred)
        {
            Dictionary<string, object> parameters = new Dictionary<string, object>();
            parameters["y_true"] = y_true;
            parameters["y_pred"] = y_pred;
            return new NDarray(InvokeStaticMethod(caller, "mean_absolute_percentage_error", parameters));
        }

        public static NDarray MeanSquaredLogarithmicError(NDarray y_true, NDarray y_pred)
        {
            Dictionary<string, object> parameters = new Dictionary<string, object>();
            parameters["y_true"] = y_true;
            parameters["y_pred"] = y_pred;
            return new NDarray(InvokeStaticMethod(caller, "mean_squared_logarithmic_error", parameters));
        }

        public static NDarray SquaredHinge(NDarray y_true, NDarray y_pred)
        {
            Dictionary<string, object> parameters = new Dictionary<string, object>();
            parameters["y_true"] = y_true;
            parameters["y_pred"] = y_pred;
            return new NDarray(InvokeStaticMethod(caller, "squared_hinge", parameters));
        }

        public static NDarray Hinge(NDarray y_true, NDarray y_pred)
        {
            Dictionary<string, object> parameters = new Dictionary<string, object>();
            parameters["y_true"] = y_true;
            parameters["y_pred"] = y_pred;
            return new NDarray(InvokeStaticMethod(caller, "hinge", parameters));
        }

        public static NDarray CategoricalHinge(NDarray y_true, NDarray y_pred)
        {
            Dictionary<string, object> parameters = new Dictionary<string, object>();
            parameters["y_true"] = y_true;
            parameters["y_pred"] = y_pred;
            return new NDarray(InvokeStaticMethod(caller, "categorical_hinge", parameters));
        }

        public static NDarray LogCosh(NDarray y_true, NDarray y_pred)
        {
            Dictionary<string, object> parameters = new Dictionary<string, object>();
            parameters["y_true"] = y_true;
            parameters["y_pred"] = y_pred;
            return new NDarray(InvokeStaticMethod(caller, "logcosh", parameters));
        }

        public static NDarray CategoricalCrossentropy(NDarray y_true, NDarray y_pred)
        {
            Dictionary<string, object> parameters = new Dictionary<string, object>();
            parameters["y_true"] = y_true;
            parameters["y_pred"] = y_pred;
            return new NDarray(InvokeStaticMethod(caller, "categorical_crossentropy", parameters));
        }

        public static NDarray SparseCategoricalCrossentropy(NDarray y_true, NDarray y_pred)
        {
            Dictionary<string, object> parameters = new Dictionary<string, object>();
            parameters["y_true"] = y_true;
            parameters["y_pred"] = y_pred;
            return new NDarray(InvokeStaticMethod(caller, "sparse_categorical_crossentropy", parameters));
        }

        public static NDarray BinaryCrossentropy(NDarray y_true, NDarray y_pred)
        {
            Dictionary<string, object> parameters = new Dictionary<string, object>();
            parameters["y_true"] = y_true;
            parameters["y_pred"] = y_pred;
            return new NDarray(InvokeStaticMethod(caller, "binary_crossentropy", parameters));
        }

        public static NDarray KullbackLeiblerDivergence(NDarray y_true, NDarray y_pred)
        {
            Dictionary<string, object> parameters = new Dictionary<string, object>();
            parameters["y_true"] = y_true;
            parameters["y_pred"] = y_pred;
            return new NDarray(InvokeStaticMethod(caller, "kullback_leibler_divergence", parameters));
        }

        public static NDarray Poisson(NDarray y_true, NDarray y_pred)
        {
            Dictionary<string, object> parameters = new Dictionary<string, object>();
            parameters["y_true"] = y_true;
            parameters["y_pred"] = y_pred;
            return new NDarray(InvokeStaticMethod(caller, "poisson", parameters));
        }

        public static NDarray CosineProximity(NDarray y_true, NDarray y_pred)
        {
            Dictionary<string, object> parameters = new Dictionary<string, object>();
            parameters["y_true"] = y_true;
            parameters["y_pred"] = y_pred;
            return new NDarray(InvokeStaticMethod(caller, "cosine_proximity", parameters));
        }
    }
}
