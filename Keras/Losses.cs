using Numpy;
using System;
using System.Collections.Generic;
using System.Text;

namespace Keras
{
    public class Losses : Base
    {
        static dynamic caller = Instance.self.losses;

        public static NDarray MeanSquaredError(NDarray y_true, NDarray y_pred)
        {
            return new NDarray(InvokeStaticMethod(caller, "mean_squared_error", y_true, y_pred));
        }

        public static NDarray MeanAbsoluteError(NDarray y_true, NDarray y_pred)
        {
            return new NDarray(InvokeStaticMethod(caller, "mean_absolute_error", y_true, y_pred));
        }

        public static NDarray MeanAbsolutePercentageError(NDarray y_true, NDarray y_pred)
        {
            return new NDarray(InvokeStaticMethod(caller, "mean_absolute_percentage_error", y_true, y_pred));
        }

        public static NDarray MeanSquaredLogarithmicError(NDarray y_true, NDarray y_pred)
        {
            return new NDarray(InvokeStaticMethod(caller, "mean_squared_logarithmic_error", y_true, y_pred));
        }

        public static NDarray SquaredHinge(NDarray y_true, NDarray y_pred)
        {
            return new NDarray(InvokeStaticMethod(caller, "squared_hinge", y_true, y_pred));
        }

        public static NDarray Hinge(NDarray y_true, NDarray y_pred)
        {
            return new NDarray(InvokeStaticMethod(caller, "hinge", y_true, y_pred));
        }

        public static NDarray CategoricalHinge(NDarray y_true, NDarray y_pred)
        {
            return new NDarray(InvokeStaticMethod(caller, "categorical_hinge", y_true, y_pred));
        }

        public static NDarray LogCosh(NDarray y_true, NDarray y_pred)
        {
            return new NDarray(InvokeStaticMethod(caller, "logcosh", y_true, y_pred));
        }

        public static NDarray CategoricalCrossentropy(NDarray y_true, NDarray y_pred)
        {
            return new NDarray(InvokeStaticMethod(caller, "categorical_crossentropy", y_true, y_pred));
        }

        public static NDarray SparseCategoricalCrossentropy(NDarray y_true, NDarray y_pred)
        {
            return new NDarray(InvokeStaticMethod(caller, "sparse_categorical_crossentropy", y_true, y_pred));
        }

        public static NDarray BinaryCrossentropy(NDarray y_true, NDarray y_pred)
        {
            return new NDarray(InvokeStaticMethod(caller, "binary_crossentropy", y_true, y_pred));
        }

        public static NDarray KullbackLeiblerDivergence(NDarray y_true, NDarray y_pred)
        {
            return new NDarray(InvokeStaticMethod(caller, "kullback_leibler_divergence", y_true, y_pred));
        }

        public static NDarray Poisson(NDarray y_true, NDarray y_pred)
        {
            return new NDarray(InvokeStaticMethod(caller, "poisson", y_true, y_pred));
        }

        public static NDarray CosineProximity(NDarray y_true, NDarray y_pred)
        {
            return new NDarray(InvokeStaticMethod(caller, "cosine_proximity", y_true, y_pred));
        }
    }
}
