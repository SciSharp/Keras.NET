using Numpy;
using System;
using System.Collections.Generic;
using System.Text;

namespace Keras
{
    public class Metrics : Base
    {
        static dynamic caller = Instance.self.metrics;

        public static NDarray MSE(NDarray y_true, NDarray y_pred)
        {
            return new NDarray(InvokeStaticMethod(caller, "mse", y_true, y_pred));
        }

        public static NDarray MAE(NDarray y_true, NDarray y_pred)
        {
            return new NDarray(InvokeStaticMethod(caller, "mae", y_true, y_pred));
        }

        public static NDarray MAPE(NDarray y_true, NDarray y_pred)
        {
            return new NDarray(InvokeStaticMethod(caller, "mape", y_true, y_pred));
        }

        public static NDarray MSLE(NDarray y_true, NDarray y_pred)
        {
            return new NDarray(InvokeStaticMethod(caller, "msle", y_true, y_pred));
        }

        public static NDarray Cosine(NDarray y_true, NDarray y_pred)
        {
            return new NDarray(InvokeStaticMethod(caller, "cosine", y_true, y_pred));
        }

        public static NDarray BinaryAccuracy(NDarray y_true, NDarray y_pred)
        {
            return new NDarray(InvokeStaticMethod(caller, "binary_accuracy", y_true, y_pred));
        }

        public static NDarray CategoricalAccuracy(NDarray y_true, NDarray y_pred)
        {
            return new NDarray(InvokeStaticMethod(caller, "categorical_accuracy", y_true, y_pred));
        }

        public static NDarray SparseCategoricalAccuracy(NDarray y_true, NDarray y_pred)
        {
            return new NDarray(InvokeStaticMethod(caller, "sparse_categorical_accuracy", y_true, y_pred));
        }

        public static NDarray TopKCategoricalAccuracy(NDarray y_true, NDarray y_pred, int k = 5)
        {
            return new NDarray(InvokeStaticMethod(caller, "top_k_categorical_accuracy", y_true, y_pred, k));
        }

        public static NDarray SparseTopKCategoricalAccuracy(NDarray y_true, NDarray y_pred, int k = 5)
        {
            return new NDarray(InvokeStaticMethod(caller, "sparse_top_k_categorical_accuracy", y_true, y_pred, k));
        }
    }
}
