using Numpy;
using Python.Runtime;
using System;
using System.Collections.Generic;
using System.Text;

namespace Keras
{
    public partial class Activations : Base
    {
        static dynamic caller = Instance.keras.activations;

        public static NDarray Softmax(NDarray x, int axis = -1)
        {
            Dictionary<string, object> parameters = new Dictionary<string, object>();
            parameters["x"] = x;
            parameters["axis"] = axis;
            return new NDarray(InvokeStaticMethod(caller, "softmax", parameters));
        }

        public static NDarray Relu(NDarray x, float alpha = 0.0f, float? max_value = null, float threshold = 0.0f)
        {
            Dictionary<string, object> parameters = new Dictionary<string, object>();
            parameters["x"] = x;
            parameters["alpha"] = alpha;
            parameters["max_value"] = max_value;
            parameters["threshold"] = threshold;

            return new NDarray(InvokeStaticMethod(caller, "relu", parameters));
        }

        public static NDarray Sigmoid(NDarray x)
        {
            Dictionary<string, object> parameters = new Dictionary<string, object>();
            parameters["x"] = x;
            return new NDarray(InvokeStaticMethod(caller, "sigmoid", parameters));
        }

        public static NDarray Elu(NDarray x, float alpha = 1.0f)
        {
            Dictionary<string, object> parameters = new Dictionary<string, object>();
            parameters["x"] = x;
            parameters["alpha"] = alpha;
            return new NDarray(InvokeStaticMethod(caller, "elu", parameters));
        }

        public static NDarray Selu(NDarray x)
        {
            Dictionary<string, object> parameters = new Dictionary<string, object>();
            parameters["x"] = x;
            return new NDarray(InvokeStaticMethod(caller, "selu", parameters));
        }

        public static NDarray Softplus(NDarray x)
        {
            Dictionary<string, object> parameters = new Dictionary<string, object>();
            parameters["x"] = x;
            return new NDarray(InvokeStaticMethod(caller, "softplus", parameters));
        }

        public static NDarray Softsign(NDarray x)
        {
            Dictionary<string, object> parameters = new Dictionary<string, object>();
            parameters["x"] = x;
            return new NDarray(InvokeStaticMethod(caller, "softsign", parameters));
        }

        public static NDarray Tanh(NDarray x)
        {
            Dictionary<string, object> parameters = new Dictionary<string, object>();
            parameters["x"] = x;
            return new NDarray(InvokeStaticMethod(caller, "tanh", parameters));
        }

        public static NDarray HardSigmoid(NDarray x)
        {
            Dictionary<string, object> parameters = new Dictionary<string, object>();
            parameters["x"] = x;
            return new NDarray(InvokeStaticMethod(caller, "hard_sigmoid", parameters));
        }

        public static NDarray Exponential(NDarray x)
        {
            Dictionary<string, object> parameters = new Dictionary<string, object>();
            parameters["x"] = x;
            return new NDarray(InvokeStaticMethod(caller, "exponential", parameters));
        }

        public static NDarray Linear(NDarray x)
        {
            Dictionary<string, object> parameters = new Dictionary<string, object>();
            parameters["x"] = x;
            return new NDarray(InvokeStaticMethod(caller, "linear", parameters));
        }
    }
}
