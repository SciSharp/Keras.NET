using Numpy;
using Python.Runtime;
using System;
using System.Collections.Generic;
using System.Text;

namespace Keras
{
    public partial class Activations : Base
    {
        static dynamic caller = Instance.self.activations;

        public static NDarray Softmax(NDarray x, int axis = -1)
        {
            return new NDarray(InvokeStaticMethod(caller, "softmax", x, axis));
        }

        public static NDarray Relu(NDarray x, float alpha = 0.0f, float? max_value = null, float threshold = 0.0f)
        {
            return new NDarray(InvokeStaticMethod(caller, "relu", x, alpha, max_value, threshold));
        }

        public static NDarray Sigmoid(NDarray x)
        {
            return new NDarray(InvokeStaticMethod(caller, "sigmoid", x));
        }

        public static NDarray Elu(NDarray x, float alpha = 1.0f)
        {
            return new NDarray(InvokeStaticMethod(caller, "elu", x, alpha));
        }

        public static NDarray Selu(NDarray x)
        {
            return new NDarray(InvokeStaticMethod(caller, "selu", x));
        }

        public static NDarray Softplus(NDarray x)
        {
            return new NDarray(InvokeStaticMethod(caller, "softplus", x));
        }

        public static NDarray Softsign(NDarray x)
        {
            return new NDarray(InvokeStaticMethod(caller, "softsign", x));
        }

        public static NDarray Tanh(NDarray x)
        {
            return new NDarray(InvokeStaticMethod(caller, "tanh", x));
        }

        public static NDarray HardSigmoid(NDarray x)
        {
            return new NDarray(InvokeStaticMethod(caller, "hard_sigmoid", x));
        }

        public static NDarray Exponential(NDarray x)
        {
            return new NDarray(InvokeStaticMethod(caller, "exponential", x));
        }

        public static NDarray Linear(NDarray x)
        {
            return new NDarray(InvokeStaticMethod(caller, "linear", x));
        }
    }
}
