using Numpy;
using Python.Runtime;
using System;
using System.Collections.Generic;
using System.Text;

namespace Keras
{
    public partial class Activations : Base
    {
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
    }
}
