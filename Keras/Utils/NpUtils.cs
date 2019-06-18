using Numpy;
using Python.Runtime;
using System;
using System.Collections.Generic;
using System.Text;

namespace Keras
{
    public partial class Utils : Base
    {
        public static NDarray ToCategorical(NDarray y, int? num_classes = null, string dtype = "float32")
        {
            return new NDarray((PyObject)Instance.self.utils.to_categorical(y: y.PyObject, num_classes: num_classes, dtype: dtype));
        }

        public static NDarray Normalize(NDarray y, int axis = -1, int order = 2)
        {
            return new NDarray((PyObject)Instance.self.utils.normalize(y: y.PyObject, axis: axis, order: order));
        }
    }
}
