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
    }
}
