using Python.Runtime;
using System;
using System.Collections.Generic;
using System.Text;

namespace Keras
{
    public class StringOrInstance
    {
        public PyObject PyObject;

        public StringOrInstance(PyObject obj)
        {
            PyObject = obj;
        }

        public static implicit operator StringOrInstance(string opt)
        {
            return new StringOrInstance(opt.ToPython());
        }

        public static implicit operator StringOrInstance(Base opt)
        {
            return new StringOrInstance(opt.ToPython());
        }
    }
}
