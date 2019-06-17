using Python.Runtime;
using System;
using System.Collections.Generic;
using System.Text;
using System.Linq;

namespace Keras
{
    public abstract class Base : Keras
    {
        internal dynamic __self__;
        public Dictionary<string, object> Parameters = new Dictionary<string, object>();

        public virtual PyObject GetPythonObject()
        {
            var pyargs = ToTuple(new object[]
            {
                Parameters.FirstOrDefault().Value
            });

            var kwargs = new PyDict();

            bool skip = true;
            foreach (var item in Parameters)
            {
                if (skip)
                {
                    skip = false;
                    continue;
                }

                if (item.Value != null && !string.IsNullOrWhiteSpace(item.Value.ToString()))
                {
                    kwargs[item.Key] = ToPython(item.Value);
                }
            }

            if (Parameters.Count > 0)
                return __self__.Invoke(pyargs, kwargs);
            else
                return __self__.Invoke(null, null);
        }

        public static PyObject InvokeStatic(dynamic caller, string method, params object[] parameters)
        {
            var pyargs = ToTuple(parameters);
            var kwargs = new PyDict();
            if (parameters.Length > 0)
                return caller.InvokeMethod(method, pyargs, kwargs);
            else
                return caller.InvokeMethod(method, null, null);
        }

        public object this[string name]
        {
            get
            {
                return Parameters[name];
            }
            set
            {
                Parameters[name] = value;
            }
        }
    }
}
