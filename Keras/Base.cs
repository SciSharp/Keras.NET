using Python.Runtime;
using System;
using System.Collections.Generic;
using System.Text;
using System.Linq;

namespace Keras
{
    public abstract class Base : Keras
    {
        internal dynamic PyInstance;
        public Dictionary<string, object> Parameters = new Dictionary<string, object>();

        public object None = null;

        public void Init()
        {
            PyInstance = Instantiate();
        }

        public virtual PyObject Instantiate()
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
                return PyInstance.Invoke(pyargs, kwargs);
            else
                return PyInstance.Invoke(null, null);
        }

        public virtual PyObject ToPython()
        {
            return PyInstance;
        }

        //public static PyObject InvokeStaticMethod(dynamic caller, string method, params object[] parameters)
        //{
        //    var pyargs = ToTuple(parameters);
        //    var kwargs = new PyDict();
        //    if (parameters.Length > 0)
        //        return caller.InvokeMethod(method, pyargs, kwargs);
        //    else
        //        return caller.InvokeMethod(method, null, null);
        //}

        public static PyObject InvokeStaticMethod(dynamic caller, string method, Dictionary<string, object> args)
        {
            var pyargs = ToTuple(new object[]
            {
                args.FirstOrDefault().Value
            });

            var kwargs = new PyDict();

            bool skip = true;
            foreach (var item in args)
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

            if (args.Count > 0)
                return caller.InvokeMethod(method, pyargs, kwargs);
            else
                return caller.InvokeMethod(method, null, null);
        }

        public PyObject InvokeMethod(string method, Dictionary<string, object> args)
        {
           var pyargs = ToTuple(new object[]
           {
                args.FirstOrDefault().Value
           });

            var kwargs = new PyDict();

            bool skip = true;
            foreach (var item in args)
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

            if (args.Count > 0)
                return PyInstance.InvokeMethod(method, pyargs, kwargs);
            else
                return PyInstance.InvokeMethod(method, null, null);
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
