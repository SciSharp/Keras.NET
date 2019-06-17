using Numpy;
using Numpy.Models;
using Python.Included;
using Python.Runtime;
using System;
using System.Collections.Generic;
using System.Text;
using static Python.Runtime.Py;

namespace Keras
{
    public class Keras : IDisposable
    {
        private PyObject _pyobj = null;
        private static GILState state = null;
        public static Keras Instance => _instance.Value;

        private static Lazy<Keras> _instance = new Lazy<Keras>(() =>
        {
            var instance = new Keras();
            try
            {
                instance._pyobj = InstallAndImport();
            }
            catch (Exception)
            {
                // retry to fix the installation by forcing a repair.
                instance._pyobj = InstallAndImport(force: true);
            }
            return instance;
        }
        );

        private static PyObject InstallAndImport(bool force = false)
        {
            //var installer = new Installer();
            //installer.SetupPython(force).Wait();
            //Environment.SetEnvironmentVariable("Path", installer.EmbeddedPythonHome);
            //installer.InstallWheel(typeof(Keras).Assembly, "keras").Wait();
            //PythonEngine.Initialize();
            state = Py.GIL();
            var mod = Py.Import("keras");
            return mod;
        }

        public dynamic self => _pyobj;
        private bool IsInitialized => _pyobj != null;

        internal Keras() { }

        public void Dispose()
        {
            self?.Dispose();
            state.Dispose();
        }

        internal static PyObject ToPython(object obj)
        {
            if (obj == null) return Runtime.GetPyNone();
            switch (obj)
            {
                // basic types
                case int o: return new PyInt(o);
                case float o: return new PyFloat(o);
                case double o: return new PyFloat(o);
                case string o: return new PyString(o);
                case bool o:
                    if (o)
                        return new PyObject(Runtime.PyTrue);
                    else
                        return new PyObject(Runtime.PyFalse);

                // sequence types
                case Array o: return ToTuple(o);
                case List<string> o: return ToTuple(o.ToArray());
                // special types from 'ToPythonConversions'
                case Shape o: return ToTuple(o.Dimensions);
                case Slice o: return o.ToPython();
                case PythonObject o: return o.PyObject;
                case Base o: return o.ToPython();
                default: throw new NotImplementedException($"Type is not yet supported: { obj.GetType().Name}. Add it to 'ToPythonConversions'");
            }
        }

        protected static PyTuple ToTuple(Array input)
        {
            var array = new PyObject[input.Length];
            for (int i = 0; i < input.Length; i++)
            {
                array[i] = ToPython(input.GetValue(i));
            }

            return new PyTuple(array);
        }
    }
}
