using Keras.Layers;
using Keras.Utils;
using Numpy;
using Numpy.Models;
using Python.Runtime;
using System;
using System.Collections.Generic;
using System.Text;
using static Python.Runtime.Py;

namespace Keras
{
    public class Keras : IDisposable
    {
        public static Keras Instance => _instance.Value;

        private static Lazy<Keras> _instance = new Lazy<Keras>(() =>
        {
            var instance = new Keras();
            instance.keras = InstallAndImport("keras");
            try
            {
                instance.keras2onnx = InstallAndImport("onnxmltools");
            }
            catch(Exception ex)
            {
                Console.WriteLine("Warning! onnxmltools is not installed to export to onnx");
            }

            try
            {
                instance.tfjs = InstallAndImport("tensorflowjs");
            }
            catch (Exception ex)
            {
                Console.WriteLine("Warning! tensorflowjs is not installed. Required to convert keras model to tensorflow js format");
            }
            
            return instance;
        }
        );

        private static PyObject InstallAndImport(string module)
        {
            if(!PythonEngine.IsInitialized)
                PythonEngine.Initialize();
            var mod = Py.Import(module);
            return mod;
        }

        public dynamic keras = null;

        public dynamic keras2onnx = null;

        public dynamic tfjs = null;

        private bool IsInitialized => keras != null;

        internal Keras() { }

        public void Dispose()
        {
            keras?.Dispose();
            PythonEngine.Shutdown();
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
                case Array o: return ToList(o);
                // special types from 'ToPythonConversions'
                case Shape o: return ToTuple(o.Dimensions);
                case Slice o: return o.ToPython();
                case PythonObject o: return o.PyObject;
                case PyObject o: return o;
                case Sequence o: return o.ToPython();
                case StringOrInstance o: return o.PyObject;
                case KerasFunction o: return o.PyObject;
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

        protected static PyList ToList(Array input)
        {
            var array = new PyObject[input.Length];
            for (int i = 0; i < input.Length; i++)
            {
                array[i] = ToPython(input.GetValue(i));
            }

            return new PyList(array);
        }

    }
}
