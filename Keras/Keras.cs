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
            instance.keras = InstallAndImport(Setup.KerasModule);

            try
            {
                instance.tensorflow = InstallAndImport("tensorflow");
            }
            catch (Exception ex)
            {
                Console.WriteLine("Warning! tensorflow is not installed. Required to load models");
            }

            try
            {
                instance.keras2onnx = InstallAndImport("onnxmltools");
            }
            catch (Exception ex)
            {
            }

            try
            {
                instance.tfjs = InstallAndImport("tensorflowjs");
            }
            catch (Exception ex)
            {
            }

            return instance;
        }
        );

        private static PyObject InstallAndImport(string module)
        {
            Console.WriteLine(module);
            if(!PythonEngine.IsInitialized)
                PythonEngine.Initialize();
            var mod = Py.Import(module);
            return mod;
        }

        public dynamic keras = null;

        public dynamic tensorflow = null;

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
                case ValueTuple<int> o: return ToTuple(o);
                case ValueTuple<int, int> o: return ToTuple(o);
                case ValueTuple<int, int, int> o: return ToTuple(o);
                case Slice o: return o.ToPython();
                case PythonObject o: return o.PyObject;
                case PyObject o: return o;
                case Sequence o: return o.PyInstance;
                case StringOrInstance o: return o.PyObject;
                case KerasFunction o: return o.PyObject;
                case Base o: return o.PyInstance;
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

        protected static PyTuple ToTuple(ValueTuple<int> input)
        {
            var array = new PyObject[1];
            array[0] = ToPython(input.Item1);

            return new PyTuple(array);
        }

        protected static PyTuple ToTuple(ValueTuple<int, int> input)
        {
            var array = new PyObject[2];
            array[0] = ToPython(input.Item1);
            array[1] = ToPython(input.Item2);

            return new PyTuple(array);
        }

        protected static PyTuple ToTuple(ValueTuple<int, int, int> input)
        {
            var array = new PyObject[3];
            array[0] = ToPython(input.Item1);
            array[1] = ToPython(input.Item2);
            array[2] = ToPython(input.Item3);

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