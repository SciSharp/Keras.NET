using Numpy;
using Python.Runtime;
using System;
using System.Collections.Generic;
using System.Text;

namespace Keras.Helper
{
    public class TupleSolver
    {
        public static T[] TupleToList<T>(PyObject obj)
        {
            List<T> result = new List<T>();
            GetTTupleList(new PyIter(obj), ref result);

            return result.ToArray();
        }

        private static void GetTTupleList<T>(PyObject obj, ref List<T> result)
        {
            PyIter iter = new PyIter(obj);

            while (iter.MoveNext())
            {
                var r = iter.Current.ToPython();
                if (PyTuple.IsTupleType(r))
                {
                    GetTTupleList<T>(r, ref result);
                    continue;
                }

                switch (typeof(T).Name)
                {
                    case "Single":
                    case "Double":
                    case "Int32":
                    case "Int64":
                    case "UInt32":
                    case "UInt64":
                    case "Byte":
                    case "Object":
                    case "String":
                    case "SByte":
                        result.Add(r.As<T>());
                        break;
                    default:
                        break;
                }
            }
        }

        public static NDarray[] TupleToList(PyObject obj)
        {
            PyIter iter = new PyIter(obj);
            List<NDarray> result = new List<NDarray>();
            GetNdListFromTuple(new PyIter(obj), ref result);

            return result.ToArray();
        }

        private static void GetNdListFromTuple(PyObject obj, ref List<NDarray> result)
        {
            PyIter iter = new PyIter(obj);
            
            while (iter.MoveNext())
            {
                var r = iter.Current.ToPython();

                if (PyTuple.IsTupleType(r))
                {
                    GetNdListFromTuple(r, ref result);
                    continue;
                }

                result.Add(new NDarray(r));
            }
        }
    }
}
