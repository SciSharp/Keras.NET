using Python.Runtime;
using System;
using System.Collections.Generic;
using System.Text;

namespace Keras
{
    /// <summary>
    /// String or instance of a class
    /// </summary>
    public class StringOrInstance
    {
        /// <summary>
        /// The py object which is the base variable
        /// </summary>
        public PyObject PyObject;

        /// <summary>
        /// Initializes a new instance of the <see cref="StringOrInstance"/> class.
        /// </summary>
        /// <param name="obj">The object.</param>
        public StringOrInstance(PyObject obj)
        {
            PyObject = obj;
        }

        /// <summary>
        /// Performs an implicit conversion from <see cref="System.String"/> to <see cref="StringOrInstance"/>.
        /// </summary>
        /// <param name="opt">The opt.</param>
        /// <returns>
        /// The result of the conversion.
        /// </returns>
        public static implicit operator StringOrInstance(string opt)
        {
            return new StringOrInstance(opt.ToPython());
        }

        /// <summary>
        /// Performs an implicit conversion from <see cref="Base"/> to <see cref="StringOrInstance"/>.
        /// </summary>
        /// <param name="opt">The opt.</param>
        /// <returns>
        /// The result of the conversion.
        /// </returns>
        public static implicit operator StringOrInstance(Base opt)
        {
            return new StringOrInstance(opt.PyInstance);
        }
    }

    public class KerasFunction
    {
        /// <summary>
        /// The py object which is the base variable
        /// </summary>
        public PyObject PyObject;

        /// <summary>
        /// Initializes a new instance of the <see cref="StringOrInstance"/> class.
        /// </summary>
        /// <param name="obj">The object.</param>
        public KerasFunction(PyObject obj)
        {
            PyObject = obj;
        }
    }

    public class KerasIterator : object
    {
        /// <summary>
        /// The py object which is the base variable
        /// </summary>
        public PyObject PyObject;

        /// <summary>
        /// Initializes a new instance of the <see cref="StringOrInstance"/> class.
        /// </summary>
        /// <param name="obj">The object.</param>
        public KerasIterator(PyObject obj)
        {
            PyObject = obj;
        }
    }

    public class DirectoryIterator : object
    {
        /// <summary>
        /// The py object which is the base variable
        /// </summary>
        public PyObject PyObject;

        /// <summary>
        /// Initializes a new instance of the <see cref="StringOrInstance"/> class.
        /// </summary>
        /// <param name="obj">The object.</param>
        public DirectoryIterator(PyObject obj)
        {
            PyObject = obj;
        }
    }
}
