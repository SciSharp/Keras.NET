using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Keras
{
    public class Shape
    {
        public int[] Dimensions { get; }

        public Shape(params int[] shape)
        {
            this.Dimensions = shape;
        }

        public int this[int n] => Dimensions[n];

        #region Equality

        public static bool operator ==(Shape a, Shape b)
        {
            if (b is null) return false;
            return Enumerable.SequenceEqual(a.Dimensions, b?.Dimensions);
        }

        public static bool operator !=(Shape a, Shape b)
        {
            return !(a == b);
        }

        public override bool Equals(object obj)
        {
            if (obj.GetType() != typeof(Shape))
                return false;
            return Enumerable.SequenceEqual(Dimensions, ((Shape)obj).Dimensions);
        }

        public override int GetHashCode()
        {
            return (Dimensions ?? new int[0]).GetHashCode();
        }

        public override string ToString()
        {
            return $"({string.Join(", ", Dimensions ?? new int[0])})";
        }

        public static implicit operator Shape(ValueTuple<int> shape)
        {
            return new Shape(shape.Item1);
        }

        public static implicit operator Shape(ValueTuple<int, int> shape)
        {
            return new Shape(shape.Item1, shape.Item2);
        }

        public static implicit operator Shape(ValueTuple<int, int, int> shape)
        {
            return new Shape(shape.Item1, shape.Item2, shape.Item3);
        }

        public static implicit operator Shape(ValueTuple<int, int, int, int> shape)
        {
            return new Shape(shape.Item1, shape.Item2, shape.Item3, shape.Item4);
        }

        public static implicit operator Shape(ValueTuple<int, int, int, int, int> shape)
        {
            return new Shape(shape.Item1, shape.Item2, shape.Item3, shape.Item4, shape.Item5);
        }

        #endregion
    }
}
