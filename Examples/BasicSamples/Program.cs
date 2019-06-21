using System;

namespace BasicSamples
{
    class Program
    {
        static void Main(string[] args)
        {
            //XOR.Run();

            //PrimaIndiansDiabetics.Run();
            //var input1 = new Keras.Layers.Input(shape: (16));
            //var d1 = new Keras.Layers.Dense(8);
            //d1.Set(input1);
            //var input2 = new Keras.Layers.Input(shape: (32));
            //var d2 = new Keras.Layers.Dense(8);
            //d2.Set(input2);
            //d2.ToPython();

            //var added = new Keras.Layers.Add(d1, d2);
            //added.Set(d1, d2);
            //added.ToPython();

            MNIST_CNN.Run();

            Console.ReadLine();
        }
    }
}
