using Keras;
using Keras.Models;
using Keras.PreProcessing.Image;
using System;

namespace BasicSamples
{
    class Program
    {
        static void Main(string[] args)
        {
            //Run to setup keras and backend.
            //Setup.Run(SetupBackend.TensorflowGPU);

            //MergeExample.Run();
            ImplementCallback.Run();

            Console.WriteLine("Running XOR");
            //XOR.Run();

            Console.WriteLine("Running PrimaIndiansDiabetics");
            //PrimaIndiansDiabetics.Run();

            Console.ReadLine();
        }
    }
}
