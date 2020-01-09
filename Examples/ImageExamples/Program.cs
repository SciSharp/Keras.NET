using Keras;
using Keras.Layers;
using Keras.Models;
using Keras.PreProcessing.Image;
using System;
using System.Collections.Generic;
using K = Keras.Backend;

namespace ImageExamples
{
    class Program
    {
        static void Main(string[] args)
        {
            //Disable Eager Execution for tensorflow 1.15
            K.DisableEager();


            //ImageRecognitionApplication.Run();
            
            MNIST_CNN.Run();

            Cifar10_CNN.Run();
            var cifar_pred  = Cifar10_CNN.Predict("..\\..\\..\\img\\cifar_test2.jpg");
            Console.WriteLine("cifar_test2.jpg is " + cifar_pred);
            Console.ReadLine();


        }
    }
}
