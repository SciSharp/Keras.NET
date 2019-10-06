using Keras;
using Keras.Layers;
using Keras.Models;
using Keras.PreProcessing.Image;
using System;

namespace ImageExamples
{
    class Program
    {
        static void Main(string[] args)
        {
            //ImageRecognitionApplication.Run();

            MNIST_CNN.Run();

            Cifar10_CNN.Run();
            var cifar_pred  = Cifar10_CNN.Predict("./img/cifar_test2.jpg");
            Console.WriteLine("cifar_test2.jpg is " + cifar_pred);
            Console.ReadLine();


        }
    }
}
