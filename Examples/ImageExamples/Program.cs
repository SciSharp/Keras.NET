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
            var model = new Sequential();
            model.Add(new Conv2D(32, (3, 3).ToTuple(), input_shape: new Shape(64, 64, 3), activation: "relu"));

            model.Add(new MaxPooling2D(pool_size: (2, 2).ToTuple()));

            model.Add(new Flatten());

            model.Add(new Dense(128, activation: "relu"));

            model.Add(new Dense(1, activation: "sigmoid"));

            model.Compile(optimizer: "adam", loss: "binary_crossentropy", metrics: new string[] { "accuracy" });


            var datagen = new ImageDataGenerator(rescale: (float)1.0 / 255, shear_range: (float)0.2, zoom_range: (float)0.2, horizontal_flip: true);

            //var train_datta = datagen.FlowFromDirectory("dataset/", class_mode:"binary", batch_size:64);

            var train_dataset = datagen.FlowFromDirectory(directory: @"C:\Users\blitz\Downloads\kagglecatsanddogs_3367a\PetImages", target_size: (64, 64).ToTuple(), batch_size: 32, class_mode: "binary");


            var test_dataset = datagen.FlowFromDirectory(directory: @"C:\Users\blitz\Downloads\kagglecatsanddogs_3367a\PetImages", target_size: (64, 64).ToTuple(), batch_size: 32, class_mode: "binary");

            model.FitGenerator(train_dataset, steps_per_epoch: 8000, epochs: 25, validation_steps: 800, validation_data: test_dataset);

            //ImageRecognitionApplication.Run();

            //MNIST_CNN.Run();

            Cifar10_CNN.Run();
            var cifar_pred  = Cifar10_CNN.Predict("./img/cifar_test2.jpg");
            Console.WriteLine("cifar_test2.jpg is " + cifar_pred);
            Console.ReadLine();


        }
    }
}
