using Keras.Datasets;
using System;
using System.Collections.Generic;
using System.Text;
using Numpy;
using K = Keras.Backend;
using Keras;
using Keras.Models;
using Keras.Layers;
using Keras.Utils;
using Keras.Optimizers;
using Keras.Optimizers.Legacy;

namespace BasicSamples
{
    public class MNIST_CNN
    {
        public static void Run()
        {
            int batch_size = 128; //Training batch size
            int num_classes = 10; //No. of classes
            int epochs = 12; //No. of epoches we will train

            // input image dimensions
            int img_rows = 28, img_cols = 28;

            // Declare the input shape for the network
            Shape input_shape = null;

            // Load the MNIST dataset into Numpy array
            var ((x_train, y_train), (x_test, y_test)) = MNIST.LoadData();

            //Check if its channel fist or last and rearrange the dataset accordingly
            if(K.ImageDataFormat() == "channels_first")
            {
                x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols);
                x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols);
                input_shape = (1, img_rows, img_cols);
            }
            else
            {
                x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1);
                x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1);
                input_shape = (img_rows, img_cols, 1);
            }

            //Normalize the input data
            x_train = x_train.astype(np.float32);
            x_test = x_test.astype(np.float32);
            x_train /= 255;
            x_test /= 255;
            Console.WriteLine("x_train shape: " + x_train.shape);
            Console.WriteLine(x_train.shape[0] + " train samples");
            Console.WriteLine(x_test.shape[0] + " test samples");

            // Convert class vectors to binary class matrices
            y_train = Util.ToCategorical(y_train, num_classes);
            y_test = Util.ToCategorical(y_test, num_classes);

            // Build CNN model
            var model = new Sequential();
            model.Add(new Conv2D(32, kernel_size: (3, 3).ToTuple(),
                                 activation: "relu",
                                 input_shape: input_shape));
            model.Add(new Conv2D(64, (3, 3).ToTuple(), activation: "relu"));
            model.Add(new MaxPooling2D(pool_size: (2, 2).ToTuple()));
            model.Add(new Dropout(0.25));
            model.Add(new Flatten());
            model.Add(new Dense(128, activation: "relu"));
            model.Add(new Dropout(0.5));
            model.Add(new Dense(num_classes, activation: "softmax"));

            //Compile with loss, metrics and optimizer
            model.Compile(loss: "categorical_crossentropy",
              optimizer: new Adadelta(), metrics: new string[] { "accuracy" });

            //Train the model
            model.Fit(x_train, y_train,
                      batch_size: batch_size,
                      epochs: epochs,
                      verbose: 1,
                      validation_data: new NDarray[] { x_test, y_test });


            //Score the model for performance
            var score = model.Evaluate(x_test, y_test, verbose: 0);
            Console.WriteLine("Test loss:" + score[0]);
            Console.WriteLine("Test accuracy:" + score[1]);

            // Save the model to HDF5 format which can be loaded later or ported to other application
            model.Save("model.h5");
            // Save it to Tensorflow JS format and we will test it in browser.
            model.SaveTensorflowJSFormat("./");
        }
    }
}
