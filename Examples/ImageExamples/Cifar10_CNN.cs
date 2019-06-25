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
using Keras.PreProcessing.Image;
using System.IO;

namespace ImageExamples
{
    public class Cifar10_CNN
    {
        private static string[] labels = new string[] { "airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck" };
        public static void Run()
        {
            int batch_size = 128;
            int num_classes = 10;
            int epochs = 100;

            // the data, split between train and test sets
            var ((x_train, y_train), (x_test, y_test)) = Cifar10.LoadData();

            Console.WriteLine("x_train shape: " + x_train.shape);
            Console.WriteLine(x_train.shape[0] + " train samples");
            Console.WriteLine(x_test.shape[0] + " test samples");

            // convert class vectors to binary class matrices
            y_train = Util.ToCategorical(y_train, num_classes);
            y_test = Util.ToCategorical(y_test, num_classes);

            // Build CNN model
            var model = new Sequential();
            model.Add(new Conv2D(32, kernel_size: (3, 3).ToTuple(),
                                 padding: "same",
                                 input_shape: new Shape(32, 32, 3)));
            model.Add(new Activation("relu"));
            model.Add(new Conv2D(32, (3, 3).ToTuple()));
            model.Add(new Activation("relu"));
            model.Add(new MaxPooling2D(pool_size: (2, 2).ToTuple()));
            model.Add(new Dropout(0.25));

            model.Add(new Conv2D(64, kernel_size: (3, 3).ToTuple(),
                                padding: "same"));
            model.Add(new Activation("relu"));
            model.Add(new Conv2D(64, (3, 3).ToTuple()));
            model.Add(new Activation("relu"));
            model.Add(new MaxPooling2D(pool_size: (2, 2).ToTuple()));
            model.Add(new Dropout(0.25));

            model.Add(new Flatten());
            model.Add(new Dense(512));
            model.Add(new Activation("relu"));
            model.Add(new Dropout(0.5));
            model.Add(new Dense(num_classes));
            model.Add(new Activation("softmax"));

            model.Compile(loss: "categorical_crossentropy",
              optimizer: new RMSprop(lr: 0.0001f, decay: 1e-6f), metrics: new string[] { "accuracy" });

            x_train = x_train.astype(np.float32);
            x_test = x_test.astype(np.float32);
            x_train /= 255;
            x_test /= 255;

            model.Fit(x_train, y_train,
                          batch_size: batch_size,
                          epochs: epochs,
                          verbose: 1,
                          validation_data: new NDarray[] { x_test, y_test },
                          shuffle: true);

            //Save model and weights
            //string model_path = "./model.json";
            //string weight_path = "./weights.h5";
            //string json = model.ToJson();
            //File.WriteAllText(model_path, json);
            //model.SaveWeight(weight_path);
            model.Save("model.h5");
            model.SaveTensorflowJSFormat("./");

            //Score trained model.
            var score = model.Evaluate(x_test, y_test, verbose: 0);
            Console.WriteLine("Test loss:" + score[0]);
            Console.WriteLine("Test accuracy:" + score[1]);
        }

        public static string Predict(string imagePath)
        {
            var img = ImageUtil.LoadImg(imagePath, target_size: new Shape(32, 32));
            NDarray x = ImageUtil.ImageToArray(img);
            x = x.reshape(1, x.shape[0], x.shape[1], x.shape[2]);
            var model = Sequential.LoadModel("model.h5");
            model.LoadWeight("weights.h5");
            var y = model.Predict(x);
            y = y.argmax();
            int index = y.asscalar<int>();
            return labels[index];
        }
    }
}
