using Keras.Applications;
using Keras.PreProcessing.Image;
using System;
using System.Collections.Generic;
using System.Text;
using Numpy;
using Keras.Applications.ResNet;

namespace ImageExamples
{
    public class ImageRecognitionApplication
    {
        public static void Run()
        {
            var model = new Xception(weights: "imagenet");
            string img_path = "./img/elephant.jpg";
            var img = ImageUtil.LoadImg(img_path, target_size: (224, 224));
            var x = ImageUtil.ImageToArray(img);
            x = np.expand_dims(x, axis: 0);
            x = model.PreprocessInput(x);
            var preds = model.Predict(x);

            var predictions = model.DecodePredictions(preds, top: 2);
            foreach (ImageNetPrediction item in predictions)
            {
                Console.WriteLine("Name: {0}, Acc: {1}", item.Word, item.PredictedValue);
            }
        }
    }
}
