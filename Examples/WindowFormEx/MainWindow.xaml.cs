using Keras;
using Keras.Layers;
using Keras.Models;
using Keras.Optimizers;
using Numpy;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;
namespace WindowFormEx
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        public MainWindow()
        {
            InitializeComponent();
            //Setup.SetPythonPath(@"C:\Python36");
            Keras.Keras.DisablePySysConsoleLog = true;
        }

        private void Button_Click(object sender, RoutedEventArgs e)
        {
            Console.WriteLine("test");
            NDarray x = np.array(new float[,] { { 0, 0 }, { 0, 1 }, { 1, 0 }, { 1, 1 } });
            NDarray y = np.array(new float[] { 0, 1, 1, 0 });

            //Build functional model
            var input = new Input(shape: new Keras.Shape(2));
            var hidden1 = new Dense(32, activation: "relu").Set(input);
            var hidden2 = new Dense(64, activation: "relu").Set(hidden1);
            var output = new Dense(1, activation: "sigmoid").Set(hidden2);
            var model = new Keras.Models.Model(new Input[] { input }, new BaseLayer[] { output });
            
            //Compile and train
            model.Compile(optimizer: new Adam(), loss: "binary_crossentropy", metrics: new string[] { "accuracy" });

            
            var history = model.Fit(x, y, batch_size: 2, epochs: 10, verbose: 1);
            string stdout = Keras.Keras.GetStdOut();
            //var weights = model.GetWeights();
            //model.SetWeights(weights);
            var logs = history.HistoryLogs;
            //Save model and weights
            string json = model.ToJson();
            File.WriteAllText("model.json", json);
            model.SaveWeight("model.h5");
            //Load model and weight
            var loaded_model = Sequential.ModelFromJson(File.ReadAllText("model.json"));
            loaded_model.LoadWeight("model.h5");

            
        }
    }
}
