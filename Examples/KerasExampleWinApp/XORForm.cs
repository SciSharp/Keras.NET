using Keras;
using Keras.Layers;
using Keras.Optimizers;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using Numpy;
using K = Keras.Backend;
using Keras.Utils;
using Keras.Models;
using Keras.Datasets;
using System.IO;

namespace KerasExampleWinApp
{
    public partial class XORForm : Form
    {
        public XORForm()
        {
            InitializeComponent();
            Keras.Keras.DisablePySysConsoleLog = true;
        }

        private void btnTrain_Click(object sender, EventArgs e)
        {
            worker.RunWorkerAsync();
            worker.RunWorkerCompleted += Woker_RunWorkerCompleted;
        }

        private void Woker_RunWorkerCompleted(object sender, RunWorkerCompletedEventArgs e)
        {
            txtTrainingResult.Text = Keras.Keras.GetStdOut();
        }

        private void woker_DoWork(object sender, DoWorkEventArgs e)
        {
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
            //var weights = model.GetWeights();
            //model.SetWeights(weights);
            var logs = history.HistoryLogs;
            //Save model and weights
            string json = model.ToJson();
            File.WriteAllText("model.json", json);
            model.SaveWeight("model.h5");
        }
    }
}
