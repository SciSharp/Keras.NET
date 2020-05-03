using Keras.Callbacks;
using Keras.Layers;
using Keras.Models;
using Keras.Optimizers;
using Melanchall.DryWetMidi.Core;
using Melanchall.DryWetMidi.Devices;
using Melanchall.DryWetMidi.Interaction;
using Numpy;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace MusicGeneration
{
    public class WavenetMusicGeneration
    {
        static int no_of_timesteps = 32;
        static NDarray train_x = null;
        static NDarray train_y = null;
        static int output = 0;

        public static void PrepData()
        {
            List<Note[]> notes_array = new List<Note[]>();
            var files = Directory.GetFiles("./PianoDataset");
            foreach (var file in files)
            {
                notes_array.Add(ReadMidi(file));
            }

            List<float> x = new List<float>();
            List<float> y = new List<float>();
           
            foreach (var notes in notes_array)
            {
                for (int i = 0; i < notes.Length - no_of_timesteps; i++)
                {
                    var input = notes.Skip(i).Take(no_of_timesteps).Select(x => Convert.ToSingle(x.NoteNumber)).ToArray();
                    var output = Convert.ToSingle(notes[i + no_of_timesteps].NoteNumber);
                    x.AddRange(input);
                    y.Add(output);
                }
            }

            var unique_set = x.ToHashSet();

            output = (int)unique_set.Max();

            train_x = np.array(x.ToArray(), dtype: np.float32).reshape(-1, 32);
            train_y = np.array(y.ToArray(), dtype: np.float32);
        }

        public static void Train()
        {
            var model = new Sequential();
            // embedding layer
            model.Add(new Embedding(output, 100, input_length: 32));

            model.Add(new Conv1D(64, 3, padding: "causal", activation: "tanh"));
            model.Add(new Dropout(0.2));
            model.Add(new MaxPooling1D(2));


            model.Add(new Conv1D(128, 3, activation: "relu", dilation_rate: 2, padding: "causal"));
            model.Add(new Dropout(0.2));
            model.Add(new MaxPooling1D(2));

            model.Add(new Conv1D(256, 3, activation: "relu", dilation_rate: 4, padding: "causal"));
            model.Add(new Dropout(0.2));
            model.Add(new MaxPooling1D(2));

            //model.Add(new Conv1D(256, 5, activation: "relu"));
            model.Add(new GlobalMaxPooling1D());

            model.Add(new Dense(256, activation: "relu"));
            model.Add(new Dense(output, activation: "softmax"));

            model.Compile(loss: "sparse_categorical_crossentropy", optimizer: new Adam());
            model.Summary();

            var mc = new ModelCheckpoint("best_model.h5", monitor: "val_loss", mode: "min", save_best_only: true, verbose: 1);
            var history = model.Fit(train_x, train_y, batch_size: 32, epochs: 100, validation_split: 0.25f, verbose: 1, callbacks: new Callback[] { mc });

            model.Save("last_epoch.h5");
        }

        public static List<int> GenerateNewMusic(int n = 20)
        {
            var model = Model.LoadModel("last_epoch.h5");
            var ind = np.random.randint(0, train_x.shape[0]);
            var random_music  = train_x[ind];
            List<float> predictions = new List<float>();
            predictions.AddRange(random_music.GetData<float>());
            for (int i = 0; i < n; i++)
            {
                random_music = random_music.reshape(1, no_of_timesteps);
                var prob = model.Predict(random_music)[0];
                var y_pred = np.argmax(prob, axis: 0);
                predictions.Add(y_pred.asscalar<float>());
                random_music = np.array(predictions.Skip(i + 1).ToArray());
            }

            return predictions.Skip(no_of_timesteps - 1).Select(x=>(int)x).ToList();
        }

        private static Note[] ReadMidi(string file)
        {
            Console.WriteLine("Loading music file: " + file);
            var mf = MidiFile.Read(file);
            return mf.GetNotes().ToArray();
        }
    }
}
