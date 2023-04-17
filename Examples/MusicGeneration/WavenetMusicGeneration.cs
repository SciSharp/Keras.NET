using Keras.Callbacks;
using Keras.Layers;
using Keras.Models;
using Keras.Optimizers;
using Melanchall.DryWetMidi.Core;
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
        static int no_of_timesteps = 32; //No. of notes to teach per step
        static NDarray train_x = null; // Training notes set
        static NDarray train_y = null; // Next note to play 
        static int output = 127; //Max number of notes a piano have

        public static void PrepData()
        {
            //notes array list declaration
            List<Note[]> notes_array = new List<Note[]>();

            //Get all the file list
            var files = Directory.GetFiles("./PianoDataset");

            //Loop through the files and read the notes, put them in the array
            foreach (var file in files)
            {
                notes_array.Add(ReadMidi(file));
            }

            //Show first 15 notes for the first music file
            Console.WriteLine("\n\nDisplaying first 15 notes of first music\n");
            foreach (var n in notes_array[0].Take(15))
            {
                Console.Write(n.ToString() + " ");
            }

            //Declare X and Y which will hold the training set
            List<float> x = new List<float>();
            List<float> y = new List<float>();
           
            //Loop through the notes and prepare X and Y set.
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

            // Finally convert them to numpy array format for neural network training.
            train_x = np.array(x.ToArray(), dtype: np.float32).reshape(-1, 32);
            train_y = np.array(y.ToArray(), dtype: np.float32);
        }

        private static Note[] ReadMidi(string file)
        {
            Console.WriteLine("Loading music file: " + file);
            var mf = MidiFile.Read(file);
            return mf.GetNotes().ToArray();
        }

        public static void BuildAndTrain()
        {
            //Model to hold the neural network architecture which in this case is WaveNet
            var model = new Sequential();
            // Starts with embedding layer
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

            // Compile with Adam optimizer
            model.Compile(loss: "sparse_categorical_crossentropy", optimizer: new Adam());
            model.Summary();

            // Callback to store the best trained model
            var mc = new ModelCheckpoint("best_model.h5", monitor: "val_loss", mode: "min", save_best_only: true, verbose: 1);

            //Method to actually train the model for 100 iteration
            var history = model.Fit(train_x, train_y, batch_size: 32, epochs: 100, validation_split: 0.25f, verbose: 1, callbacks: new Callback[] { mc });

            // Save the final trained model which we are going to use for prediction
            model.Save("last_epoch.h5");
        }

        public static List<int> GenerateNewMusic(int n = 20)
        {
            //Load the trained model
            var model = Model.LoadModel("last_epoch.h5");
            //Get a random 32 notes from the train set which we will use to get new notes
            var ind = np.random.randint(0, train_x.shape[0]);
            var random_music  = train_x[ind];

            //Build the prediction variable with sample 32 notes
            List<float> predictions = new List<float>();
            predictions.AddRange(random_music.GetData<float>());

            //Loop through N times which means N new notes, by default its 20
            for (int i = 0; i < n; i++)
            {
                // Reshape to model adaptaed shape
                random_music = random_music.reshape(1, no_of_timesteps);
                //Predict the next best note to be played
                var prob = model.Predict(random_music)[0];
                var y_pred = np.argmax(prob, axis: 0);

                //Add the prediction and pick the last 32 to predict the next music note
                predictions.Add(y_pred.asscalar<float>());
                random_music = np.array(predictions.Skip(i + 1).ToArray());
            }

            //Finally skip the first 32 sample notes and return the rest N new predicted notes.
            return predictions.Skip(no_of_timesteps - 1).Select(x=>(int)x).ToList();
        }

        
    }
}
