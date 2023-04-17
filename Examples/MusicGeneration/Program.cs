using Melanchall.DryWetMidi.Common;
using Melanchall.DryWetMidi.Core;
using Melanchall.DryWetMidi.Interaction;
using Melanchall.DryWetMidi.Multimedia;
using System;
using System.Collections.Generic;
using System.Linq;

namespace MusicGeneration
{
    class Program
    {
        static void Main(string[] args)
        {
            WavenetMusicGeneration.PrepData();
            //WavenetMusicGeneration.BuildAndTrain();
            var notes = WavenetMusicGeneration.GenerateNewMusic(20);
            Console.WriteLine("\n\nPlaying auto generated music....\n");
            PlayNotes(notes);
        }

        private static void PlayNotes(List<int> notes)
        {
            List<List<MidiEvent>> musicNotes = new List<List<MidiEvent>>();
            var playbackDevice = OutputDevice.GetAll().FirstOrDefault();
            foreach (var note in notes)
            {
                Note n = new Note(SevenBitNumber.Parse(note.ToString()));
                Console.Write(n + " ");
                //playbackDevice.SendEvent(new NoteOnEvent(SevenBitNumber.Parse(note.ToString()), SevenBitNumber.MaxValue));
            }
        }
    }
}
