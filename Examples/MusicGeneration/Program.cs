using Melanchall.DryWetMidi.Common;
using Melanchall.DryWetMidi.Core;
using Melanchall.DryWetMidi.Devices;
using Melanchall.DryWetMidi.Interaction;
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
            //WavenetMusicGeneration.Train();
            var notes = WavenetMusicGeneration.GenerateNewMusic(100);
            PlayNotes(notes);
        }

        private static void PlayNotes(List<int> notes)
        {
            List<List<MidiEvent>> musicNotes = new List<List<MidiEvent>>();
            foreach (var note in notes)
            {
                OutputDevice.GetAll().FirstOrDefault().SendEvent(new NoteOffEvent(SevenBitNumber.Parse(note.ToString()), SevenBitNumber.MaxValue));
                //List<MidiEvent> events = new List<MidiEvent>();
                //events.Add(new NoteOnEvent(SevenBitNumber.Parse(note.ToString()), SevenBitNumber.MaxValue));
                //Playback playback = new Playback(events, TempoMap.Default);
                //playback.OutputDevice = OutputDevice.GetAll().FirstOrDefault();
                //playback.Play();
                //playback.Stop();
            }

            //var midiFile = new MidiFile();
            //var tempoMap = midiFile.GetTempoMap();

            //var trackChunk = new TrackChunk();
            //using (var notesManager = trackChunk.ManageNotes())
            //{
            //    var length = LengthConverter.ConvertFrom(2 * MusicalTimeSpan.Eighth.Triplet(),
            //                                             0,
            //                                             tempoMap);
            //    foreach (var note in notes)
            //    {
            //        notesManager.Notes.Add(new Note(SevenBitNumber.Parse(note.ToString())));
            //    }
            //}

            //midiFile.Chunks.Add(trackChunk);
            //midiFile.Write("Single note great song.mid");
            //midiFile.Play();
            


        }
    }
}
