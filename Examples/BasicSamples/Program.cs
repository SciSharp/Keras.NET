using System;

namespace BasicSamples
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("Running XOR");
            XOR.Run();

            Console.WriteLine("Running PrimaIndiansDiabetics");
            PrimaIndiansDiabetics.Run();

            Console.ReadLine();
        }
    }
}
