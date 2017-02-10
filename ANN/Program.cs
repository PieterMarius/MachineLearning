using System;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using MachineLearning.Helpers;

namespace MachineLearning
{
    class Program
    {
        static void Main(string[] args)
        {

            //TestExp();

            int T = 35000;
            int nFeature = 784;
            int nLabel = 10;
            int nLayer = 3;
            double momentum = 0.0;
            IFunction type = new Sigmoid(0.666666, 0.0);
            //IFunction type = new Tanh(0.666666, 1.7159, 0.0);

            double[] eta = new double[] { 3.5, 3.5, 3.5 };
            int[] nodeLayer = new int[] { nFeature, 100, nLabel };
            double[][] dataMatrix = null;
            double[][] labelMatrix = null;

            readDataset("data.dat", new[] { "  " }, ref dataMatrix);
            readDataset("tlabel.dat", new[] { " " },ref labelMatrix);

            double[][] testDataMatrix = null;
            double[][] testLabelMatrix = null;

            readDataset("testmnist.dat", new[] { "  " }, ref testDataMatrix);
            readDataset("tlabeltest.dat", new[] { " " }, ref testLabelMatrix);

            ANN network = new ANN(T, nLayer, nodeLayer, eta, momentum, type);
            network.DimBatch = 10;
            network.DimBatchMax = 20;
            network.DimBatchMin = 10;
            network.Thread = 4;

            Stopwatch stopwatch = new Stopwatch();
            stopwatch.Reset();
            stopwatch.Start();

            network.Train(dataMatrix, labelMatrix, 0.0001);

            stopwatch.Stop();
            Console.WriteLine("Train Elapsed={0}", stopwatch.ElapsedMilliseconds * 0.001);

            double[][] result = new double[testDataMatrix.Length][];
            for (int i = 0; i < testDataMatrix.Length; i++)
            {
                result[i] = network.GetNetworkOutput(testDataMatrix[i]);
            }

            //Finalizzo il risultato
            for (int i = 0; i < testDataMatrix.Length; i++)
            {
                double max = double.MinValue;
                int index = 0;
                for (int j = 0; j < result[i].Length; j++)
                {
                    if (result[i][j] > max)
                    {
                        max = result[i][j];
                        index = j;
                    }
                    result[i][j] = 0.0;
                }
                result[i][index] = 1.0;
            }
            Console.WriteLine("Network error " + GetError(result, testLabelMatrix));
            Console.ReadLine();
        }

        static double GetError(
            double[][] output,
            double[][] expectedOutput)
        {
            double error = 0.0;
            for (int i = 0; i < output.Length; i++)
            {
                for (int j = 0; j < output[i].Length; j++)
                {
                    double b = output[i][j] - expectedOutput[i][j];
                    error += b * b;
                }
            }
            return error;
        }

        static void readDataset(
            string fileName,
            string[] separator,
            ref double[][] inputMatrix)
        {
            StreamReader file = new StreamReader(fileName);
            string line;
            int lineNumber = 0;
            while ((line = file.ReadLine()) != null)
                lineNumber++;

            file.DiscardBufferedData();
            file.BaseStream.Seek(0, System.IO.SeekOrigin.Begin);

            inputMatrix = new double[lineNumber][];
            
            int index = 0;
            while ((line = file.ReadLine()) != null)
            {
                string[] str = line.Split(separator, StringSplitOptions.None);
                inputMatrix[index] = new double[str.Length];
                for (int i = 0; i < str.Length; i++)
                {
                    inputMatrix[index][i] = Double.Parse(str[i], CultureInfo.InvariantCulture);
                }
                index++;
            }
        }


        #region Test

        static void TestExp()
        {
            Console.WriteLine("Error Exp " + TestError());
            Console.WriteLine("Perf Exp " + TestPerformancePlain());
            Console.WriteLine("Perf Exp8 " + TestPerformanceExp8());

            Console.WriteLine("Exp 16 " + Helper.Exp16(-5));
            Console.WriteLine("Exp " + Math.Exp(-5));
        }

        public static double TestError()
        {
            double emax = 0.0;
            for (double x = -10.0f; x < 10.0f; x += 0.00001)
            {
                double v0 = Math.Exp(x);
                double v1 = Helper.Exp16(x);
                double e = error(v0, v1);
                if (e > emax) emax = e;
            }
            return emax;
        }

        public static double TestPerformancePlain()
        {
            Stopwatch sw = new Stopwatch();
            sw.Start();
            for (int i = 0; i < 10; i++)
            {
                for (double x = -5.0f; x < 5.0f; x += 0.00001)
                {
                    Math.Exp(x);
                }
            }
            sw.Stop();
            return sw.Elapsed.TotalMilliseconds;
        }

        public static double TestPerformanceExp8()
        {
            Stopwatch sw = new Stopwatch();
            sw.Start();
            for (int i = 0; i < 10; i++)
            {
                for (double x = -5.0f; x < 5.0f; x += 0.00001)
                {
                    Helper.Exp16(x);
                }
            }
            sw.Stop();
            return sw.Elapsed.TotalMilliseconds;
        }

        public static double error(double v0, double v1)
        {
            return Math.Abs(v1 - v0);
        }

        #endregion
    }
}
