using MachineLearning.Helpers;
using System;
using System.Diagnostics;
using System.Globalization;
using System.IO;

namespace MachineLearning
{
    class Program
    {
        static void Main(string[] args)
        {

            //TestExp();
            int T =1000;
            int nFeature = 784;
            int nLabel = 10;
            double momentum = 0.0;
            IFunction[] type = new IFunction[] { new Sigmoid(1.0, 0.0), new Sigmoid(1.0, 0.0), new SoftMax() };
            IErrorFunction errorFunction = new CrossEntropyErrorFunc();
            //IFunction type = new Tanh(0.666666, 1.7159, 0.0);

            double[] eta = new double[] { 0.5, 0.5, 0.5, 0.5 };
            double[] dropoutValue = new double[] { 1.0, 1.0, 1.0 };
            int[] nodeLayer = new int[] { nFeature, 400, 20, nLabel };
            double[][] dataMatrix = null;
            double[][] labelMatrix = null;

            readDataset("data.dat", new[] { "  " }, ref dataMatrix);
            readDataset("tlabel.dat", new[] { " " },ref labelMatrix);

            double[][] testDataMatrix = null;
            double[][] testLabelMatrix = null;

            readDataset("testmnist.dat", new[] { "  " }, ref testDataMatrix);
            readDataset("tlabeltest.dat", new[] { " " }, ref testLabelMatrix);

            ANN network = new ANN(
                T, 
                nodeLayer, 
                eta, 
                momentum, 
                dropoutValue, 
                type,
                errorFunction);

            network.SetThread(6);

            #region Start test error

            double[][] result = new double[testDataMatrix.Length][];
            for (int i = 0; i < testDataMatrix.Length; i++)
            {
                result[i] = network.GetNetworkOutput(testDataMatrix[i]);
            }

            result = ResultFinalize(result);
            Console.WriteLine("Network error " + GetError(result, testLabelMatrix));

            #endregion

            Stopwatch stopwatch = new Stopwatch();
            stopwatch.Reset();
            stopwatch.Start();

            double[][] resultTrain = new double[dataMatrix.Length][];
            int nTrial = 80;
            for (int i = 0; i < nTrial; i++)
            {
                network.Train(dataMatrix, labelMatrix, 10);

                for (int j = 0; j < testDataMatrix.Length; j++)
                {
                    result[j] = network.GetNetworkOutput(testDataMatrix[j]);
                }
                result = ResultFinalize(result);
                Console.WriteLine("Network test error " + GetError(result, testLabelMatrix));

                for (int j = 0; j < dataMatrix.Length; j++)
                {
                    resultTrain[j] = network.GetNetworkOutput(dataMatrix[j]);
                }
                resultTrain = ResultFinalize(resultTrain);
                Console.WriteLine("Network train error " + GetError(resultTrain, labelMatrix));
            }
            
            stopwatch.Stop();
            Console.WriteLine("Train Elapsed={0}", stopwatch.ElapsedMilliseconds * 0.001);

            

            result = new double[testDataMatrix.Length][];
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


        static double[][] ResultFinalize(
            double[][] input)
        {
            //Finalizzo i dati risultanti
            for (int i = 0; i < input.Length; i++)
            {
                double max = double.MinValue;
                int index = 0;
                for (int j = 0; j < input[i].Length; j++)
                {
                    if (input[i][j] > max)
                    {
                        max = input[i][j];
                        index = j;
                    }
                    input[i][j] = 0.0;
                }
                input[i][index] = 1.0;
            }

            return input;
        }

        static double GetError(
            double[][] output,
            double[][] expectedOutput)
        {
            double error = 0.0;
            for (int i = 0; i < output.Length; i++)
            { 
                double b = 0.0;
                for (int j = 0; j < output[i].Length; j++)
                {
                    b += Math.Abs(output[i][j] - expectedOutput[i][j]);
                }
                if (Math.Abs(b) > 0.00001)
                    error += 1;
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

        //void TestNumeric()
        //{
        //    int nvalue = 10000000;
        //    stopwatch.Reset();
        //    stopwatch.Start();
        //    System.Numerics.Vector3[] testVector = new System.Numerics.Vector3[nvalue];

        //    //double[] vv = new double[] { 0.0, 0.0, 0.0, 0.0 };
        //    for (int i = 0; i < nvalue; i++)
        //        testVector[i] = new System.Numerics.Vector3(0, 0, 0);

        //    System.Numerics.Vector3 test;
        //    for (int i = 0; i < nvalue; i++)
        //        test = testVector[i] * 1.0f;

        //    stopwatch.Stop();
        //    Console.WriteLine("Engine Elapsed={0}", stopwatch.ElapsedMilliseconds);

        //    stopwatch.Reset();
        //    stopwatch.Start();

        //    PhysicsEngineMathUtility.Vector3[] testVector1 = new PhysicsEngineMathUtility.Vector3[nvalue];

        //    for (int i = 0; i < nvalue; i++)
        //        testVector1[i] = new PhysicsEngineMathUtility.Vector3(0.0, 0.0, 0.0);

        //    PhysicsEngineMathUtility.Vector3 test1;
        //    for (int i = 0; i < nvalue; i++)
        //        test1 = testVector1[i] * 1.0;

        //    stopwatch.Stop();
        //    Console.WriteLine("Engine Elapsed={0}", stopwatch.ElapsedMilliseconds);


        //}

        #endregion
    }
}
