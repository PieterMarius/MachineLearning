using MachineLearning.Helpers;
using System;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;

namespace MachineLearning
{
    class Program
    {
        #region Temporal Dataset Generation

        //        Description
        //-----------
        //The input has 6 channels.At any time all channels are 0 except for one
        //which has value 1 (i.e.the 6 channels are used for a one-hot encoding
        //of a 6 possible symbols).
        //The first two channels are reserved for symbols {A, B
        //    }, the others
        //to {c,d,e,f
        //}. At one random position `p0` in [1, L/10]
        //either A or B
        //is showed.The same happens at a second position `p1` in [5*L/10, 6*L/10].
        //At all other position a random symbol from { c,d,e,f } is used.
        //At the end of the sequence one has to predict the order in which the
        //symbols where provided(either AA, AB, BA or BB).

        private static void GenerateDataset(
            int timeLength,
            int dim,
            ref double[][] data,
            ref double[][] label)
        {
            data = new double[dim][];
            label = new double[dim][];
            double tVar = timeLength * 0.1;

            for (int i = 0; i < dim; i++)
            {
                data[i] = new double[6];
                label[i] = new double[4];
            }

            for (int i = 0; i < dim / timeLength; i++)
            {
                int symbolType = Helper.GetRandom(0, 4);

                label[(i * timeLength) + timeLength - 1][symbolType] = 1.0;
                                
                for (int j = 0; j < timeLength; j++)
                {
                    int index = (i * timeLength) + j;
                    int rnd = Helper.GetRandom(2, 6);

                    data[index][rnd] = 1.0;
                }

                int indexA = Helper.GetRandom(0, timeLength - 2);
                int indexB = Helper.GetRandom(indexA + 1, timeLength - 1);
                //if (tVar > 1)
                //    indexA = Helper.GetRandom(1, Convert.ToInt32(tVar));
                //else
                //    indexA = Helper.GetRandom(0, 1);

                //int indexB = Helper.GetRandom(Convert.ToInt32(5 * tVar), Convert.ToInt32(6 * tVar));

                switch (symbolType)
                {
                    case 0:
                        //AA
                        data[(i * timeLength) + indexA] = new double[6];
                        data[(i * timeLength) + indexA][0] = 1;
                        data[(i * timeLength) + indexB] = new double[6];
                        data[(i * timeLength) + indexB][0] = 1;
                        break;
                    case 1:
                        //AB
                        data[(i * timeLength) + indexA] = new double[6];
                        data[(i * timeLength) + indexA][0] = 1;
                        data[(i * timeLength) + indexB] = new double[6];
                        data[(i * timeLength) + indexB][1] = 1;
                        break;
                    case 2:
                        //BA
                        data[(i * timeLength) + indexA] = new double[6];
                        data[(i * timeLength) + indexA][1] = 1;
                        data[(i * timeLength) + indexB] = new double[6];
                        data[(i * timeLength) + indexB][0] = 1;
                        break;
                    case 3:
                        //BB
                        data[(i * timeLength) + indexA] = new double[6];
                        data[(i * timeLength) + indexA][1] = 1;
                        data[(i * timeLength) + indexB] = new double[6];
                        data[(i * timeLength) + indexB][1] = 1;
                        break;
                }
            }
        }

        private static void GenerateDataset1(
            int timeLength,
            int dim,
            ref double[][] data,
            ref double[][] label)
        {
            data = new double[dim][];
            label = new double[dim][];
            double tVar = timeLength * 0.1;

            for (int i = 0; i < dim; i++)
            {
                data[i] = new double[6];
                label[i] = new double[4];
            }

            for (int i = 0; i < dim / timeLength; i++)
            {
                int symbolType = Helper.GetRandom(0, 4);

                label[(i * timeLength) + timeLength - 1][symbolType] = 1;

                for (int j = 0; j < timeLength; j++)
                {
                    int index = (i * timeLength) + j;
                    int rnd = Helper.GetRandom(2, 6);

                    data[index][rnd] = 1.0;
                }

                int indexA;
                if (tVar > 1)
                    indexA = Helper.GetRandom(1, Convert.ToInt32(tVar));
                else
                    indexA = Helper.GetRandom(0, 1);

                int indexB = Helper.GetRandom(Convert.ToInt32(5 * tVar), Convert.ToInt32(6 * tVar));

                switch (symbolType)
                {
                    case 0:
                        //AA
                        data[(i * timeLength) + indexA] = new double[6];
                        data[(i * timeLength) + indexA][0] = 1;
                        data[(i * timeLength) + indexB] = new double[6];
                        data[(i * timeLength) + indexB][0] = 1;
                        break;
                    case 1:
                        //AB
                        data[(i * timeLength) + indexA] = new double[6];
                        data[(i * timeLength) + indexA][0] = 1;
                        data[(i * timeLength) + indexB] = new double[6];
                        data[(i * timeLength) + indexB][1] = 1;
                        break;
                    case 2:
                        //BA
                        data[(i * timeLength) + indexA] = new double[6];
                        data[(i * timeLength) + indexA][1] = 1;
                        data[(i * timeLength) + indexB] = new double[6];
                        data[(i * timeLength) + indexB][0] = 1;
                        break;
                    case 3:
                        //BB
                        data[(i * timeLength) + indexA] = new double[6];
                        data[(i * timeLength) + indexA][1] = 1;
                        data[(i * timeLength) + indexB] = new double[6];
                        data[(i * timeLength) + indexB][1] = 1;
                        break;
                }
            }
        }

        #endregion

        static void Main(string[] args)
        {

            //TestExp();
            //ANNExample();
            RNNExample();
            
        }

        static void RNNExample()
        {
            int T = 1000;
            int nFeature = 6;
            int nLabel = 4;
            double momentum = 0.0;
            //IFunction[] type = new IFunction[] { new Sigmoid(1.0, 0.0), new SoftMax() };
            //IFunction[] type = new IFunction[] { new Tanh(0.666666, 1.7159, 0.0), new SoftMax() };
            //IFunction[] type = new IFunction[] { new Tanh(0.666666, 1.7159, 0.0), new SoftMax() };
            IFunction[] type = new IFunction[] { new Sigmoid(1.0, 0.0), new Sigmoid(1.0, 0.0) };
            //IErrorFunction errorFunction = new CrossEntropyErrorFunc();
            IErrorFunction errorFunction = new MediumSquareErrorFunction();

            double[] eta = new double[] { 0.01, 0.01, 0.01 };
            double[] dropoutValue = new double[] { 1.0, 1.0 };
            int[] nodeLayer = new int[] { nFeature, 50, nLabel };
            int timeLength = 40;

            double[][] dataMatrix = null;
            double[][] labelMatrix = null;

            //Training set
            GenerateDataset(timeLength, 12000, ref dataMatrix, ref labelMatrix);

            //readDataset("international-airline-passengers.csv", new[] { "  " }, ref dataMatrix);
            //readDataset("international-airline-passengers.csv", new[] { " " }, ref labelMatrix);

            //NormalizeDataColumn(ref dataMatrix);
            //NormalizeDataColumn(ref labelMatrix);

            RNN network = new RNN(
                T,
                timeLength,
                10,
                nodeLayer,
                eta,
                momentum,
                dropoutValue,
                type,
                errorFunction);

            int nTrial = 60;
            for (int i = 0; i < nTrial; i++)
            {
                int batch = 1;// Helper.GetRandom(1, 5);
                network.Train(dataMatrix, labelMatrix, batch, true);
                Console.WriteLine("Trial " + i);
                Console.WriteLine();
            }

            double[][] testDataMatrix = null;
            double[][] testLabelMatrix = null;

            //Network test
            Console.WriteLine("TEST");
            double totErr = 0.0;
            for (int i = 0; i < 200; i++)
            {
                GenerateDataset(timeLength, timeLength, ref testDataMatrix, ref testLabelMatrix);
                double[] res = network.GetNetworkOutput(testDataMatrix);
                Console.WriteLine("Res " + res[0] + " " + res[1] + " " + res[2] + " " + res[3]);
                Console.WriteLine("Expected " + testLabelMatrix[timeLength - 1][0] + " " +
                                                testLabelMatrix[timeLength - 1][1] + " " +
                                                testLabelMatrix[timeLength - 1][2] + " " +
                                                testLabelMatrix[timeLength - 1][3] + " ");
                //double diff = (testLabelMatrix[timeLength - 1][0] - res[0]);
                //totErr = diff * diff;

                //Console.WriteLine("Diff " + diff);
                Console.WriteLine();
            }

            //Console.WriteLine("Test err " + totErr / 100);

            Console.ReadLine();
        }

        static void ANNExample()
        {
            int T = 1000;
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
            readDataset("tlabel.dat", new[] { " " }, ref labelMatrix);

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
            file.BaseStream.Seek(0, SeekOrigin.Begin);

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

        static void NormalizeDataColumn(
            ref double[][] data)
        {
            double max = double.MinValue;
            double min = double.MaxValue;
            for (int i = 0; i < data.Length; i++)
            {
                for (int j = 0; j < data[i].Length; j++)
                {
                    if (data[i][j] > max)
                        max = data[i][j];
                    if (data[i][j] < min)
                        min = data[i][j];
                }
            }

            double diff = max - min;
            for (int i = 0; i < data.Length; i++)
            {
                for (int j = 0; j < data[i].Length; j++)
                { 
                    data[i][j] = (data[i][j] - min) / diff;
                    //Console.WriteLine(data[i][j]);
                }
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
