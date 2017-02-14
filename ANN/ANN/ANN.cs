using System;
using System.Threading.Tasks;
using MachineLearning.Helpers;

namespace MachineLearning
{
    public sealed class ANN
    {
        #region Public Fields

        public int Ephocs { get; private set; }
        public int Layer { get; private set; }
        public int[] NodeLayer { get; private set; }
        public double[] LearningRate { get; private set; }
        public double Momentum { get; private set; }
        public int DimBatch { get; set; }
        public int DimBatchMin { get; set; }
        public int DimBatchMax { get; set; }
        public IFunction ActivationFunc { get; private set; }
        public double[] DropoutValue { get; private set; }

        public int Thread { get; set; }

        #endregion

        #region Private Fields

        private int[] WeigthLayer;
        private double[][] Weigth;
        private double[][] Bias;
        private double[][] DeltaBias;
        private double[][] NodeStatus;
        private double[][] Net;
        private double[][] Delta;
        private double[][] DeltaWeigth;
        private double[][] OldDeltaWeigth;
        private bool[][] Dropout;
        private double[] Error;
        private double[] EtaQ;
        private int[][] IndexBuffer;

        #endregion

        #region Constructor

        public ANN(
            int ephocs,
            int[] nodeLayer,
            double learningRate,
            double momentum,
            IFunction activationFunc)
        {
            Ephocs = ephocs;
            Layer = nodeLayer.Length;
            NodeLayer = nodeLayer;
            Momentum = momentum;
            ActivationFunc = activationFunc;
            DropoutValue = new double[Layer];

            LearningRate = new double[Layer];
            for (int i = 0; i < Layer; i++)
            {
                LearningRate[i] = learningRate;
                DropoutValue[i] = 1.0;
            }

            InitVariables();
        }

        public ANN(
            int ephocs,
            int[] nodeLayer,
            double[] learningRate,
            double momentum,
            IFunction activationFunc)
        {
            Ephocs = ephocs;
            Layer = nodeLayer.Length;
            NodeLayer = nodeLayer;
            LearningRate = learningRate;
            Momentum = momentum;
            ActivationFunc = activationFunc;
            DropoutValue = new double[Layer];

            for (int i = 0; i < Layer; i++)
            {
                DropoutValue[i] = 1.0;
            }

            InitVariables();
        }

        public ANN(
            int ephocs,
            int[] nodeLayer,
            double[] learningRate,
            double momentum,
            double[] dropoutValue,
            IFunction activationFunc)
        {
            Ephocs = ephocs;
            Layer = nodeLayer.Length;
            NodeLayer = nodeLayer;
            LearningRate = learningRate;
            Momentum = momentum;
            ActivationFunc = activationFunc;
            DropoutValue = dropoutValue;

            InitVariables();
        }

        #endregion

        #region Public Methods

        public void Train(
            double[][] input,
            double[][] output,
            double exitValue)
        {
            double mse, oldmse = 1.0;
            int index;
            double q = 1.0 / DimBatch;
            SetEtaQ(q);

            for (int i = 0; i < Ephocs; i++)
            {
                for (int k = 0; k < DimBatch; k++)
                {
                    //Randomize input
                    index = Helper.GetRandom(0, input.Length);

                    TrainExec(input[index], output[index]);
                }

                UpdateWeigth(q);

                #region Parte da sistemare
                if (i % 1000 == 0 || 
                    i + 1 == Ephocs)
                {
                    mse = GetMSE(input, output);

                    //------------------------------------
                    Console.WriteLine("nIter " + i + " mse: " + mse);

                    if (mse >= oldmse || i + 110 > Ephocs)
                    {
                        DimBatch = DimBatchMax;
                        Console.WriteLine("dimbatch " + DimBatch);
                    }
                    else
                    {
                        DimBatch = DimBatchMin;
                        Console.WriteLine("dimbatch " + DimBatch);
                    }
                    oldmse = mse;

                    if (mse < exitValue) break;
                }
                #endregion
            }
        }

        public double[] GetNetworkOutput(
            double[] input)
        {
            GetANNLayerOutput(Layer, input);

            double[] output = new double[NodeLayer[Layer - 1]];
            NodeStatus[Layer - 1].CopyTo(output, 0);

            return output;
        }

        public double[] GetNetworkOutput(
            int layer,
            double[] input)
        {
            GetANNLayerOutput(layer, input);

            double[] output = new double[NodeLayer[layer - 1]];
            NodeStatus[layer - 1].CopyTo(output, 0);

            return output;
        }

        public double[][] GetWeigth()
        {
            return Weigth;
        }

        public double[][] GetBias()
        {
            return Bias;
        }

        public void SetBias(double[][] bias)
        {
            Bias = bias;
        }

        public void SetWeigth(double[][] weigth)
        {
            Weigth = weigth;
        }

        #endregion

        #region Private Methods

        private void TrainExec(
            double[] input,
            double[] output)
        {
            ExecuteInternalNetworkOutput(Layer, input);

            //Forward error calculation
            Parallel.For(0, NodeLayer[Layer - 1], new ParallelOptions { MaxDegreeOfParallelism = Thread },
                z =>
                {
                    double err = NodeStatus[Layer - 1][z] - output[z];
                    Error[z] = err;
                    Delta[Layer - 1][z] = err * ActivationFunc.GetDerivative(Net[Layer - 1][z]);
                });

            ExecuteBackpropagation();
        }

        private void ExecuteBackpropagation()
        {
            //Delta hidden layer
            for (int z = Layer - 2; z > 0; z--)
            {
                int nLayer = z + 1;
                Parallel.For(0, NodeLayer[z], new ParallelOptions { MaxDegreeOfParallelism = Thread },
                    j =>
                    {   //layer node
                        double s = 0.0;
                        for (int h = 0; h < NodeLayer[nLayer]; h++)
                        {
                            int index = (h * NodeLayer[z]) + j;
                            if (!Dropout[z][index])
                            {
                                s += Delta[nLayer][h] * Weigth[z][index];
                            }  
                        }
                        Delta[z][j] = s * ActivationFunc.GetDerivative(Net[z][j]);
                    });
            }

            for (int z = 0; z < Layer - 1; z++)
            {
                Parallel.For(0, WeigthLayer[z], new ParallelOptions { MaxDegreeOfParallelism = Thread },
                    h =>
                    {
                        //int IndexBuffer[][] = (int)Math.Floor((double)(h / NodeLayer[z]));
                        DeltaWeigth[z][h] += Delta[z + 1][IndexBuffer[z][h]] * NodeStatus[z][h % NodeLayer[z]];
                    });
            }

            //delta bias
            for (int z = 1; z < Layer; z++)
            {
                for (int j = 0; j < NodeLayer[z]; j++)
                    DeltaBias[z][j] += Delta[z][j];
            }
        }

        private void UpdateWeigth(double q)
        {
            for (int z = 0; z < Layer - 1; z++)
            {
                double etaQ = EtaQ[z];
                Parallel.For(0, WeigthLayer[z], new ParallelOptions { MaxDegreeOfParallelism = Thread },
                    h =>
                    {
                        if (!Dropout[z][h])
                        {
                            double buf = etaQ * DeltaWeigth[z][h] + (Momentum * OldDeltaWeigth[z][h]);
                            Weigth[z][h] -= buf;
                            OldDeltaWeigth[z][h] = buf;
                        }
                    });
            }

            //Azzero i delta
            for (int z = 0; z < Layer - 1; z++)
                Array.Clear(DeltaWeigth[z], 0, DeltaWeigth[z].Length);

            //bias update
            for (int z = 1; z < Layer; z++)
            {
                double etaQ = EtaQ[z];
                for (int j = 0; j < NodeLayer[z]; j++)
                    Bias[z][j] -= etaQ * DeltaBias[z][j];
            }

            //azzero i bias
            for (int z = 0; z < Layer; z++)
                Array.Clear(DeltaBias[z], 0, DeltaBias[z].Length);
        }

        private void ExecuteInternalNetworkOutput(
            int layerNumber,
            double[] input)
        {
            //Input Layer
            Array.Copy(input, NodeStatus[0], NodeStatus[0].Length);

            //output node status
            for (int z = 1; z < layerNumber; z++)
            {
                int nLayer = z - 1;
                Parallel.For(0, NodeLayer[z], new ParallelOptions { MaxDegreeOfParallelism = Thread },
                j =>
                {
                    double s = 0.0;
                    int nodeLayer = NodeLayer[nLayer] * j;

                    if (DropoutValue[nLayer] < 1.0)
                    {
                        for (int h = 0; h < NodeLayer[nLayer]; h++)
                        {
                            Dropout[nLayer][nodeLayer + h] = Helper.GetRandomBernoulli(DropoutValue[nLayer]);
                            if (!Dropout[nLayer][nodeLayer + h])
                                s += NodeStatus[nLayer][h] * Weigth[nLayer][nodeLayer + h];
                        }
                    }
                    else
                    {
                        for (int h = 0; h < NodeLayer[nLayer]; h++)
                        {
                            s += NodeStatus[nLayer][h] * Weigth[nLayer][nodeLayer + h];
                        }
                    }

                    Net[z][j] = s + Bias[z][j];
                    NodeStatus[z][j] = ActivationFunc.GetResult(Net[z][j]);
                });
            }
        }

        private void GetANNLayerOutput(
            int layerNumber,
            double[] input)
        {
            //Input Layer
            Array.Copy(input, NodeStatus[0], NodeStatus[0].Length);

            //output node status
            for (int z = 1; z < layerNumber; z++)
            {
                int nLayer = z - 1;
                Parallel.For(0, NodeLayer[z], new ParallelOptions { MaxDegreeOfParallelism = Thread },
                j =>
                {
                    double s = 0.0;
                    int nodeLayer = NodeLayer[nLayer] * j;

                    for (int h = 0; h < NodeLayer[nLayer]; h++)
                        s += NodeStatus[nLayer][h] * Weigth[nLayer][nodeLayer + h] * DropoutValue[nLayer];
                   
                    Net[z][j] = s + Bias[z][j];
                    NodeStatus[z][j] = ActivationFunc.GetResult(Net[z][j]);
                });
            }
        }

        private void InitVariables()
        {
            WeigthLayer = new int[Layer - 1];
            Weigth = new double[Layer - 1][];
            Dropout = new bool[Layer - 1][];
            DeltaWeigth = new double[Layer - 1][];
            OldDeltaWeigth = new double[Layer - 1][];
            IndexBuffer = new int[Layer - 1][];
            
            Bias = new double[Layer][];
            DeltaBias = new double[Layer][];
            NodeStatus = new double[Layer][];
            Net = new double[Layer][];
            Delta = new double[Layer][];
            EtaQ = new double[Layer];
            
            for (int i = 0; i < Layer - 1; i++)
            { 
                WeigthLayer[i] = NodeLayer[i] * NodeLayer[i + 1];
                Weigth[i] = new double[WeigthLayer[i]];
                DeltaWeigth[i] = new double[WeigthLayer[i]];
                OldDeltaWeigth[i] = new double[WeigthLayer[i]];
                IndexBuffer[i] = new int[WeigthLayer[i]];
                Dropout[i] = new bool[WeigthLayer[i]];

                for (int j = 0; j < WeigthLayer[i]; j++)
                { 
                    Weigth[i][j] = 0.1 * Helper.GetRandomGaussian(0.0, 1.0);
                    IndexBuffer[i][j] = (int)Math.Floor((double)(j / NodeLayer[i]));
                    Dropout[i][j] = false;
                }
            }

            for (int i = 0; i < Layer; i++)
            {
                Bias[i] = new double[NodeLayer[i]];
                DeltaBias[i] = new double[NodeLayer[i]];
                NodeStatus[i] = new double[NodeLayer[i]];
                Net[i] = new double[NodeLayer[i]];
                Delta[i] = new double[NodeLayer[i]];
                
                for (int j = 0; j < NodeLayer[i]; j++)
                    Bias[i][j] = 0.1 * Helper.GetRandomGaussian(0.0, 1.0);
            }

            Error = new double[NodeLayer[Layer - 1]];
        }

        private void SetEtaQ(double q)
        {
            for (int i = 0; i < Layer; i++)
                EtaQ[i] = LearningRate[i] * q;
        }

        private double GetMSE(
            double[][] input, 
            double[][] output)
        {
            double total = 0.0;
            for (int i = 0; i < input.Length; i++)
            {
                GetNetworkOutput(input[i]);
                double buf = 0.0;
                for (int z = 0; z < NodeLayer[Layer - 1]; z++)
                {
                    double err = NodeStatus[Layer - 1][z] - output[i][z];
                    buf += err * err;
                }
                total += buf / NodeLayer[Layer - 1];
            }

            return total / input.Length;
        }

        //private double ActivationFunction(double x)
        //{
        //    switch(FType)
        //    {
        //        case FunctionType.Sigmoid:
        //            return Helper.Sigmoid(0.666666, 0.0, x);
        //        case FunctionType.Tanh:
        //            return Helper.Tanh(1.7159, 0.66666, 0.0, x);
        //        case FunctionType.SoftPlus:
        //            return Helper.SoftPlus(x);
        //        case FunctionType.ReLU:
        //            return Helper.ReLU(x);
        //        default:
        //            return x;
        //    }
        //}

        //private double DerivativeActivationFunction(double x)
        //{
        //    switch (FType)
        //    {
        //        case FunctionType.Sigmoid:
        //            return Helper.DerivativeSigmoid(0.666666, 0.0, x);
        //        case FunctionType.Tanh:
        //            return Helper.DerivativeTanh(1.7159, 0.66666, 0.0, x);
        //        case FunctionType.SoftPlus:
        //            return Helper.DerivativeSoftPlus(x);
        //        case FunctionType.ReLU:
        //            return Helper.ReLU(x);
        //        default:
        //            return x;
        //    }
        //}

        #endregion
    }
}
