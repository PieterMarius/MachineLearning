using System;
using System.Threading.Tasks;
using MachineLearning.Helpers;

namespace MachineLearning
{
    public sealed class ANN
    {
        #region Public Fields

        public int T { get; private set; }
        public int Layer { get; private set; }
        public int[] NodeLayer { get; private set; }
        public double[] Eta { get; private set; }
        public double Momentum { get; private set; }
        public int DimBatch { get; set; }
        public int DimBatchMin { get; set; }
        public int DimBatchMax { get; set; }
        public IFunction ActivationFunc { get; set; }

        public int Thread { get; set; }

        #endregion

        #region Private Fields

        private int TotalNode;
        private int[] WeigthLayer;
        private double[][] Weigth;
        private double[][] Bias;
        private double[][] DeltaBias;
        private double[][] NodeStatus;
        private double[][] Net;
        private double[][] Delta;
        private double[][] DeltaWeigth;
        private double[][] OldDeltaWeigth;
        private double[] Error;

        #endregion

        #region Constructor

        public ANN(
            int t,
            int nLayer,
            int[] nodeLayer,
            double[] eta,
            double momentum,
            IFunction activationFunc)
        {
            T = t;
            Layer = nLayer;
            NodeLayer = nodeLayer;
            Eta = eta;
            Momentum = momentum;
            ActivationFunc = activationFunc;

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

            for (int i = 0; i < T; i++)
            {
                for (int k = 0; k < DimBatch; k++)
                {
                    //Randomize input

                    index = (int)Math.Floor(Helper.GetRandom(0.0, 1.0) * input.Length);

                    ExecuteInternalNetworkOutput(input[index]);
                    
                    //Calcolo errore output
                    for (int z = 0; z < NodeLayer[Layer - 1]; z++)
                    {
                        double err = NodeStatus[Layer - 1][z] - output[index][z];
                        Error[z] = err;
                        Delta[Layer - 1][z] = Error[z] *
                                              ActivationFunc.GetDerivative(Net[Layer - 1][z]);
                    }
                    
                    //Backpropagation

                    //Delta hidden layer
                    for (int z = Layer - 2; z > 0; z--)
                    { //for each layer
                        Parallel.For(0, NodeLayer[z], new ParallelOptions { MaxDegreeOfParallelism = Thread },
                            j =>
                            { //layer node
                                double s = 0.0;
                                int nLayer = z + 1;
                                for (int h = 0; h < NodeLayer[nLayer]; h++)
                                { //weight number 
                                    s += Delta[nLayer][h] * Weigth[z][(h * NodeLayer[z]) + j];
                                }
                                Delta[z][j] = s * ActivationFunc.GetDerivative(Net[z][j]);
                            });

                    }

                    for (int z = 0; z < Layer - 1; z++)
                    {
                        Parallel.For(0, WeigthLayer[z], new ParallelOptions { MaxDegreeOfParallelism = Thread },
                            h =>
                            {
                                DeltaWeigth[z][h] += Delta[z + 1][(int)Math.Floor((double)(h / NodeLayer[z]))] * NodeStatus[z][h % NodeLayer[z]];
                            });
                    }

                    //delta bias
                    for (int z = 1; z < Layer; z++)
                    {
                        for (int j = 0; j < NodeLayer[z]; j++)
                        {
                            DeltaBias[z][j] += Delta[z][j];
                        }
                    }

                }

                //Aggiorno i pesi in base alla dimensione del batch
                double q = 1.0 / (double)DimBatch;

                for (int z = 0; z < Layer - 1; z++)
                {
                    Parallel.For(0, WeigthLayer[z], new ParallelOptions { MaxDegreeOfParallelism = Thread },
                        h =>
                        {
                            double buf = Eta[z] * q * DeltaWeigth[z][h] + (Momentum * OldDeltaWeigth[z][h]);
                            Weigth[z][h] -= buf;
                            OldDeltaWeigth[z][h] = buf;
                        });
                }
                
                //Azzero i delta
                for (int z = 0; z < Layer - 1; z++)
                    Array.Clear(DeltaWeigth[z], 0, DeltaWeigth[z].Length);

                //bias update
                for (int z = 1; z < Layer; z++)
                {
                    for (int j = 0; j < NodeLayer[z]; j++)
                    {
                        Bias[z][j] -= Eta[z] * q * DeltaBias[z][j];
                    }
                }

                //azzero i bias
                for (int z = 0; z < Layer; z++)
                    Array.Clear(DeltaBias[z], 0, DeltaBias[z].Length);
                
                //Parte da sistemare
                if (i % 1000 == 0 || 
                    i + 1 == T)
                {
                    mse = GetMSE(input, output);

                    //------------------------------------
                    Console.WriteLine("nIter " + i + " mse: " + mse);

                    if (mse >= oldmse || i + 110 > T)
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
            }
        }

        public double[] GetNetworkOutput(
            double[] input)
        {
            ExecuteInternalNetworkOutput(input);

            double[] output = new double[NodeLayer[Layer - 1]];
            NodeStatus[Layer - 1].CopyTo(output, 0);

            return output;
        }

        #endregion

        #region Private Methods

        private void InitVariables()
        {
            for (int i = 0; i < Layer; i++)
                TotalNode += NodeLayer[i];

            WeigthLayer = new int[Layer - 1];
            Weigth = new double[Layer - 1][];
            DeltaWeigth = new double[Layer - 1][];
            OldDeltaWeigth = new double[Layer - 1][];

            Bias = new double[Layer][];
            DeltaBias = new double[Layer][];
            NodeStatus = new double[Layer][];
            Net = new double[Layer][];
            Delta = new double[Layer][];
            
            for (int i = 0; i < Layer - 1; i++)
            { 
                WeigthLayer[i] = NodeLayer[i] * NodeLayer[i + 1];
                Weigth[i] = new double[WeigthLayer[i]];
                DeltaWeigth[i] = new double[WeigthLayer[i]];
                OldDeltaWeigth[i] = new double[WeigthLayer[i]];

                for (int j = 0; j < WeigthLayer[i]; j++)
                    Weigth[i][j] = 0.1 * Helper.GetRandomGaussian(0.0, 1.0);
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

        private void ExecuteInternalNetworkOutput(
            double[] input)
        {
            //input node status (first layer)
            for (int z = 0; z < NodeLayer[0]; z++)
                NodeStatus[0][z] = input[z];

            //output node status
            for (int z = 1; z < Layer; z++)
            { //for each layer
                Parallel.For(0, NodeLayer[z], new ParallelOptions { MaxDegreeOfParallelism = Thread },
                j =>
                { //layer node
                    double s = 0.0;
                    int nLayer = z - 1;
                    int nodeLayer = NodeLayer[nLayer] * j;

                    for (int h = 0; h < NodeLayer[nLayer]; h++)
                        s += NodeStatus[nLayer][h] * Weigth[nLayer][nodeLayer + h];

                    Net[z][j] = s + Bias[z][j];
                    NodeStatus[z][j] = ActivationFunc.GetResult(Net[z][j]);
                });
            }
        }

        private double GetMSE(
            double[][] input, 
            double[][] output)
        {
            //MSE
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
