using MachineLearning.Helpers;
using System;
using System.Threading.Tasks;

namespace MachineLearning
{
    public sealed class RNN
    {
        #region Public Fields

        public int Ephocs { get; private set; }
        public int Layer { get; private set; }
        public int[] NodeLayer { get; private set; }
        public int TimeStepSize { get; private set; }
        public double[] LearningRate { get; private set; }
        public double Momentum { get; private set; }
        public IFunction[] ActivationFunc { get; private set; }
        public IErrorFunction ErrorFunc { get; private set; }
        public double[] DropoutValue { get; private set; }
        public int Thread { get; private set; }

        #endregion

        #region Private Fields

        private int[] WeigthLayer;
        private double[][] Weigth;
        private double[][] TimeWeigth;
        private double[][] Bias;
        private double[][] DeltaBias;
        private double[][][] NodeStatus;
        private double[][][] Net;
        private double[][][] Delta;
        private double[][] DeltaWeigth;
        private double[][] OldDeltaWeigth;
        private bool[][] Dropout;
        private double[] EtaQ;
        private int[][] IndexBuffer;

        #endregion

        #region Constructor

        public RNN(
            int ephocs,
            int timeStepSize,
            int[] nodeLayer,
            double[] learningRate,
            double momentum,
            double[] dropoutValue,
            IFunction[] activationFunc,
            IErrorFunction errorFunc)
        {
            Ephocs = ephocs;
            TimeStepSize = timeStepSize;
            Layer = nodeLayer.Length;
            NodeLayer = nodeLayer;
            LearningRate = learningRate;
            Momentum = momentum;
            ActivationFunc = activationFunc;
            DropoutValue = dropoutValue;
            ErrorFunc = errorFunc;

            InitVariables();
        }

        public RNN(
            int ephocs,
            int timeStepSize,
            int[] nodeLayer,
            double learningRate,
            double momentum,
            IFunction[] activationFunc,
            IErrorFunction errorFunc)
        {
            Ephocs = ephocs;
            TimeStepSize = timeStepSize;
            Layer = nodeLayer.Length;
            NodeLayer = nodeLayer;
            Momentum = momentum;
            ActivationFunc = activationFunc;
            ErrorFunc = errorFunc;
            DropoutValue = new double[Layer];

            LearningRate = new double[Layer];
            for (int i = 0; i < Layer; i++)
            {
                LearningRate[i] = learningRate;
                DropoutValue[i] = 1.0;
            }

            InitVariables();
        }

        public RNN(
            int ephocs,
            int timeStepSize,
            int[] nodeLayer,
            double[] learningRate,
            double momentum,
            IFunction[] activationFunc,
            IErrorFunction errorFunc)
        {
            Ephocs = ephocs;
            TimeStepSize = timeStepSize;
            Layer = nodeLayer.Length;
            NodeLayer = nodeLayer;
            LearningRate = learningRate;
            Momentum = momentum;
            ActivationFunc = activationFunc;
            ErrorFunc = errorFunc;
            DropoutValue = new double[Layer];

            for (int i = 0; i < Layer; i++)
            {
                DropoutValue[i] = 1.0;
            }

            InitVariables();
        }

        #endregion

        #region Public Methods

        public void Train(
            double[][] input,
            double[][] target,
            int batchSize)
        {
            int index = 0;
            double q = 1.0 / batchSize;
            SetEtaQ(q);

            for (int i = 0; i < Ephocs; i++)
            {
                for (int k = 0; k < batchSize; k++)
                {
                    NodeStatus[0] = NodeStatus[TimeStepSize - 1];

                    for (int w = 1; w < TimeStepSize; w++)
                    {
                        index = index % input.Length;

                        ExecuteInternalNetworkOutput(Layer, input[index], w);
                                                
                        index++;
                    }

                    //TODO verificare come gestire input
                    ForwardError(target[index], TimeStepSize - 1);



                    //ExecuteBackpropagation();
                }

                UpdateWeigth(q);
            }
        }

        //public double[] GetNetworkOutput(
        //    double[] input)
        //{

        //    GetANNLayerOutput(Layer, input);

        //    double[] output = new double[NodeLayer[Layer - 1]];
        //    NodeStatus[Layer - 1].CopyTo(output, 0);

        //    return output;
        //}

        //public double[] GetNetworkOutput(
        //    int layer,
        //    double[] input)
        //{
        //    GetANNLayerOutput(layer, input);

        //    double[] output = new double[NodeLayer[layer - 1]];
        //    NodeStatus[layer - 1].CopyTo(output, 0);

        //    return output;
        //}

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

        public void SetThread(int thread)
        {
            Thread = thread;
        }

        //public double GetNetworkMSE(
        //    double[][] input,
        //    double[][] target)
        //{
        //    double total = 0.0;
        //    for (int i = 0; i < input.Length; i++)
        //    {
        //        GetNetworkOutput(input[i]);
        //        double buf = 0.0;
        //        for (int z = 0; z < NodeLayer[Layer - 1]; z++)
        //        {
        //            double err = NodeStatus[Layer - 1][z] - target[i][z];
        //            buf += err * err;
        //        }
        //        total += buf / NodeLayer[Layer - 1];
        //    }

        //    return total / input.Length;
        //}

        public void SetDropoutValue(double[] dropout)
        {
            DropoutValue = dropout;
        }

        #endregion

        #region Private Methods
        
        /// <summary>
        /// Forward error calculation
        /// </summary>
        private void ForwardError(
            double[] target,
            int step)
        {
            Parallel.For(0, NodeLayer[Layer - 1], new ParallelOptions { MaxDegreeOfParallelism = Thread },
                z =>
                {
                    Delta[step][Layer - 1][z] = ErrorFunc.GetDeltaForwardError(
                                                    NodeStatus[step][Layer - 1][z],
                                                    Net[step][Layer - 1][z],
                                                    target[z],
                                                    ActivationFunc[Layer - 2]);
                });
        }

        /// <summary>
        /// Backpropagation
        /// </summary>
        private void ExecuteBackpropagation()
        {
            for (int t = TimeStepSize - 1; t > 0; t--)
            {
                //Delta hidden layer
                for (int z = Layer - 2; z > 0; z--)
                {
                    int nLayer = z + 1;

                    //Parallel.For(0, NodeLayer[z], new ParallelOptions { MaxDegreeOfParallelism = Thread },
                    //    j =>
                    //layer node
                    for (int j = 0; j < NodeLayer[z]; j++)
                    {
                        double s = 0.0;
                        for (int h = 0; h < NodeLayer[nLayer]; h++)
                        {
                            int index = (h * NodeLayer[z]) + j;
                            if (!Dropout[z][index])
                            {
                                s += Delta[t][nLayer][h] * Weigth[z][index];
                            }
                        }

                        //for (int h = 0; h < NodeLayer[z]; h++)
                        //{s += Delta[t][nLayer][j] * TimeWeigth[z][j];}
                        //nLayer -1
                        //All'ultimo timestep non aggiorno i pesi memoria
                        if (t + 2 <= TimeStepSize)
                            s += Delta[t + 1][nLayer - 1][j] * TimeWeigth[z - 1][j];

                        Delta[t][z][j] = s * ActivationFunc[z - 1].GetDerivative(Net[t][z][j]);
                    }

                }
            }
            for (int z = 0; z < Layer - 1; z++)
            {
                Parallel.For(0, WeigthLayer[z], new ParallelOptions { MaxDegreeOfParallelism = Thread },
                    h =>
                    {
                        //int IndexBuffer[][] = (int)Math.Floor((double)(h / NodeLayer[z]));
                        //DeltaWeigth[z][h] += Delta[z + 1][IndexBuffer[z][h]] * NodeStatus[z][h % NodeLayer[z]];
                    });
            }

            //delta bias
            for (int z = 1; z < Layer; z++)
            {
                for (int j = 0; j < NodeLayer[z]; j++)
                { 
                    //DeltaBias[z][j] += Delta[z][j];
                }
            }
        }

        /// <summary>
        /// Weigth updating
        /// </summary>
        /// <param name="q"></param>
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

            //Bias update
            for (int z = 1; z < Layer; z++)
            {
                double etaQ = EtaQ[z];
                for (int j = 0; j < NodeLayer[z]; j++)
                    Bias[z][j] -= etaQ * DeltaBias[z][j];
            }

            //Turn delta to zero
            for (int z = 0; z < Layer - 1; z++)
                Array.Clear(DeltaWeigth[z], 0, DeltaWeigth[z].Length);

            //Turn bias to zero
            for (int z = 0; z < Layer; z++)
                Array.Clear(DeltaBias[z], 0, DeltaBias[z].Length);
        }

        private void ExecuteInternalNetworkOutput(
            int layerNumber,
            double[] input,
            int step)
        {
            //Input Layer
            Array.Copy(input, NodeStatus[step][0], NodeStatus[step][0].Length);

            //output node status
            for (int z = 1; z < layerNumber; z++)
            {
                int nLayer = z - 1;
                
                //Parallel.For(0, NodeLayer[z], new ParallelOptions { MaxDegreeOfParallelism = Thread },
                //j =>
                for(int j=0;j<NodeLayer[z];j++)
                {
                    double s = 0.0;
                    int nodeLayer = NodeLayer[nLayer] * j;

                    if (DropoutValue[nLayer] < 1.0)
                    {
                        for (int h = 0; h < NodeLayer[nLayer]; h++)
                        {
                            Dropout[nLayer][nodeLayer + h] = Helper.GetRandomBernoulli(DropoutValue[nLayer]);
                            if (!Dropout[nLayer][nodeLayer + h])
                                s = WeigthSum(nLayer, s, nodeLayer, h, step);
                        }
                    }
                    else
                    {
                        for (int h = 0; h < NodeLayer[nLayer]; h++)
                        {
                            s = WeigthSum(nLayer, s, nodeLayer, h, step);
                        }
                    }

                    double st = 0.0;
                    
                    //Check for output layer
                    if (z + 1 < layerNumber)
                    {
                        int nodeTimeLayer = NodeLayer[z] * j;

                        for (int h = 0; h < NodeLayer[z]; h++)
                        {
                            st += NodeStatus[step - 1][z][h] * TimeWeigth[nLayer][nodeTimeLayer + h];
                        }

                    }
                    Net[step][z][j] = s + st + Bias[z][j];
                    NodeStatus[step][z][j] = ActivationFunc[nLayer].GetResult(Net[step][z][j]);
                }

                ExecExponentialFunc(z, nLayer, step);
            }
        }
                
        private double WeigthSum(
            int nLayer, 
            double s, 
            int nodeLayer, 
            int h,
            int step)
        {
            s += NodeStatus[step][nLayer][h] * Weigth[nLayer][nodeLayer + h];
            return s;
        }
                
        private void ExecExponentialFunc(
            int z,
            int nLayer,
            int step)
        {
            if (ActivationFunc[nLayer] is IExponentialFunction)
            {
                IExponentialFunction expFunc = (IExponentialFunction)ActivationFunc[nLayer];
                expFunc.SetExponentialSum(Net[step][z]);

                Parallel.For(0, NodeLayer[z], new ParallelOptions { MaxDegreeOfParallelism = Thread },
                j =>
                {
                    NodeStatus[step][z][j] = expFunc.GetExponential(Net[step][z][j]);
                });
            }
        }

        //private void GetANNLayerOutput(
        //    int layerNumber,
        //    double[] input)
        //{
        //    //Input Layer
        //    Array.Copy(input, NodeStatus[0], NodeStatus[0].Length);

        //    //output node status
        //    for (int z = 1; z < layerNumber; z++)
        //    {
        //        int nLayer = z - 1;
        //        Parallel.For(0, NodeLayer[z], new ParallelOptions { MaxDegreeOfParallelism = Thread },
        //        j =>
        //        {
        //            double s = 0.0;
        //            int nodeLayer = NodeLayer[nLayer] * j;

        //            for (int h = 0; h < NodeLayer[nLayer]; h++)
        //                s += NodeStatus[nLayer][h] * Weigth[nLayer][nodeLayer + h] * DropoutValue[nLayer];

        //            Net[z][j] = s + Bias[z][j];
        //            NodeStatus[z][j] = ActivationFunc[nLayer].GetResult(Net[z][j]);
        //        });

        //        ExecExponentialFunc(z, nLayer);
        //    }
        //}

        private void InitVariables()
        {
            WeigthLayer = new int[Layer - 1];
            Weigth = new double[Layer - 1][];
            Dropout = new bool[Layer - 1][];
            DeltaWeigth = new double[Layer - 1][];
            OldDeltaWeigth = new double[Layer - 1][];
            IndexBuffer = new int[Layer - 1][];

            TimeWeigth = new double[Layer - 2][];
            for (int i = 0; i < Layer - 2; i++)
            {
                int dimTimeWeigth = NodeLayer[i + 1] * NodeLayer[i + 1];
                TimeWeigth[i] = new double[dimTimeWeigth];
                
                for (int k = 0; k < dimTimeWeigth; k++)
                    TimeWeigth[i][k] = 0.0; 
            }

            Bias = new double[Layer][];
            DeltaBias = new double[Layer][];
            EtaQ = new double[Layer];

            Delta = new double[TimeStepSize][][];
            NodeStatus = new double[TimeStepSize][][];
            Net = new double[TimeStepSize][][];
            
            for (int i = 0; i < TimeStepSize; i++)
            {
                Delta[i] = new double[Layer][];
                NodeStatus[i] = new double[Layer][];
                Net[i] = new double[Layer][];

                for (int j = 0; j < Layer; j++)
                {
                    NodeStatus[i][j] = new double[NodeLayer[j]];
                    Net[i][j] = new double[NodeLayer[j]];
                    Delta[i][j] = new double[NodeLayer[j]];
                }
            }

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
               
                for (int j = 0; j < NodeLayer[i]; j++)
                    Bias[i][j] = 0.1 * Helper.GetRandomGaussian(0.0, 1.0);
            }

            Thread = 2;
        }

        private void SetEtaQ(double q)
        {
            for (int i = 0; i < Layer; i++)
                EtaQ[i] = LearningRate[i] * q;
        }

        //TODO throw new Exception
        private void CheckNetworkInputCoherence()
        {

        }

        #endregion
    }
}
