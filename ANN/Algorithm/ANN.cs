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
        public IFunction[] ActivationFunc { get; private set; }
        public IErrorFunction ErrorFunc { get; private set; }
        public double[] DropoutValue { get; private set; }
        public int Thread { get; private set; }

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
        private double[] EtaQ;
        private int[][] IndexBuffer;

        #endregion

        #region Constructor

        public ANN(
            int ephocs,
            int[] nodeLayer,
            double[] learningRate,
            double momentum,
            double[] dropoutValue,
            IFunction[] activationFunc,
            IErrorFunction errorFunc)
        {
            Ephocs = ephocs;
            Layer = nodeLayer.Length;
            NodeLayer = nodeLayer;
            LearningRate = learningRate;
            Momentum = momentum;
            ActivationFunc = activationFunc;
            DropoutValue = dropoutValue;
            ErrorFunc = errorFunc;
            
            InitVariables();
        }
        
        public ANN(
            int ephocs,
            int[] nodeLayer,
            double learningRate,
            double momentum,
            IFunction[] activationFunc,
            IErrorFunction errorFunc)
        {
            Ephocs = ephocs;
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

        public ANN(
            int ephocs,
            int[] nodeLayer,
            double[] learningRate,
            double momentum,
            IFunction[] activationFunc,
            IErrorFunction errorFunc)
        {
            Ephocs = ephocs;
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
            int index;
            double q = 1.0 / batchSize;
            SetEtaQ(q);

            for (int i = 0; i < Ephocs; i++)
            {
                for (int k = 0; k < batchSize; k++)
                {
                    //Randomize input
                    index = Helper.GetRandom(0, input.Length);

                    TrainExec(input[index], target[index]);
                }

                UpdateWeigth(q);
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

        public void SetThread(int thread)
        {
            Thread = thread;
        }

        public double GetNetworkMSE(
            double[][] input,
            double[][] target)
        {
            double total = 0.0;
            for (int i = 0; i < input.Length; i++)
            {
                GetNetworkOutput(input[i]);
                double buf = 0.0;
                for (int z = 0; z < NodeLayer[Layer - 1]; z++)
                {
                    double err = NodeStatus[Layer - 1][z] - target[i][z];
                    buf += err * err;
                }
                total += buf / NodeLayer[Layer - 1];
            }

            return total / input.Length;
        }

        public void SetDropoutValue(double[] dropout)
        {
            DropoutValue = dropout;
        }

        #endregion

        #region Private Methods

        private void TrainExec(
            double[] input,
            double[] target)
        {
            ExecuteInternalNetworkOutput(Layer, input);

            ForwardError(target);

            ExecuteBackpropagation();
        }

        /// <summary>
        /// Forward error calculation
        /// </summary>
        private void ForwardError(double[] target)
        {
            Parallel.For(0, NodeLayer[Layer - 1], new ParallelOptions { MaxDegreeOfParallelism = Thread },
                z =>
                {
                    Delta[Layer - 1][z] = ErrorFunc.GetDeltaForwardError(
                                                    NodeStatus[Layer - 1][z],
                                                    Net[Layer - 1][z],
                                                    target[z],
                                                    ActivationFunc[Layer - 2]);
                });
        }

        /// <summary>
        /// Backpropagation
        /// </summary>
        private void ExecuteBackpropagation()
        {
            //TODO eliminare Delta[0]

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
                        Delta[z][j] = s * ActivationFunc[z - 1].GetDerivative(Net[z][j]);
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
                    NodeStatus[z][j] = ActivationFunc[nLayer].GetResult(Net[z][j]);
                });

                ExecExponentialFunc(z, nLayer);
            }
        }

        private void ExecExponentialFunc(
            int z,
            int nLayer)
        {
            if (ActivationFunc[nLayer] is IExponentialFunction)
            {
                IExponentialFunction expFunc = (IExponentialFunction)ActivationFunc[nLayer];
                expFunc.SetExponentialSum(Net[z]);

                Parallel.For(0, NodeLayer[z], new ParallelOptions { MaxDegreeOfParallelism = Thread },
                j =>
                {
                    NodeStatus[z][j] = expFunc.GetExponential(Net[z][j]);
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
                    NodeStatus[z][j] = ActivationFunc[nLayer].GetResult(Net[z][j]);
                });

                ExecExponentialFunc(z, nLayer);
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
            
            for (int i = 0; i < Layer; i++)
            {
                if (i < Layer - 1)
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

                Bias[i] = new double[NodeLayer[i]];
                DeltaBias[i] = new double[NodeLayer[i]];
                NodeStatus[i] = new double[NodeLayer[i]];
                Net[i] = new double[NodeLayer[i]];
                Delta[i] = new double[NodeLayer[i]];

                for (int j = 0; j < NodeLayer[i]; j++)
                    Bias[i][j] = 0.1 * Helper.GetRandomGaussian(0.0, 1.0);
            }
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
