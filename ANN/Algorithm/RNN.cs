using MachineLearning.Helpers;
using System;
using System.Threading.Tasks;

namespace MachineLearning
{
    public sealed class RNN
    {
        #region Public Fields

        public int Ephocs { get; private set; }
        public int[] NodeLayer { get; private set; }
        public int TimeStepSize { get; private set; }
        public int BpttTruncate { get; private set; }
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
        private double[][] Delta;
        private double[][] DeltaWeigth;
        private double[][] OldDeltaWeigth;
        private double[][] DeltaTime;
        private bool[][] Dropout;
        private double[] EtaQ;
        private int[][] IndexBuffer;
        private int[][] TimeIndexBuffer;

        #endregion

        #region Constructor

        public RNN(
            int ephocs,
            int timeStepSize,
            int bpttTruncate,
            int[] nodeLayer,
            double[] learningRate,
            double momentum,
            double[] dropoutValue,
            IFunction[] activationFunc,
            IErrorFunction errorFunc)
        {
            Ephocs = ephocs;
            TimeStepSize = timeStepSize;
            NodeLayer = nodeLayer;
            LearningRate = learningRate;
            Momentum = momentum;
            ActivationFunc = activationFunc;
            DropoutValue = dropoutValue;
            ErrorFunc = errorFunc;
            BpttTruncate = bpttTruncate;

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
            NodeLayer = nodeLayer;
            Momentum = momentum;
            ActivationFunc = activationFunc;
            ErrorFunc = errorFunc;
            DropoutValue = new double[3];

            LearningRate = new double[3];
            for (int i = 0; i < 3; i++)
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
            NodeLayer = nodeLayer;
            LearningRate = learningRate;
            Momentum = momentum;
            ActivationFunc = activationFunc;
            ErrorFunc = errorFunc;
            DropoutValue = new double[3];

            for (int i = 0; i < 3; i++)
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
            int batchSize,
            bool checkErrorFromLastStep)
        {
            int index = 0;
            double q = 1.0 / batchSize;
            SetEtaQ(q);

            //TEST
            tot = 0.0;

            for (int i = 0; i < Ephocs; i++)
            {
                for (int k = 0; k < batchSize; k++)
                {
                    for (int w = 0; w < TimeStepSize; w++)
                    {
                        index = index % input.Length;

                        ExecuteInternalNetworkOutput(input[index], w);

                        if(!checkErrorFromLastStep)
                            ForwardError(target[index], w);

                        index++;
                    }
                    if (index < input.Length)
                    {
                        if (checkErrorFromLastStep)
                            ForwardError(target[(index - 1) % input.Length], TimeStepSize - 1);

                        if (i > Ephocs - 60)
                        {
                            Console.WriteLine("index " + index);
                            Console.WriteLine("status " + NodeStatus[TimeStepSize - 1][2][0] +" "+ 
                                NodeStatus[TimeStepSize - 1][2][1] + " " + NodeStatus[TimeStepSize - 1][2][2] + " " +
                                NodeStatus[TimeStepSize - 1][2][3]);
                            Console.WriteLine("exp " + target[(index - 1) % input.Length][0] + " " +
                                                        target[(index - 1) % input.Length][1] + " " +
                                                        target[(index - 1) % input.Length][2] + " " +
                                                        target[(index - 1) % input.Length][3] + " ");
                        }

                        //TODO verificare come gestire input

                        ExecuteBackpropagation(checkErrorFromLastStep);
                    }
                }

                UpdateWeigth(q);
            }
            Console.WriteLine("Total err: " + tot / Ephocs);
           
            
        }

        //public double[] GetNetworkOutput(
        //    double[] input)
        //{

        //    GetANNLayerOutput(Layer, input);

        //    double[] output = new double[NodeLayer[Layer - 1]];
        //    NodeStatus[Layer - 1].CopyTo(output, 0);

        //    return output;
        //}

        public double[] GetNetworkOutput(
            double[][] input)
        {
            int index = 0;
            for (int w = 0; w < TimeStepSize; w++)
            {
                index = index % input.Length;

                ExecuteInternalNetworkOutput(input[index], w);

                index++;
            }
                        
            double[] output = new double[NodeLayer[2]];
                        
            NodeStatus[TimeStepSize - 1][2].CopyTo(output, 0);

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

        //private double GetInnerNetworkMSE(
        //    double[][] target)
        //{
        //    double total = 0.0;
        //    for (int i = 0; i < input.Length; i++)
        //    {
        //        double buf = 0.0;
        //        for (int z = 0; z < NodeLayer[Layer - 1]; z++)
        //        {
        //            double err = NodeStatus[TimeStepSize-1][Layer - 1][z] - target[i][z];
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

        //TEST
        double tot = 0.0;

        /// <summary>
        /// Forward error calculation
        /// </summary>
        private void ForwardError(
            double[] target,
            int step)
        {
            Parallel.For(0, NodeLayer[2], new ParallelOptions { MaxDegreeOfParallelism = Thread },
                z =>
                {
                    Delta[2][z] = ErrorFunc.GetDeltaForwardError(
                                                    NodeStatus[step][2][z],
                                                    Net[step][2][z],
                                                    target[z],
                                                    ActivationFunc[1]);
                });

            //DEBUG
            double errSum = 0.0;
            for (int i = 0; i < NodeLayer[2]; i++)
            {
                double err = NodeStatus[step][2][i] - target[i];
                errSum += err * err;
            }
            tot += errSum / NodeLayer[2];
        }

        /// <summary>
        /// Backpropagation
        /// </summary>
        private void ExecuteBackpropagation(
            bool checkErrorFromLastStep)
        {
            for (int t = TimeStepSize - 1; t >= 0; t--)
            {
                //Delta hidden layer
                int nLayer = 2;

                //Last Weigth Layer
                for (int h = 0; h < WeigthLayer[1]; h++)
                {
                    //int IndexBuffer[][] = (int)Math.Floor((double)(h / NodeLayer[z]));
                    DeltaWeigth[1][h] += Delta[2][IndexBuffer[1][h]] * NodeStatus[t][1][h % NodeLayer[1]];
                }

                Parallel.For(0, NodeLayer[2], new ParallelOptions { MaxDegreeOfParallelism = Thread },
                    j =>
                    {
                        DeltaBias[2][j] += Delta[2][j];
                    });

                //# Initial delta calculation: dL/dz
                Parallel.For(0, NodeLayer[1], new ParallelOptions { MaxDegreeOfParallelism = Thread },
                    j =>
                    {

                        double s = 0.0;
                        for (int h = 0; h < NodeLayer[nLayer]; h++)
                        {
                            int index = (h * NodeLayer[1]) + j;
                            if (!Dropout[1][index])
                            {
                                s += Delta[2][h] * Weigth[1][index];
                            }
                        }

                        Delta[1][j] = s * ActivationFunc[0].GetDerivative(Net[t][1][j]);
                    });


                //Second step
                //# Backpropagation through time (for at most self.bptt_truncate steps)
                //Parallel.For(0, NodeLayer[z], new ParallelOptions { MaxDegreeOfParallelism = Thread },
                //    j =>
                for (int bpStep = t; bpStep > Math.Min(t, Math.Max(0, t - BpttTruncate)); bpStep--)
                {
                    //# Add to gradients at each previous step
                    Parallel.For(0, NodeLayer[1] * NodeLayer[1], new ParallelOptions { MaxDegreeOfParallelism = Thread },
                    h =>
                    {
                        //int bufIndex = (int)Math.Floor((double)(h / NodeLayer[z]));
                        DeltaTime[0][h] += Delta[1][TimeIndexBuffer[0][h]] * NodeStatus[bpStep - 1][1][h % NodeLayer[1]];
                    });

                    for (int j = 0; j < WeigthLayer[0]; j++)
                    {
                        DeltaWeigth[0][j] += Delta[1][IndexBuffer[0][j]] * NodeStatus[bpStep][0][j % NodeLayer[0]];
                    }

                    for (int j = 0; j < NodeLayer[1]; j++)
                    {
                        double s = 0.0;

                        for (int h = 0; h < NodeLayer[1]; h++)
                        {
                            int index = (h * NodeLayer[1]) + j;
                            s += Delta[1][h] * TimeWeigth[0][index];
                        }

                        Delta[1][j] = s * ActivationFunc[0].GetDerivative(Net[bpStep - 1][1][j]);
                    }

                    //delta bias
                    Parallel.For(0, NodeLayer[1], new ParallelOptions { MaxDegreeOfParallelism = Thread },
                        j =>
                        {
                            DeltaBias[1][j] += Delta[1][j];
                        });

                } 
            }
        }

        /// <summary>
        /// Weigth updating
        /// </summary>
        /// <param name="q"></param>
        private void UpdateWeigth(double q)
        {
            for (int z = 0; z < 2; z++)
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

                if (z < DeltaTime.Length)
                {
                    Parallel.For(0, NodeLayer[z + 1] * NodeLayer[z + 1], new ParallelOptions { MaxDegreeOfParallelism = Thread },
                       h =>
                       {
                           double buf = etaQ * DeltaTime[z][h]; //+ (Momentum * OldDeltaWeigth[z][h]);
                           TimeWeigth[z][h] -= buf;
                           //OldDeltaWeigth[z][h] = buf;
                       });
                }
            }

            //Bias update
            for (int z = 1; z < 3; z++)
            {
                double etaQ = EtaQ[z];
                for (int j = 0; j < NodeLayer[z]; j++)
                    Bias[z][j] -= etaQ * DeltaBias[z][j];
            }

            //Turn delta to zero
            for (int z = 0; z < 2; z++)
                Array.Clear(DeltaWeigth[z], 0, DeltaWeigth[z].Length);

            //Turn Delta time to zero
            Array.Clear(DeltaTime[0], 0, DeltaTime[0].Length);

            //Turn bias to zero
            for (int z = 0; z < 3; z++)
                Array.Clear(DeltaBias[z], 0, DeltaBias[z].Length);
        }

        private void ExecuteInternalNetworkOutput(
            double[] input,
            int step)
        {
            //Input Layer
            Array.Copy(input, NodeStatus[step][0], NodeStatus[step][0].Length);

            //output node status
            for (int z = 1; z < 3; z++)
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
                    if (z + 1 < 3 &&
                        step > 0)
                    {
                        int nodeTimeLayer = NodeLayer[z] * j;

                        for (int h = 0; h < NodeLayer[z]; h++)
                        {
                            st += NodeStatus[step - 1][z][h] * TimeWeigth[nLayer][nodeTimeLayer + h];
                        }

                    }
                    Net[step][z][j] = s + st + Bias[z][j];
                    NodeStatus[step][z][j] = ActivationFunc[nLayer].GetResult(Net[step][z][j]);
                });

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
            WeigthLayer = new int[2];
            Weigth = new double[2][];
            Dropout = new bool[2][];
            DeltaWeigth = new double[2][];
            OldDeltaWeigth = new double[2][];
            IndexBuffer = new int[2][];

            TimeWeigth = new double[1][];
            DeltaTime = new double[1][];
            TimeIndexBuffer = new int[2][];

            Bias = new double[3][];
            DeltaBias = new double[3][];
            EtaQ = new double[3];

            Delta = new double[3][];
            NodeStatus = new double[TimeStepSize][][];
            Net = new double[TimeStepSize][][];
            
            for (int i = 0; i < TimeStepSize; i++)
            {
                
                NodeStatus[i] = new double[3][];
                Net[i] = new double[3][];

                for (int j = 0; j < 3; j++)
                {
                    NodeStatus[i][j] = new double[NodeLayer[j]];
                    Net[i][j] = new double[NodeLayer[j]];
                }
            }

            for (int i = 0; i < 3; i++)
            {
                Delta[i] = new double[NodeLayer[i]];

                if (i < 2)
                {
                    WeigthLayer[i] = NodeLayer[i] * NodeLayer[i + 1];
                    Weigth[i] = new double[WeigthLayer[i]];
                    DeltaWeigth[i] = new double[WeigthLayer[i]];
                    OldDeltaWeigth[i] = new double[WeigthLayer[i]];
                    IndexBuffer[i] = new int[WeigthLayer[i]];
                    Dropout[i] = new bool[WeigthLayer[i]];

                    for (int j = 0; j < WeigthLayer[i]; j++)
                    {
                        Weigth[i][j] = 0.01 * Helper.GetRandomGaussian(0.0, 1.0);
                        IndexBuffer[i][j] = (int)Math.Floor((double)(j / NodeLayer[i]));
                        Dropout[i][j] = false;
                    }
                }

                if (i < 1)
                {
                    int dimTimeWeigth = NodeLayer[i + 1] * NodeLayer[i + 1];
                    TimeWeigth[i] = new double[dimTimeWeigth];
                    DeltaTime[i] = new double[dimTimeWeigth];
                    TimeIndexBuffer[i] = new int[dimTimeWeigth];

                    for (int h = 0; h < dimTimeWeigth; h++)
                        TimeIndexBuffer[i][h] = (int)Math.Floor((double)(h / NodeLayer[i + 1]));

                    for (int k = 0; k < dimTimeWeigth; k++)
                        TimeWeigth[i][k] = 0.01 * Helper.GetRandomGaussian(0.0, 1.0);
                }

                Bias[i] = new double[NodeLayer[i]];
                DeltaBias[i] = new double[NodeLayer[i]];

                for (int j = 0; j < NodeLayer[i]; j++)
                    Bias[i][j] = 0.01 * Helper.GetRandomGaussian(0.0, 1.0);
            }
            
            Thread = 2;
        }

        private void SetEtaQ(double q)
        {
            for (int i = 0; i < 3; i++)
                EtaQ[i] = LearningRate[i] * q;
        }

        //TODO throw new Exception
        private void CheckNetworkInputCoherence()
        {

        }

        #endregion
    }
}
