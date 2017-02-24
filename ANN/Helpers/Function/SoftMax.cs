namespace MachineLearning.Helpers.Function
{
    public sealed class SoftMax : IFunction, IExponentialFunction
    {
        private double ExpSum;

        #region Constructor

        public SoftMax()
        { }

        #endregion

        #region IFunction methods

        public double GetDerivative(double x)
        {
            return x;
        }

        public double GetResult(double x)
        {
            return x;
        }

        #endregion

        #region IExponentialFunction

        public void SetExponentialSum(double[] x)
        {
            ExpSum = 0.0;
            for (int i = 0; i < x.Length; i++)
                ExpSum += Helper.Exp16(x[i]);
        }

        public double GetExponential(double x)
        {
            return Helper.Exp16(x) / ExpSum;
        }

        #endregion
    }
}
