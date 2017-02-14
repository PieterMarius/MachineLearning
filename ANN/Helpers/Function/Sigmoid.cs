using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MachineLearning.Helpers
{
    public sealed class Sigmoid : IFunction
    {
        #region Fields

        public readonly double P;
        public readonly double Lin;

        #endregion

        #region Constructor

        public Sigmoid(
            double p,
            double lin)
        {
            P = p;
            Lin = lin;
        }

        #endregion

        #region Public Methods

        public double GetDerivative(double x)
        {
            double b = GetResult(x);
            return P * b * (1.0 - b) + Lin;
        }

        public double GetResult(double x)
        {
            return 1.0 / (1.0 + Helper.Exp16(-P * x)) + Lin;
        }

        #endregion
    }
}
