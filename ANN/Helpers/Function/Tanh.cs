using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MachineLearning.Helpers
{
    public class Tanh: IFunction
    {
        #region Fields

        public readonly double P;
        public readonly double K;
        public readonly double Lin;

        #endregion

        #region Constructor

        public Tanh(
            double p,
            double k,
            double lin)
        {
            P = p;
            K = k;
            Lin = lin;
        }

        #endregion

        #region Public Methods

        public double GetDerivative(double x)
        {
            double b = 1.0 / Math.Cosh(P * x);
            return P * K * b * b + Lin;
        }

        public double GetResult(double x)
        {
            return K * Math.Tanh(P * x) + Lin;
        }

        #endregion
    }
}
