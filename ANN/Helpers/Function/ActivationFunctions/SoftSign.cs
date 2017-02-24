using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MachineLearning.Helpers
{
    public sealed class SoftSign : IFunction
    {

        #region Constructor

        public SoftSign() { }

        #endregion

        #region Public Methods

        public double GetDerivative(double x)
        {
            double b = 1.0 + Math.Abs(x);
            return 1.0 / (b * b);
        }

        public double GetResult(double x)
        {
            return x / (1.0 + Math.Abs(x));
        }

        #endregion
    }
}
