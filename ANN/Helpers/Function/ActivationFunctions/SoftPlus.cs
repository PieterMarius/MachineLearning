using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MachineLearning.Helpers.Function
{
    public sealed class SoftPlus : IFunction
    {
        #region Constructor

        public SoftPlus() { }

        #endregion

        #region Public Methods

        public double GetDerivative(double x)
        {
            return Math.Log(1 + Helper.Exp16(x));
        }

        public double GetResult(double x)
        {
            return 1.0 / (1.0 + Helper.Exp16(-x));
        }

        #endregion
    }
}
