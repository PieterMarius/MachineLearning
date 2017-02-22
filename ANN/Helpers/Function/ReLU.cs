using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MachineLearning.Helpers
{
    public class ReLU : IFunction
    {
        #region Constructor

        public ReLU() {}

        #endregion

        #region Public Methods

        public double GetDerivative(double x)
        {
            return (x > 0) ? 1 : 0;
        }

        public double GetResult(double x)
        {
            return (x < 0) ? 0 : x;
        }

        #endregion
    }
}
