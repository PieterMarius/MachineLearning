using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MachineLearning.Helpers.Function
{
    public sealed class SoftMax : IFunction
    {
        #region Constructor

        public SoftMax()
        { }

        #endregion

        #region IFunction methods

        public double GetDerivative(double x)
        {
            throw new NotImplementedException();
        }

        public double GetResult(double x)
        {
            throw new NotImplementedException();
        }

        #endregion
    }
}
