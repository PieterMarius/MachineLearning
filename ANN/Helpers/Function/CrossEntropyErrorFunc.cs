using System;

namespace MachineLearning.Helpers.Function
{
    public sealed class CrossEntropyErrorFunc : IErrorFunction
    {
        #region Constructor

        public CrossEntropyErrorFunc() { }
        
        #endregion

        #region Public Methods

        public double GetDeltaForwardError(double outpuLayer, double outputNet, double target, IFunction function)
        {
            throw new NotImplementedException();
        }

        #endregion

    }
}
