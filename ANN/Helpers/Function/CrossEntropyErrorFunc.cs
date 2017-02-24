using System;

namespace MachineLearning.Helpers
{
    public sealed class CrossEntropyErrorFunc : IErrorFunction
    {
        #region Constructor

        public CrossEntropyErrorFunc() { }
        
        #endregion

        #region Public Methods

        public double GetDeltaForwardError(
            double outpuLayer, 
            double outputNet, 
            double target, 
            IFunction function)
        {
            return outpuLayer - target;
        }

        #endregion

    }
}
