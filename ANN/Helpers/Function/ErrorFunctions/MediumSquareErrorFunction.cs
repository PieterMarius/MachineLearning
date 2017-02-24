
namespace MachineLearning.Helpers
{
    public sealed class MediumSquareErrorFunction : IErrorFunction
    {
        #region Constructor

        public MediumSquareErrorFunction() { }

        #endregion

        #region Public Methods

        public double GetDeltaForwardError(
            double outpuLayer, 
            double outputNet, 
            double target,
            IFunction function)
        {
            double err = outpuLayer - target;
            return err * function.GetDerivative(outputNet);
        }

        #endregion

    }
}
