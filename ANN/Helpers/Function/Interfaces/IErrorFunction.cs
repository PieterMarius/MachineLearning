namespace MachineLearning.Helpers
{
    public interface IErrorFunction
    {
        double GetDeltaForwardError(
            double outpuLayer, 
            double outputNet, 
            double target, 
            IFunction function);
    }
}