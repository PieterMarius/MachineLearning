
namespace MachineLearning.Helpers
{
    public interface IFunction
    {
        double GetResult(double x);
        double GetDerivative(double x);
    }
}
