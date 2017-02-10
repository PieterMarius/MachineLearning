using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MachineLearning.Helpers
{
    public interface IFunction
    {
        double GetResult(double x);
        double GetDerivative(double x);
    }
}
