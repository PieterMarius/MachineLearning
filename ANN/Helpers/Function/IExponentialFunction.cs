using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MachineLearning.Helpers
{
    public interface IExponentialFunction
    {
        void SetExponentialSum(double[] x);
        double GetExponential(double x);
    }
}
