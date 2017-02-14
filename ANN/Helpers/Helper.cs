using System;
using System.Collections.Generic;
using System.Threading;

namespace MachineLearning.Helpers
{
    public static class Helper
    {
        #region Private Static Methods

        static readonly Random random = new Random();
        static readonly object syncLock = new object();

        #endregion

        static int seed = Environment.TickCount;

        static readonly ThreadLocal<Random> randomThread =
            new ThreadLocal<Random>(() => new Random(Interlocked.Increment(ref seed)));

        public static double Rand()
        {
            return randomThread.Value.NextDouble();
        }

        public static double GetRandom(double min, double max)
        {
            lock (syncLock)
            {
                return random.NextDouble() * (max - min) + min;
            }
        }

        public static double GetRandomGaussian(
            double mean, 
            double stdDev)
        {
            lock (syncLock)
            {
                double u1 = 1.0 - random.NextDouble(); //uniform(0,1] random doubles
                double u2 = 1.0 - random.NextDouble();
                double randStdNormal = Math.Sqrt(-2.0 * Math.Log(u1)) *
                             Math.Sin(2.0 * Math.PI * u2); //random normal(0,1)

                return mean + (stdDev * stdDev) * randStdNormal;
            }
        }

        public static bool GetRandomBernoulli(double p)
        {
            if (Rand() < p)
                return false;
            else
                return true;
        }

        /// <summary>
        /// Gets random int.
        /// </summary>
        /// <returns>The random.</returns>
        /// <param name="min">Minimum.</param>
        /// <param name="max">Max.</param>
        public static int GetRandom(int min, int max)
        {
            lock (syncLock)
            {
                return random.Next(min, max);
            }
        }

        public static void Shuffle<T>(this IList<T> list)
        {
            int n = list.Count;
            while (n > 1)
            {
                n--;
                int k = random.Next(n + 1);
                T value = list[k];
                list[k] = list[n];
                list[n] = value;
            }
        }

        public static double Exp8(double x)
        {
            double eapprox = (1.0 + x / 256.0);
            eapprox *= eapprox;
            eapprox *= eapprox;
            eapprox *= eapprox;
            eapprox *= eapprox;
            eapprox *= eapprox;
            eapprox *= eapprox;
            eapprox *= eapprox;
            eapprox *= eapprox;
            return eapprox;
        }

        public static double Exp16(double x)
        {
            double eapprox = (1.0 + x * 0.0000152587890625); // 1/65536 
            eapprox *= eapprox;
            eapprox *= eapprox;
            eapprox *= eapprox;
            eapprox *= eapprox;
            eapprox *= eapprox;
            eapprox *= eapprox;
            eapprox *= eapprox;
            eapprox *= eapprox;
            eapprox *= eapprox;
            eapprox *= eapprox;
            eapprox *= eapprox;
            eapprox *= eapprox;
            eapprox *= eapprox;
            eapprox *= eapprox;
            eapprox *= eapprox;
            eapprox *= eapprox;
            return eapprox;
        }

        public static double Exp(double x)
        {
            var tmp = (long)(1512775 * x + 1072632447);
            return BitConverter.Int64BitsToDouble(tmp << 32);
        }

        public static double Sigmoid(
            double p,
            double lin,
            double x)
        {
            return 1.0 / (1.0 + Exp16(-p * x)) + lin;
        }

        public static double Tanh(
            double k,
            double p,
            double lin,
            double x)
        {
            return k * Math.Tanh(p * x) + lin;
        }

        public static double SoftPlus(double x)
        {
            return Math.Log(1.0 + Math.Exp(x));
        }

        public static double ReLU(double x)
        {
            return (x < 0.0) ? 0.0 : x;
        }

        public static double DerivativeSigmoid(
            double p,
            double lin,
            double x)
        {
            double b = Sigmoid(p, lin, x);
            return p * b * (1.0 - b) + lin;
        }

        public static double DerivativeTanh(
            double k,
            double p,
            double lin,
            double x)
        {
            double b = 1.0 / Math.Cosh(p * x);
            return p * k * b * b + lin;
        }

        public static double DerivativeSoftPlus(double x)
        {
            return Sigmoid(1.0, 0.0, x);
        }
    }
}
