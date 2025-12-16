#define _USE_MATH_DEFINES

#include "Source.h"

#include <numeric>
#include <cmath>
#include <fstream>
#include <limits>

#include <boost/math/distributions/fisher_f.hpp>
#include <boost/math/distributions/normal.hpp>
#include <boost/math/special_functions/beta.hpp>

Sample::Sample(const std::vector<double>& sample) : sample(sample) {
    this->mean = calculateMean();
    this->stdDev = calculateStandardDeviation();
}

double Sample::calculateMean() const {
    double sum = std::accumulate(this->sample.begin(), this->sample.end(), 0.0);
    const int n = static_cast<int>(this->sample.size());
    if (n == 0) {
        return 0.0;
    }
    return sum / static_cast<double>(n);
}

double Sample::calculateStandardDeviation() const {
    const int n = static_cast<int>(this->sample.size());
    if (n <= 1) {
        return 0.0;
    }

    double standartDeviation = 0.0;
    for (int i = 0; i < n; ++i) {
        double diff = this->sample[i] - this->mean;
        standartDeviation += diff * diff;
    }
    standartDeviation /= static_cast<double>(n - 1);
    standartDeviation = std::sqrt(standartDeviation);
    return standartDeviation;
}

double Sample::getMean() const {
    return this->mean;
}

double Sample::getStdDev() const {
    return this->stdDev;
}

int Sample::getSampleSize() const {
    return static_cast<int>(this->sample.size());
}

const std::vector<double>& Sample::getSample() const {
    return this->sample;
}

std::mt19937& globalGenerator() {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    return gen;
}

int generateIntNumberFromAToB(std::mt19937& generator, int a, int b) {
    std::uniform_int_distribution<int> uni(a, b);
    return uni(generator);
}

Sample generateSample(double mean, double stdDev) {
    if (stdDev <= 0.0) {
        stdDev = 0.1;
    }

    std::mt19937& gen = globalGenerator();
    const int n = generateIntNumberFromAToB(gen);

    std::vector<double> sample(n);

    std::normal_distribution<double> dist(mean, stdDev);
    for (double& el : sample) {
        el = dist(gen);
    }
    std::sort(sample.begin(), sample.end());
    return Sample(sample);
}

double calculateSquaredDeviations(const Sample& sample) {
    double res = 0;
    for (double el : sample.getSample()) {
        double diff = el - sample.getMean();
        res += diff * diff;
    }
    return res;
}

double standardNormalCdf(double x) {
    static const boost::math::normal_distribution<double> nd(0.0, 1.0);
    return boost::math::cdf(nd, x);
}

double standardNormalQuantile(double p) {
    if (p <= 0.0) {
        return -std::numeric_limits<double>::infinity();
    }
    if (p >= 1.0) {
        return std::numeric_limits<double>::infinity();
    }
    static const boost::math::normal_distribution<double> nd(0.0, 1.0);
    return boost::math::quantile(nd, p);
}

double standardNormalPdf(double x) {
    static const boost::math::normal_distribution<double> nd(0.0, 1.0);
    return boost::math::pdf(nd, x);
}

std::vector<double> computeStandardNormalOrderStatMeans(int n) {
    if (n <= 0) {
        throw std::invalid_argument("computeStandardNormalOrderStatMeans: n must be > 0");
    }

    std::vector<double> expectations(n);

    for (int i = 1; i <= n; ++i) {
        double p = static_cast<double>(i) / (n + 1.0);
        double u_p = standardNormalQuantile(p);
        double f_u = standardNormalPdf(u_p);

        if (f_u <= 0.0) {
            expectations[i - 1] = u_p;
            continue;
        }

        double f_prime = -u_p * f_u;
        double e = u_p;

        double corr1 = (p * (1.0 - p)) / (2.0 * (n + 2.0)) * (f_prime / (f_u * f_u));
        e += corr1;

        expectations[i - 1] = e;
    }

    return expectations;
}

std::vector<std::vector<double>> computeStandardNormalOrderStatCovariance(int n) {
    if (n <= 0) {
        throw std::invalid_argument("computeStandardNormalOrderStatCovariance: n must be > 0");
    }

    std::vector<std::vector<double>> cov(n, std::vector<double>(n, 0.0));

    std::vector<double> p(n);
    std::vector<double> u(n);
    std::vector<double> f(n);

    for (int i = 0; i < n; ++i) {
        p[i] = static_cast<double>(i + 1) / (n + 1.0);
        u[i] = standardNormalQuantile(p[i]);
        f[i] = standardNormalPdf(u[i]);

        if (f[i] <= 0.0) {
            f[i] = 1e-16;
        }
    }

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (i == j) {
                cov[i][j] = p[i] * (1.0 - p[i]) / ((n + 2.0) * f[i] * f[i]);
            }
            else {
                double p_min = (p[i] < p[j]) ? p[i] : p[j];
                double p_max = (p[i] > p[j]) ? p[i] : p[j];
                cov[i][j] = (p_min * (1.0 - p_max)) / ((n + 2.0) * f[i] * f[j]);
            }
        }
    }

    return cov;
}

std::vector<double> solveLinearSystemGaussJordan(std::vector<std::vector<double>> V, std::vector<double> b) {
    int n = static_cast<int>(V.size());
    if (n == 0) {
        throw std::invalid_argument("solveLinearSystemGaussJordan: matrix size is zero");
    }
    if (static_cast<int>(b.size()) != n) {
        throw std::invalid_argument("solveLinearSystemGaussJordan: size mismatch");
    }

    for (int i = 0; i < n; ++i) {
        double pivot = V[i][i];
        if (std::fabs(pivot) < 1e-14) {
            throw std::runtime_error("solveLinearSystemGaussJordan: pivot is too small");
        }

        for (int j = i; j < n; ++j) {
            V[i][j] /= pivot;
        }
        b[i] /= pivot;

        for (int k = 0; k < n; ++k) {
            if (k == i) continue;
            double factor = V[k][i];
            for (int j = i; j < n; ++j) {
                V[k][j] -= factor * V[i][j];
            }
            b[k] -= factor * b[i];
        }
    }

    return b;
}

std::vector<double> computeShapiroWilkAiCoefficients(int n) {
    if (n <= 0) {
        throw std::invalid_argument("computeShapiroWilkAiCoefficients: n must be > 0");
    }

    std::vector<double> orderStatMeans = computeStandardNormalOrderStatMeans(n);
    std::vector<std::vector<double>> cov = computeStandardNormalOrderStatCovariance(n);
    std::vector<double> y = solveLinearSystemGaussJordan(cov, orderStatMeans);

    double norm2 = 0.0;
    for (double val : y) {
        norm2 += val * val;
    }
    if (norm2 <= 0.0) {
        throw std::runtime_error("computeShapiroWilkAiCoefficients: non-positive norm");
    }

    double norm = std::sqrt(norm2);
    std::vector<double> a(n);
    for (int i = 0; i < n; ++i) {
        a[i] = y[i] / norm;
    }

    return a;
}

double calculateBCoef(const std::vector<double>& a_i, const Sample& sample) {
    double result = 0;
    for (std::size_t i = 0; i < sample.getSampleSize(); i++) {
        result += a_i[i] * sample.getSample()[i];
    }
    return result;
}

double calculateWilkStat(const double b, const double sSquared) {
    return (b * b) / sSquared;
}

double calculateWAlpha(int n) {
    static const double Wcrit05[51] = {
        0.0, 0.0, 0.0, 0.767, 0.748, 0.762, 0.788, 0.803, 0.818, 0.829,
        0.842, 0.850, 0.859, 0.866, 0.874, 0.881, 0.887, 0.892, 0.897, 0.901,
        0.905, 0.908, 0.911, 0.914, 0.916, 0.918, 0.920, 0.923, 0.924, 0.926,
        0.927, 0.929, 0.930, 0.931, 0.933, 0.934, 0.935, 0.936, 0.938, 0.939,
        0.940, 0.941, 0.942, 0.943, 0.944, 0.945, 0.945, 0.946, 0.947, 0.947,
        0.947
    };

    if (n < 3 || n > 50) {
        throw std::invalid_argument("Критические значения W по Агамирову заданы только для 3 <= n <= 50");
    }

    return Wcrit05[n];
}

Sample generateInputFileWithNewSample(const std::string& filePath, double mean, double stdDev) {
    Sample sample = generateSample(mean, stdDev);

    std::ofstream fout(filePath, std::ios::trunc);
    if (!fout.is_open()) {
        throw std::runtime_error("Не удалось открыть файл для записи: " + filePath);
    }

    const int n = sample.getSampleSize();
    const std::vector<double>& data = sample.getSample();

    fout << n << '\n';
    for (int i = 0; i < n; ++i) {
        fout << data[i];
        if (i + 1 < n) {
            fout << ' ';
        }
    }
    fout << '\n';

    return sample;
}

void writeShapiroWilkResultToFile(const std::string& filePath, const Sample& sample, const std::vector<double>& a_i, double alpha) {
    std::ofstream fout(filePath, std::ios::trunc);
    if (!fout.is_open()) {
        throw std::runtime_error("Не удалось открыть файл для записи: " + filePath);
    }

    const int n = sample.getSampleSize();
    const double mean = sample.getMean();
    const double stdDev = sample.getStdDev();

    double sSquared = calculateSquaredDeviations(sample);
    double b = calculateBCoef(a_i, sample);
    double W_observed = calculateWilkStat(b, sSquared);
    double W_crit = calculateWAlpha(n);

    fout << "Критерий Шапиро-Уилка\n";
    fout << "Уровень значимости alpha = " << alpha << "\n\n";

    fout << "Размер выборки n = " << n << "\n";
    fout << "Выборочное среднее x̄ = " << mean << "\n";
    fout << "Выборочное СКО s = " << stdDev << "\n\n";

    fout << "Сумма квадратов отклонений s^2 = " << sSquared << "\n";
    fout << "Коэффициент b = " << b << "\n";
    fout << "Наблюдаемое значение статистики Wнабл = " << W_observed << "\n";
    fout << "Критическое значение Wкр = " << W_crit << "\n\n";

    if (W_observed >= W_crit) {
        fout << "Вывод: Wнабл > Wкр, нет оснований отвергать нулевую гипотезу.\n";
        fout << "Распределение можно считать нормальным.\n";
    }
    else {
        fout << "Вывод: Wнабл < Wкр, нулевая гипотеза отвергается.\n";
        fout << "Распределение нельзя считать нормальным.\n";
    }
}
