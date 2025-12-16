#include "Source.h"
#include <numeric>
#include <cmath>
#include <fstream>
#include <algorithm>
#include <stdexcept>
#include <boost/math/distributions/normal.hpp>

Sample::Sample() : sample(), mean(0.0), stdDev(0.0) {
}

Sample::Sample(const std::vector<double>& sample) : sample(sample) {
    this->mean = calculateMean();
    this->stdDev = calculateStandardDeviation();
}

double Sample::calculateMean() const {
    const int n = static_cast<int>(this->sample.size());
    if (n == 0) return 0.0;
    double sum = std::accumulate(this->sample.begin(), this->sample.end(), 0.0);
    return sum / static_cast<double>(n);
}

double Sample::calculateStandardDeviation() const {
    const int n = static_cast<int>(this->sample.size());
    if (n <= 1) return 0.0;
    double variance = 0.0;
    for (int i = 0; i < n; ++i) {
        double diff = this->sample[static_cast<std::size_t>(i)] - this->mean;
        variance += diff * diff;
    }
    variance /= static_cast<double>(n - 1);
    return std::sqrt(variance);
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

double normPPF(double alpha) {
    if (alpha <= 0.0 || alpha >= 1.0) return 0.0;
    boost::math::normal_distribution<> d(0.0, 1.0);
    return quantile(d, alpha);
}

Sample generateSampleNormal(double mean, double stdDev, int n, bool shift) {
    if (stdDev <= 0.0) stdDev = 0.1;
    std::mt19937& gen = globalGenerator();
    if (n <= 0) n = generateIntNumberFromAToB(gen, 20, 40);
    double delta = shift ? 0.05 : 0.0;
    std::vector<double> sample(static_cast<std::size_t>(n));
    std::normal_distribution<double> dist(mean + delta, stdDev);
    for (double& el : sample) el = dist(gen);
    return Sample(sample);
}

double calculateWilcoxonStat(const Sample& sample1, const Sample& sample2) {
    const std::vector<double>& x = sample1.getSample();
    const std::vector<double>& y = sample2.getSample();
    int m = static_cast<int>(x.size());
    int n = static_cast<int>(y.size());
    if (m <= 0 || n <= 0) throw std::invalid_argument("Выборки не должны быть пустыми");

    struct Item { double value; int group; int idx; };
    std::vector<Item> united;
    united.reserve(static_cast<std::size_t>(m + n));

    for (int i = 0; i < m; ++i) united.push_back(Item{ x[static_cast<std::size_t>(i)], 0, i });
    for (int j = 0; j < n; ++j) united.push_back(Item{ y[static_cast<std::size_t>(j)], 1, j });

    std::sort(united.begin(), united.end(), [](const Item& a, const Item& b) {
        if (a.value < b.value) return true;
        if (a.value > b.value) return false;
        if (a.group < b.group) return true;
        if (a.group > b.group) return false;
        return a.idx < b.idx;
        });

    int smallGroupId = (m <= n) ? 0 : 1;
    int N = m + n;

    double W = 0.0;
    for (int i = 0; i < N; ++i) {
        int rank = i + 1;
        if (united[static_cast<std::size_t>(i)].group == smallGroupId) W += static_cast<double>(rank);
    }
    return W;
}

std::pair<double, double> calculateApproximately(int m, int n, double alpha, bool oneSided) {
    std::pair<double, double> wCrit{ 0.0, 0.0 };
    double mu = (static_cast<double>(m) * (m + n + 1)) / 2.0;
    double sigma = std::sqrt(static_cast<double>(m) * n * (m + n + 1) / 12.0);

    if (oneSided) {
        double z = normPPF(alpha);
        wCrit.first = mu + z * sigma;
        wCrit.second = 0.0;
        return wCrit;
    }

    double zLow = normPPF(alpha / 2.0);
    double zHigh = normPPF(1.0 - alpha / 2.0);
    wCrit.first = mu + zLow * sigma;
    wCrit.second = mu + zHigh * sigma;
    return wCrit;
}

long double binomial_coefficient(int n, int k) {
    if (k < 0 || k > n) return 0.0L;
    if (k == 0 || k == n) return 1.0L;
    if (k > n - k) k = n - k;
    long double result = 1.0L;
    for (int i = 1; i <= k; ++i) {
        result *= static_cast<long double>(n - k + i);
        result /= static_cast<long double>(i);
    }
    return result;
}

void buildExactWilcoxonDistribution(int m, int n, int& minW, int& maxW, int& maxU, std::vector<long double>& pU, std::vector<long double>& cdfU) {
    if (m <= 0 || n <= 0) throw std::invalid_argument("m и n должны быть положительными");

    int N = m + n;
    minW = m * (m + 1) / 2;
    maxW = m * (2 * N - m + 1) / 2;
    maxU = m * n;

    std::vector<std::vector<long double>> dp(m + 1, std::vector<long double>(maxW + 1, 0.0L));
    dp[0][0] = 1.0L;

    for (int r = 1; r <= N; ++r) {
        int max_i = std::min(r, m);
        for (int i = max_i; i >= 1; --i) {
            for (int s = maxW; s >= r; --s) {
                if (dp[i - 1][s - r] > 0.0L) dp[i][s] += dp[i - 1][s - r];
            }
        }
    }

    long double total = binomial_coefficient(N, m);

    pU.assign(maxU + 1, 0.0L);
    for (int W = minW; W <= maxW; ++W) {
        int U = W - minW;
        long double ways = dp[m][W];
        if (U >= 0 && U <= maxU && ways > 0.0L) pU[U] = ways / total;
    }

    cdfU.assign(maxU + 1, 0.0L);
    long double cumulative = 0.0L;
    for (int u = 0; u <= maxU; ++u) {
        cumulative += pU[u];
        cdfU[u] = cumulative;
    }
}

int findQuantileU(const std::vector<long double>& cdfU, int maxU, long double alpha) {
    if (alpha <= 0.0L) return 0;
    if (alpha >= 1.0L) return maxU;
    for (int u = 0; u <= maxU; ++u) if (cdfU[static_cast<std::size_t>(u)] >= alpha) return u;
    return maxU;
}

int convertUtoW(int minW, int U_quantile) {
    return minW + U_quantile;
}

std::pair<double, double> calculatePrecisely(int m, int n, double alpha, bool oneSided) {
    std::pair<double, double> wCrit{ 0.0, 0.0 };

    int minW = 0;
    int maxW = 0;
    int maxU = 0;
    std::vector<long double> pU;
    std::vector<long double> cdfU;

    buildExactWilcoxonDistribution(m, n, minW, maxW, maxU, pU, cdfU);

    if (oneSided) {
        int U = findQuantileU(cdfU, maxU, static_cast<long double>(alpha));
        wCrit.first = static_cast<double>(convertUtoW(minW, U));
        wCrit.second = 0.0;
        return wCrit;
    }

    long double tail = static_cast<long double>(alpha) / 2.0L;
    int Uleft = findQuantileU(cdfU, maxU, tail);
    int Uright = maxU - Uleft;
    wCrit.first = static_cast<double>(convertUtoW(minW, Uleft));
    wCrit.second = static_cast<double>(convertUtoW(minW, Uright));
    return wCrit;
}

std::pair<double, double> calculateWCrit(const Sample& sample1, const Sample& sample2, double alpha, bool oneSided) {
    int m1 = sample1.getSampleSize();
    int n1 = sample2.getSampleSize();
    if (m1 <= 0 || n1 <= 0) throw std::invalid_argument("Размеры выборок должны быть положительными");

    int m = (m1 <= n1) ? m1 : n1;
    int n = (m1 <= n1) ? n1 : m1;

    if (m + n <= 40) return calculatePrecisely(m, n, alpha, oneSided);
    return calculateApproximately(m, n, alpha, oneSided);
}

WilcoxonDecision wilcoxonTwoSidedDecision(const Sample& sample1, const Sample& sample2, double alpha) {
    double Wobs = calculateWilcoxonStat(sample1, sample2);
    std::pair<double, double> wCrit = calculateWCrit(sample1, sample2, alpha, false);
    if (Wobs <= wCrit.first || Wobs >= wCrit.second) return WilcoxonDecision::RejectH0;
    return WilcoxonDecision::AcceptH0;
}

void generateTwoSidedWilcoxonInp(double alpha) {
    std::mt19937& gen = globalGenerator();
    int m = generateIntNumberFromAToB(gen, 8, 20);
    int n = generateIntNumberFromAToB(gen, 8, 20);
    bool shiftSecond = (generateIntNumberFromAToB(gen, 0, 1) != 0);

    Sample sample1 = generateSampleNormal(0.0, 1.0, m, false);
    Sample sample2 = generateSampleNormal(0.0, 1.0, n, shiftSecond);

    std::ofstream fout("Inp/twosidedwilcoxon.inp");
    if (!fout.is_open()) throw std::runtime_error("Не удалось открыть Inp/twosidedwilcoxon.inp");

    fout.setf(std::ios::fixed);
    fout.precision(6);

    fout << alpha << '\n';
    fout << m << ' ' << n << '\n';
    fout << (shiftSecond ? 1 : 0) << '\n';

    const std::vector<double>& x = sample1.getSample();
    for (int i = 0; i < m; ++i) {
        fout << x[static_cast<std::size_t>(i)];
        if (i + 1 < m) fout << ' ';
    }
    fout << '\n';

    const std::vector<double>& y = sample2.getSample();
    for (int j = 0; j < n; ++j) {
        fout << y[static_cast<std::size_t>(j)];
        if (j + 1 < n) fout << ' ';
    }
    fout << '\n';
}

void runTwoSidedWilcoxonFromFile() {
    std::ifstream fin("Inp/twosidedwilcoxon.inp");
    if (!fin.is_open()) throw std::runtime_error("Не удалось открыть Inp/twosidedwilcoxon.inp");

    double alpha = 0.0;
    int m_inp = 0;
    int n_inp = 0;
    int shiftFlagInt = 0;

    if (!(fin >> alpha)) throw std::runtime_error("Ошибка чтения alpha");
    if (!(fin >> m_inp >> n_inp)) throw std::runtime_error("Ошибка чтения m n");
    if (!(fin >> shiftFlagInt)) throw std::runtime_error("Ошибка чтения shiftFlag");

    bool shiftSecond = (shiftFlagInt != 0);
    if (m_inp <= 0 || n_inp <= 0) throw std::runtime_error("Некорректные размеры выборок");

    std::vector<double> x(static_cast<std::size_t>(m_inp));
    std::vector<double> y(static_cast<std::size_t>(n_inp));

    for (int i = 0; i < m_inp; ++i) if (!(fin >> x[static_cast<std::size_t>(i)])) throw std::runtime_error("Ошибка чтения выборки 1");
    for (int j = 0; j < n_inp; ++j) if (!(fin >> y[static_cast<std::size_t>(j)])) throw std::runtime_error("Ошибка чтения выборки 2");

    Sample sample1(x);
    Sample sample2(y);

    int m1 = sample1.getSampleSize();
    int n1 = sample2.getSampleSize();
    int N = m1 + n1;

    int m_small = (m1 <= n1) ? m1 : n1;
    int n_large = (m1 <= n1) ? n1 : m1;
    int smallGroupId = (m1 <= n1) ? 0 : 1;

    bool useExact = (m_small + n_large <= 40);

    double W_obs = calculateWilcoxonStat(sample1, sample2);
    std::pair<double, double> wCrit = calculateWCrit(sample1, sample2, alpha, false);

    const std::vector<double>& xs = sample1.getSample();
    const std::vector<double>& ys = sample2.getSample();

    struct Item { double value; int group; int idx; };
    std::vector<Item> united;
    united.reserve(static_cast<std::size_t>(N));
    for (int i = 0; i < m1; ++i) united.push_back(Item{ xs[static_cast<std::size_t>(i)], 0, i });
    for (int j = 0; j < n1; ++j) united.push_back(Item{ ys[static_cast<std::size_t>(j)], 1, j });

    std::sort(united.begin(), united.end(), [](const Item& a, const Item& b) {
        if (a.value < b.value) return true;
        if (a.value > b.value) return false;
        if (a.group < b.group) return true;
        if (a.group > b.group) return false;
        return a.idx < b.idx;
        });

    std::ofstream fout("Out/twosidedwilcoxon.out");
    if (!fout.is_open()) throw std::runtime_error("Не удалось открыть Out/twosidedwilcoxon.out");

    fout.setf(std::ios::fixed);
    fout.precision(6);

    fout << "================ Двусторонний критерий Уилкоксона (Mann–Whitney) ================\n\n";
    fout << "alpha = " << alpha << "\n";
    fout << "m1 = " << m1 << ", n1 = " << n1 << ", N = " << N << "\n";
    fout << "Меньшая выборка: " << (smallGroupId + 1) << " (m = " << m_small << ")\n\n";

    fout << "Сдвиг среднего второй выборки на +0.05: " << (shiftSecond ? "ДА" : "НЕТ") << "\n\n";

    fout << "Выборка 1:\n";
    for (int i = 0; i < m1; ++i) fout << "x[" << (i + 1) << "] = " << xs[static_cast<std::size_t>(i)] << "\n";
    fout << "\n";

    fout << "Выборка 2:\n";
    for (int j = 0; j < n1; ++j) fout << "y[" << (j + 1) << "] = " << ys[static_cast<std::size_t>(j)] << "\n";
    fout << "\n";

    fout << "mean1 = " << sample1.getMean() << ", std1 = " << sample1.getStdDev() << "\n";
    fout << "mean2 = " << sample2.getMean() << ", std2 = " << sample2.getStdDev() << "\n\n";

    fout << "Ранги (ties игнорируются): индекс, значение, выборка, ранг\n";
    for (int i = 0; i < N; ++i) fout << (i + 1) << " " << united[static_cast<std::size_t>(i)].value << " " << (united[static_cast<std::size_t>(i)].group + 1) << " " << (i + 1) << "\n";
    fout << "\n";

    fout << "W_obs = " << W_obs << "\n";
    fout << "Режим критических значений: " << (useExact ? "ТОЧНО" : "ПРИБЛИЖЁННО") << "\n";
    fout << "W_lower = " << wCrit.first << "\n";
    fout << "W_upper = " << wCrit.second << "\n\n";

    fout << "H0: распределения совпадают\n";
    fout << "H1: распределения различаются\n\n";

    bool reject = (W_obs <= wCrit.first) || (W_obs >= wCrit.second);
    if (reject) fout << "Решение: H0 отвергается\n";
    else fout << "Решение: H0 не отвергается\n";
}
