#include "chebyshev.cuh"

std::vector<double> Chebyshev::getCoefficients(std::function<double(double)> func, double a, double b, uint32_t degree) {
    if (!degree) {
        throw std::invalid_argument("The degree of approximation cannot be zero");
    }
    // the number of coefficients to be generated should be degree+1 as zero is also included
    size_t coeffTotal{degree+1};
    double bMinusA = 0.5 * (b - a);
    double bPlusA  = 0.5 * (b + a);
    double PiByDeg = M_PI / static_cast<double>(coeffTotal);
    std::vector<double> functionPoints(coeffTotal);
    for (size_t i = 0; i < coeffTotal; ++i)
        functionPoints[i] = func(std::cos(PiByDeg * (i + 0.5)) * bMinusA + bPlusA);

    double multFactor = 2.0 / static_cast<double>(coeffTotal);
    std::vector<double> coefficients(coeffTotal);
    for (size_t i = 0; i < coeffTotal; ++i) {
        for (size_t j = 0; j < coeffTotal; ++j)
            coefficients[i] += functionPoints[j] * std::cos(PiByDeg * i * (j + 0.5));
        coefficients[i] *= multFactor;
    }
    return coefficients;
}

Ciphertext<Scheme::CKKS> Chebyshev::evalSeriesLinear(const Ciphertext<Scheme::CKKS> x,
    const std::vector<double>& coefficients, double a, double b){
    size_t k = coefficients.size() - 1;
    if (k == 0) {
        throw std::invalid_argument("The coefficients vector must contain at least 2 elements (c0 and c1).");
    }

    // 区间映射：将x从[a,b]映射到[-1,1]，得到y（T1(y) = y）
    std::vector<Ciphertext<Scheme::CKKS>> T(k);  // T[0]对应T1，T[1]对应T2，...，T[k-1]对应Tk
    if (std::abs(a + 1) < 1e-10 && std::abs(b - 1) < 1e-10) {
        // 若原区间已是[-1,1]，直接令y = x（T1 = x）
        T[0] = x;
    } else {
        // 一般情况：y = 2*(x - a)/(b - a) - 1 = (2/(b-a))*x - (2a/(b-a) + 1)
        Plaintext<Scheme::CKKS> alpha(ctx_);
        encoder_.encode(alpha, std::vector<double>(slot_count, 2.0 / (b - a)), scale_);
        Plaintext<Scheme::CKKS> beta(ctx_);
        encoder_.encode(beta, std::vector<double>(slot_count, 2.0 * a / (b - a) + 1.0), scale_);
        T[0] = x;
        evaluator_.multiply(T[0], alpha);
        evaluator_.sub(T[0], beta);
    }

    // 缓存y的副本（用于奇数次多项式递推）
    Ciphertext<Scheme::CKKS> y = T[0];

    // 递推计算T2到Tk（切比雪夫多项式）
    for (size_t i = 2; i <= k; ++i) {  // i是多项式次数（T2到Tk）
        if ((i & (i - 1)) == 0) {
            // 情况1：i是2的幂次
            // 公式：T_{2m}(y) = 2*T_m(y)^2 - 1（这里i=2m → m=i/2）
            size_t m = i / 2;
            // Tm(y)的平方（T[m-1]是Tm）
            Ciphertext<Scheme::CKKS> square; 
            evaluator_.square(T[m - 1], square);
            // 2*Tm(y)^2
            evaluator_.add(square, square, T[i - 1]);
            Plaintext<Scheme::CKKS> pt(ctx_);
            // 减1，得到Ti(y)
            encoder_.encode(pt, std::vector<double>(slot_count, 1.0), scale_);
            evaluator_.sub(T[i - 1], pt);
        } else if (i % 2 == 1) {
            // 情况2：i是奇数（非2的幂次，如3,5...）
            // 公式：T_{2m+1}(y) = 2*Tm(y)*T_{m+1}(y) - y（这里i=2m+1 → m=(i-1)/2）
            size_t m = (i - 1) / 2;
            // Tm(y) * T_{m+1}(y)
            Ciphertext<Scheme::CKKS> prod;
            evaluator_.multiply(T[m - 1], T[m], prod);
            evaluator_.add(prod, prod, T[i - 1]);
            evaluator_.sub(T[i - 1], y);
        } else {
            // 情况3：i是偶数（非2的幂次，如6,10...）
            // 公式：T_i(y) = 2*T_{i/2}(y)^2 - 1（同2的幂次逻辑）
            size_t m = i / 2;
            Ciphertext<Scheme::CKKS> square; 
            evaluator_.square(T[m - 1], square);
            evaluator_.add(square, square, T[i - 1]);
            Plaintext<Scheme::CKKS> pt(ctx_);
            encoder_.encode(pt, std::vector<double>(slot_count, 1.0), scale_);
            evaluator_.sub(T[i - 1], pt);
        }
    }

    // 组合系数与多项式，计算最终结果
    // 先处理最高次项：ck * Tk(y)
    Ciphertext<Scheme::CKKS> result;
    Plaintext<Scheme::CKKS> pt(ctx_);
    encoder_.encode(pt, std::vector<double>(slot_count, coefficients[k]), scale_);
    evaluator_.multiply(T[k - 1], pt, result);

    // 累加低次项：c1*T1(y) + c2*T2(y) + ... + c(k-1)*T(k-1)(y)
    for (size_t i = 0; i < k - 1; ++i) {
        if (coefficients[i + 1] != 0) {  // 跳过系数为0的项（优化）
            Ciphertext<Scheme::CKKS> term;
            // ci+1 * T(i+1)(y)
            encoder_.encode(pt, std::vector<double>(slot_count, coefficients[i + 1]), scale_);
            evaluator_.multiply(T[i], pt, term);
            evaluator_.add(result, term);
        }
    }

    // 加常数项：c0 / 2（切比雪夫级数的0次项特殊处理）
    encoder_.encode(pt, std::vector<double>(slot_count, coefficients[0] / 2.0), scale_);
    evaluator_.add(result, pt);

    return result;
}


