#pragma once
#include <heongpu.cuh>
#include <ckks_evaluator.cuh>
#include <vector>
#include <utility>
#include <string>
#include <functional>
#include <cmath>

using namespace heongpu;

class Chebyshev{
public:
    Chebyshev(
        HEContext<Scheme::CKKS>& ctx, HEEncoder<Scheme::CKKS>& encoder,
        CKKSEvaluator& evaluator, double scale)
        : ctx_(ctx), encoder_(encoder), evaluator_(evaluator), scale_(scale) 
          {
            slot_count = ctx_.get_poly_modulus_degree() / 2;
          }

    std::vector<double> getCoefficients(
        std::function<double(double)> func, double a, double b, uint32_t degree);
    Ciphertext<Scheme::CKKS> evalSeriesLinear(const Ciphertext<Scheme::CKKS> x,
        const std::vector<double>& coefficients, double a, double b);
private:
    HEContext<Scheme::CKKS>& ctx_;
    HEEncoder<Scheme::CKKS>& encoder_;
    CKKSEvaluator& evaluator_;
    double scale_;
    int slot_count;

};