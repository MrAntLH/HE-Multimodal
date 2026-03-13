#pragma once
#include <heongpu.cuh>
#include <ckks_evaluator.cuh>
#include <chebyshev.cuh>
#include <vector>
#include <utility>
#include <string>
#include <functional>
#include <cmath>

using namespace heongpu;

class NonLinearExperi {
public:
    NonLinearExperi(
        HEContext<Scheme::CKKS>& ctx, HEEncoder<Scheme::CKKS>& encoder,
        CKKSEvaluator& evaluator, double scale)
        : ctx_(ctx), encoder_(encoder), evaluator_(evaluator), scale_(scale), 
          chebyshev_(ctx, encoder, evaluator, scale),
          slot_count(ctx_.get_poly_modulus_degree() / 2)
          {}
    Ciphertext<Scheme::CKKS> gelu_nexus(Ciphertext<Scheme::CKKS>& x);
    Ciphertext<Scheme::CKKS> gelu_thor(Ciphertext<Scheme::CKKS>& x);

    Ciphertext<Scheme::CKKS> layer_norm_nexus(Ciphertext<Scheme::CKKS>& x, int len);
    Ciphertext<Scheme::CKKS> layer_norm_thor(Ciphertext<Scheme::CKKS>& x, int len, double max_var = 10.0);

    /**
     * @brief Implementation of Softmax function, softmax(x) = e^x / sum(e^x);
     * @details Three experimental version‌ : Onlyt one line of input in cipher;
    */
    Ciphertext<Scheme::CKKS> softmax_nexus(Ciphertext<Scheme::CKKS>& x, int len, double max_x);
    Ciphertext<Scheme::CKKS> softmax_thor(Ciphertext<Scheme::CKKS>& x, int len, int delta_exp = 2, int delta_inv = 3);
    Ciphertext<Scheme::CKKS> softmax_our(Ciphertext<Scheme::CKKS>& x, int len, int delta_exp = 3);

    Ciphertext<Scheme::CKKS> ArgmaxNexus(Ciphertext<Scheme::CKKS>& x, int len);
    Ciphertext<Scheme::CKKS> ArgmaxEffi(Ciphertext<Scheme::CKKS>& x, int len);
    Ciphertext<Scheme::CKKS> ArgmaxOur(Ciphertext<Scheme::CKKS>& x, int len);

    Ciphertext<Scheme::CKKS> tanh_thor(const Ciphertext<Scheme::CKKS>& x, double scale = 40.0);

    Ciphertext<Scheme::CKKS> GridSampleOurs(const Ciphertext<Scheme::CKKS>& grid,
        const Ciphertext<Scheme::CKKS>& x_scale, const Ciphertext<Scheme::CKKS>& y_scale,
        int grid_width, int grid_height);
    void TopkGatherOurs(const Ciphertext<Scheme::CKKS>& query_scores, const std::vector<Ciphertext<Scheme::CKKS>>& query_coords,
        int Num, int Topk, std::vector<Ciphertext<Scheme::CKKS>>& ct_outputs);

private:
    HEContext<Scheme::CKKS>& ctx_;
    HEEncoder<Scheme::CKKS>& encoder_;
    CKKSEvaluator& evaluator_;
    Chebyshev chebyshev_;

    const double scale_;
    const int slot_count;

    /* low level functions */
    Ciphertext<Scheme::CKKS> invSqrt_nexus(Ciphertext<Scheme::CKKS> x, int d_newt = 4, int d_gold = 2);
    Ciphertext<Scheme::CKKS> invSqrt_thor(Ciphertext<Scheme::CKKS> x, 
        double e, double alpha = 0.001, int MAX_IT = 100, bool relax_mode = true);
    Ciphertext<Scheme::CKKS> sgn_nexus(Ciphertext<Scheme::CKKS> x, int d_g = 2, int d_f = 2, double sgn_factor = 0.5);
    Ciphertext<Scheme::CKKS> exp_nexus(Ciphertext<Scheme::CKKS> x, int iter = 7);
    Ciphertext<Scheme::CKKS> exp_thor(Ciphertext<Scheme::CKKS> x);
    Ciphertext<Scheme::CKKS> inverse_nexus(Ciphertext<Scheme::CKKS> x, int iter = 4);
    Ciphertext<Scheme::CKKS> inverse_thor(Ciphertext<Scheme::CKKS> x, int iter = 4);
    void eval_odd_deg9_poly(const std::vector<double> &a, Ciphertext<Scheme::CKKS>& x, Ciphertext<Scheme::CKKS>& dest);
    void eval_odd_deg7_poly(const std::vector<double> &a, Ciphertext<Scheme::CKKS>& x, Ciphertext<Scheme::CKKS>& dest);
    void eval_poly_stockmeyer(const std::vector<double> &a, Ciphertext<Scheme::CKKS>& x, Ciphertext<Scheme::CKKS>& dest);

    /* base func */
    Ciphertext<Scheme::CKKS> initGuess(Ciphertext<Scheme::CKKS> x);
    Ciphertext<Scheme::CKKS> newtonIter(Ciphertext<Scheme::CKKS> x, Ciphertext<Scheme::CKKS> res, int iter = 4);
    std::pair<Ciphertext<Scheme::CKKS>, Ciphertext<Scheme::CKKS>> goldSchmidtIter(
        Ciphertext<Scheme::CKKS> x_0, Ciphertext<Scheme::CKKS> y_0, int d = 1);

    /* argmax_nexus */
    Ciphertext<Scheme::CKKS> cipher_bigger(Ciphertext<Scheme::CKKS>& cipher_a, Ciphertext<Scheme::CKKS>& cipher_b);
    Ciphertext<Scheme::CKKS> cipher_max(Ciphertext<Scheme::CKKS>& x, int len);

    /* argmax_efficient */
    Ciphertext<Scheme::CKKS> transRow(Ciphertext<Scheme::CKKS>& x, int N, int log2N);
    Ciphertext<Scheme::CKKS> replicateRow(Ciphertext<Scheme::CKKS>& x, int N, int log2N);
    Ciphertext<Scheme::CKKS> replicateColumn(Ciphertext<Scheme::CKKS>& x, int N, int log2N);
    Ciphertext<Scheme::CKKS> sgnEffi(Ciphertext<Scheme::CKKS>& x, 
        int d_g = 2, int d_f = 2, double A = 1.0, double B = 0.0);
    Ciphertext<Scheme::CKKS> sgn_rsqrt(Ciphertext<Scheme::CKKS> x, int d_newt = 2, int d_gold = 2);
    Ciphertext<Scheme::CKKS> abs_isqrt(Ciphertext<Scheme::CKKS> x, int d_newt = 2, int d_gold = 2);
    Ciphertext<Scheme::CKKS> sumRows(Ciphertext<Scheme::CKKS>& x, int N, int log2N);

    /*  compute the sum of the cipher */
    Ciphertext<Scheme::CKKS> cipher_sum(Ciphertext<Scheme::CKKS>& x, int len, int log_step);
    // GridSampleOurs
    Ciphertext<Scheme::CKKS> getGridMask(const Ciphertext<Scheme::CKKS>& scale, int grid_Size, std::vector<double>& target_Ind);
    // TopkGatherOurs
    Ciphertext<Scheme::CKKS> getCipherRank(const Ciphertext<Scheme::CKKS>& ct_input, int Num);
    void getRankCoord(const Ciphertext<Scheme::CKKS>& ct_rank, const std::vector<Ciphertext<Scheme::CKKS>>& query_coords, 
        int Num, int getTopInd, std::vector<Ciphertext<Scheme::CKKS>>& ct_outputs);
};
