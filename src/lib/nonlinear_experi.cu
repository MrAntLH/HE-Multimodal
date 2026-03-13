#include "nonlinear_experi.cuh"
#include <cmath>
#include <algorithm>
#include <iostream>
#include <string>

// ------------------- gelu -------------------
Ciphertext<Scheme::CKKS> NonLinearExperi::gelu_nexus(Ciphertext<Scheme::CKKS>& x){
    const std::vector<double> gelu_poly = {2.25775755e-04, 0.5, 3.96880960e-01, 
        -6.37042698e-02, 8.38841647e-03, -7.17830961e-04, 3.49617829e-05, -7.26059653e-07};
    Ciphertext<Scheme::CKKS> b0, b1;
    Plaintext<Scheme::CKKS> p0, p1, delta;
    std::vector<double> dest(slot_count, 0.0);
    encoder_.encode(p0, std::vector<double>(slot_count, -3.5), scale_);
    encoder_.encode(p1, std::vector<double>(slot_count, 3.5), scale_);
    encoder_.encode(delta, std::vector<double>(slot_count, 1.0 / 8.5), scale_);

    // b0 = (x + 3.5) * delta
    evaluator_.sub(x, p0, b0);
    evaluator_.multiply(b0, delta);

    // b1 = (x - 3.5) * delta
    evaluator_.sub(x, p1, b1);
    evaluator_.multiply(b1, delta);

    b0 = sgn_nexus(b0, 2, 2, 0.5);
    b1 = sgn_nexus(b1, 2, 2, 0.5);

    Plaintext<Scheme::CKKS> half(ctx_);
    encoder_.encode(half, std::vector<double>(slot_count, 0.5), scale_);
    // a0: x < - 3.5
    Ciphertext<Scheme::CKKS> a0, a1, a2;
    // a1 = b0 - b1, -3.5 < x < 3.5
    evaluator_.sub(b0, b1, a1);
    // a2 = b1 + 0.5, x > 3.5
    evaluator_.add(b1, half, a2);

    Ciphertext<Scheme::CKKS> x_2, x_4, x_6, x_8, x_10, x_12;
    evaluator_.square(x, x_2);
    evaluator_.square(x_2, x_4);
    evaluator_.multiply(x_2, x_4, x_6);
    evaluator_.square(x_4, x_8);
    evaluator_.multiply(x_2, x_8, x_10);
    evaluator_.square(x_6, x_12);

    Plaintext<Scheme::CKKS> coeff;
    Ciphertext<Scheme::CKKS> Ax, temp;
    encoder_.encode(coeff, std::vector<double>(slot_count, gelu_poly[1]), scale_);
    evaluator_.multiply(x, coeff, Ax);
    encoder_.encode(coeff, std::vector<double>(slot_count, gelu_poly[2]), scale_);
    evaluator_.multiply(x_2, coeff, temp);
    evaluator_.add(Ax, temp);
    encoder_.encode(coeff, std::vector<double>(slot_count, gelu_poly[3]), scale_);
    evaluator_.multiply(x_4, coeff, temp);
    evaluator_.add(Ax, temp);
    encoder_.encode(coeff, std::vector<double>(slot_count, gelu_poly[4]), scale_);
    evaluator_.multiply(x_6, coeff, temp);
    evaluator_.add(Ax, temp);
    encoder_.encode(coeff, std::vector<double>(slot_count, gelu_poly[5]), scale_);
    evaluator_.multiply(x_8, coeff, temp);
    evaluator_.add(Ax, temp);
    encoder_.encode(coeff, std::vector<double>(slot_count, gelu_poly[6]), scale_);
    evaluator_.multiply(x_10, coeff, temp);
    evaluator_.add(Ax, temp);
    encoder_.encode(coeff, std::vector<double>(slot_count, gelu_poly[7]), scale_);
    evaluator_.multiply(x_12, coeff, temp);
    evaluator_.add(Ax, temp);
    encoder_.encode(coeff, std::vector<double>(slot_count, gelu_poly[0]), scale_);
    evaluator_.add(Ax, coeff);

    Ciphertext<Scheme::CKKS> res;
    evaluator_.multiply(Ax, a1, res);
    evaluator_.multiply(x, a2, temp);
    evaluator_.add(res, temp);

    return res;
}

Ciphertext<Scheme::CKKS> NonLinearExperi::gelu_thor(Ciphertext<Scheme::CKKS>& x){
    std::vector<double> p1 = {
        -1.06240033e-05,  1.64454894e-04, -5.83533517e-04, -3.80912692e-04,
         2.24431193e-03,  8.92295204e-03, -1.05277477e-02, -1.91827040e-02,
        -2.04634786e-01,  4.54014410e-01, -5.40759203e-01,  5.67745523e+00,
        -1.36433727e+01,  1.82574621e+01, -8.48849601e+01,  1.28686741e+02,
         3.66720281e+02, -1.01400159e+03, -1.26278856e+02,  2.21728878e+03,
        -9.95421415e+02, -2.31059465e+03,  1.73583957e+03,  1.27394360e+03,
        -1.27836230e+03, -3.66781716e+02,  4.79663919e+02,  4.94610178e+01,
        -9.06754761e+01, -2.36515790e+00,  8.74311855e+00,  1.62838703e-02};

    std::vector<double> p2 = {
        -1.70270667e+02,  6.81076279e+01,  1.79197364e+03, -6.81621043e+02,
        -8.49256169e+03,  3.05629446e+03,  2.39579397e+04, -8.10435126e+03,
        -4.48145152e+04,  1.41297616e+04,  5.86197512e+04, -1.70371505e+04,
        -5.51326382e+04,  1.45532495e+04,  3.77866438e+04, -8.87673890e+03,
        -1.89514802e+04,  3.84972853e+03,  6.94169727e+03, -1.16901058e+03,
        -1.84658407e+03,  2.41693754e+02,  3.54452276e+02, -3.24499570e+01,
        -4.91918227e+01,  2.58122977e+00,  5.78392852e+00, -9.45171527e-02};

    for(int i = 0; i < p2.size(); i++)
        p2[i] *= 0.5;
    std::reverse(p1.begin(), p1.end());
    std::reverse(p2.begin(), p2.end());
    double thor_scale = 64;
    Plaintext<Scheme::CKKS> half_pt, scale_pt;
    encoder_.encode(half_pt, std::vector<double>(slot_count, 0.5), scale_);
    encoder_.encode(scale_pt, std::vector<double>(slot_count, 1.0 / thor_scale), scale_);

    Ciphertext<Scheme::CKKS> p1_res, p2_res, res;
    evaluator_.multiply(x, scale_pt, res);
    eval_poly_stockmeyer(p1, res, p1_res);
    eval_poly_stockmeyer(p2, p1_res, p2_res);

    evaluator_.add(p2_res, half_pt);
    evaluator_.multiply(x, p2_res, res);

    return res;
}

// ------------------- layer_norm -------------------
Ciphertext<Scheme::CKKS> NonLinearExperi::layer_norm_nexus(Ciphertext<Scheme::CKKS>& x, int len){
    int log_step = 0;
    while ((1 << log_step) < len) {
        log_step++;
    }
    int max_len = pow(2, log_step);
    if(max_len > slot_count/2)
        throw std::invalid_argument("len is larger than slot_count/2");

    Ciphertext<Scheme::CKKS> x2_sum, x2, res;
    evaluator_.square(x, x2);
    x2_sum = cipher_sum(x2, max_len, log_step);
    Plaintext<Scheme::CKKS> delta;
    encoder_.encode(delta, std::vector<double>(slot_count, 1.0/len), scale_);
    evaluator_.multiply(x2_sum, delta);
    res = invSqrt_nexus(x2_sum, 4, 2);
    evaluator_.multiply(res, x);
    return res;
}

Ciphertext<Scheme::CKKS> NonLinearExperi::layer_norm_thor(Ciphertext<Scheme::CKKS>& x, int len, double max_var){
    int log_step = 0;
    while ((1 << log_step) < len) {
        log_step++;
    }
    int max_len = pow(2, log_step);
    if(max_len > slot_count/2)
        throw std::invalid_argument("len is larger than slot_count/2");
    Ciphertext<Scheme::CKKS> scaled_x = x;
    {
        Plaintext<Scheme::CKKS> pt_1_n;
        encoder_.encode(pt_1_n, std::vector<double>(slot_count, 1.0/(len * max_var)), scale_);
        evaluator_.multiply(scaled_x, pt_1_n);
    }
    Ciphertext<Scheme::CKKS> numerator, variance;
    {
        Ciphertext<Scheme::CKKS> nx, sum_x, sq_sum_x, x2, sum_x2, n_sum_x2;
        sum_x = cipher_sum(scaled_x, max_len, log_step);
        evaluator_.square(sum_x, sq_sum_x);
        evaluator_.square(scaled_x, x2);
        sum_x2 = cipher_sum(x2, max_len, log_step);
        Plaintext<Scheme::CKKS> pt_n;
        encoder_.encode(pt_n, std::vector<double>(slot_count, len), scale_);
        evaluator_.multiply(scaled_x, pt_n, nx);
        evaluator_.multiply(sum_x2, pt_n, n_sum_x2);
        // Compute numerator n*x - sigma(x)
        evaluator_.sub(nx, sum_x, numerator);
        // Compute variance = n_sum_x2 - sq_sum_x
        evaluator_.sub(n_sum_x2, sq_sum_x, variance);
        Plaintext<Scheme::CKKS> pt_delta;
        encoder_.encode(pt_delta, std::vector<double>(slot_count, 1e-5), scale_);
        evaluator_.add(variance, pt_delta);
    }
    // note: invSqrt_thor's e should be min_var/max_var
    Ciphertext<Scheme::CKKS> denominator = invSqrt_thor(variance, 1.0/len);
    Ciphertext<Scheme::CKKS> res;
    evaluator_.multiply(numerator, denominator, res);
    return res;
}

// ------------------- softmax -------------------
Ciphertext<Scheme::CKKS> NonLinearExperi::softmax_nexus(Ciphertext<Scheme::CKKS>& x, int len, double max_x){
    int log_step = 0;
    while ((1 << log_step) < len) {
        log_step++;
    }
    int max_len = pow(2, log_step);
    if(max_len > slot_count/2)
        throw std::invalid_argument("len is larger than slot_count/2");
    Ciphertext<Scheme::CKKS> scaled_x = x;
    {
        Plaintext<Scheme::CKKS> max_pt;
        encoder_.encode(max_pt, std::vector<double>(slot_count, max_x), scale_);
        evaluator_.sub(scaled_x, max_pt);
    }
    // compute e^x
    Ciphertext<Scheme::CKKS> exp_x;
    {
        exp_x = exp_nexus(scaled_x, 7);
        std::vector<double> mask_vector;
        mask_vector.reserve(slot_count);
        mask_vector.insert(mask_vector.end(), max_len, 1.0);  
        mask_vector.insert(mask_vector.end(), slot_count - max_len, 0.0);
        Plaintext<Scheme::CKKS> mask_pt(ctx_);
        encoder_.encode(mask_pt, mask_vector, scale_);
        evaluator_.multiply(exp_x, mask_pt);
    }
    // compute sum(e^x)
    Ciphertext<Scheme::CKKS> exp_sum = cipher_sum(exp_x, max_len, log_step);
    // !!! The exp_sum need to be (0,2)
    Plaintext<Scheme::CKKS> delta;
    encoder_.encode(delta, std::vector<double>(slot_count, 1.0/len), scale_);
    evaluator_.multiply(exp_sum, delta);
    // compute 1/sum(e^x)
    Ciphertext<Scheme::CKKS> res = inverse_nexus(exp_sum);
    // recover to 1/res
    evaluator_.multiply(res, delta);
    evaluator_.multiply(res, exp_x);
    return res;
}

Ciphertext<Scheme::CKKS> NonLinearExperi::exp_nexus(Ciphertext<Scheme::CKKS> x, int iter) {
    // 编码 1/2^iter
    Plaintext<Scheme::CKKS> inverse_128(ctx_); 
    encoder_.encode(inverse_128, std::vector<double>(slot_count, 1.0/pow(2, iter)), scale_); 

    // x = x * 1/2^iter
    evaluator_.multiply(x, inverse_128);

    // 编码 1.0 
    Plaintext<Scheme::CKKS> one(ctx_); 
    encoder_.encode(one, std::vector<double>(slot_count, 1), scale_);

    // x = x + 1 
    evaluator_.add(x, one);
 
    for (int i = 0; i < iter; i++)
        evaluator_.square(x);
    return x; 
}

Ciphertext<Scheme::CKKS> NonLinearExperi::inverse_nexus(Ciphertext<Scheme::CKKS> x, int iter) {
    // Note: The input need to limit to (0,2)
    Plaintext<Scheme::CKKS> onep(ctx_);
    
    Ciphertext<Scheme::CKKS> y, res, tmp;
    encoder_.encode(onep, std::vector<double>(slot_count, 1.0), scale_);
    evaluator_.sub(x, onep, y); // y = x - 1
    evaluator_.negate(y); // y = 1 - x
    encoder_.encode(onep, std::vector<double>(slot_count, 1.0), scale_);
    evaluator_.add(y, onep, res); // res = 2 - x

    for (int i = 0; i < iter; ++i) {
        evaluator_.square(y);
        encoder_.encode(onep, std::vector<double>(slot_count, 1.0), scale_);
        evaluator_.add(y, onep, tmp); // tmp = 1 + y^2
        evaluator_.multiply(res, tmp);
    }
    return res;
}

Ciphertext<Scheme::CKKS> NonLinearExperi::softmax_thor(Ciphertext<Scheme::CKKS>& x, int len, int delta_exp, int delta_inv){
    int log_step = 0;
    while ((1 << log_step) < len) {
        log_step++;
    }
    int max_len = pow(2, log_step);
    if(max_len > slot_count/2)
        throw std::invalid_argument("len is larger than slot_count/2");

    Ciphertext<Scheme::CKKS> scaled_x = x;
    {   
        // Multiply 1/( 4 * 2^delta_exp * 2^delta_inv), the 4 for y = e^(4x)
        Plaintext<Scheme::CKKS> scale_pt(ctx_);
        double scale_num = 4 * pow(2, delta_exp) * pow(2, delta_inv);
        encoder_.encode(scale_pt, std::vector<double>(slot_count, 1.0 / scale_num), scale_);
        evaluator_.multiply(scaled_x, scale_pt);
    }
    Ciphertext<Scheme::CKKS> exp_x;
    {
        exp_x = exp_thor(scaled_x);
        for(int i = 0; i < delta_exp; i++)
            evaluator_.square(exp_x);
        std::vector<double> mask_vector;
        mask_vector.reserve(slot_count);
        mask_vector.insert(mask_vector.end(), max_len, 1.0);  
        mask_vector.insert(mask_vector.end(), slot_count - max_len, 0.0);
        Plaintext<Scheme::CKKS> mask_pt(ctx_);
        encoder_.encode(mask_pt, mask_vector, scale_);
        evaluator_.multiply(exp_x, mask_pt);
    }
    Ciphertext<Scheme::CKKS> sum_exp_x = cipher_sum(exp_x, max_len, log_step);
    Plaintext<Scheme::CKKS> scaled_pt(ctx_);
    {
        encoder_.encode(scaled_pt, std::vector<double>(slot_count, 1.0 / len), scale_);
        evaluator_.multiply(sum_exp_x, scaled_pt);
    }
    // Ciphertext<Scheme::CKKS> sum_exp_x_inv = inverse_thor(sum_exp_x, delta_inv);
    Ciphertext<Scheme::CKKS> sum_exp_x_inv;
    {
        sum_exp_x_inv = inverse_nexus(sum_exp_x);
        evaluator_.multiply(sum_exp_x_inv, scaled_pt);
    }
    Ciphertext<Scheme::CKKS> res;
    {
        evaluator_.multiply(exp_x, sum_exp_x_inv, res);
        for(int i = 0; i < delta_inv; i++){
            Ciphertext<Scheme::CKKS> res_square = res, sum_exp, sum_exp_inv;
            evaluator_.square(res_square);
            sum_exp = cipher_sum(res_square, max_len, log_step);
            Plaintext<Scheme::CKKS> scaled_pt(ctx_);
            encoder_.encode(scaled_pt, std::vector<double>(slot_count, 500), scale_);
            evaluator_.multiply(sum_exp, scaled_pt);
            sum_exp_inv = inverse_nexus(sum_exp);
            encoder_.encode(scaled_pt, std::vector<double>(slot_count, 500), scale_);
            evaluator_.multiply(sum_exp_inv, scaled_pt);

            evaluator_.multiply(res_square, sum_exp_inv, res);
        }
    }
    return res;
}

Ciphertext<Scheme::CKKS> NonLinearExperi::exp_thor(Ciphertext<Scheme::CKKS> x){
    Ciphertext<Scheme::CKKS> res = x;
    // The coefficients is y = e^(4x) in [-1,1] 
    std::vector<double> p = { 
            0.9999999945, 3.9999999887, 8.0000007486, 10.6666672345, 
            10.6666501742, 8.5333250358, 5.6890263895, 3.2508477871, 
            1.6248375631, 0.7222123556, 0.2901854042, 0.1054371349, 
            0.0335527478, 0.0103825392, 0.0039797889, 0.0010451185};
    // std::reverse(p.begin(), p.end());
    eval_poly_stockmeyer(p, res, res);
    return res;
}

Ciphertext<Scheme::CKKS> NonLinearExperi::inverse_thor(Ciphertext<Scheme::CKKS> x, int iter){
    double en = pow(2, -11);
    double alpha = 0.01;
    Ciphertext<Scheme::CKKS> an;
    {
        Plaintext<Scheme::CKKS> pt;
        encoder_.encode(pt, std::vector<double>(slot_count, 1.0), scale_);
        evaluator_.encrypt(pt, an);
    }
    Ciphertext<Scheme::CKKS> bn = x;
    int inverse_iter = 0;
    while( en < 1 - alpha){
        inverse_iter++;
        double kn = 2 / (en + 1);
        // Update an = an * (2/kn - bn), bn = bn * (2/kn - bn)
        Ciphertext<Scheme::CKKS> temp = bn;
        evaluator_.negate(temp);
        Plaintext<Scheme::CKKS> pt;
        encoder_.encode(pt, std::vector<double>(slot_count, 2.0 / kn), scale_);
        evaluator_.add(temp, pt);
        evaluator_.multiply(an, temp);
        evaluator_.multiply(bn, temp);
        en = kn * en * ( 2 - kn * en);
    }
    std::cout << "[inverse_thor] inverse_iter = " << inverse_iter << std::endl;
    return an;
}

Ciphertext<Scheme::CKKS> NonLinearExperi::softmax_our(Ciphertext<Scheme::CKKS>& x, int len, int delta_exp){
    int log_step = 0;
    while ((1 << log_step) < len) {
        log_step++;
    }
    int max_len = pow(2, log_step);
    if(max_len > slot_count/2)
        throw std::invalid_argument("len is larger than slot_count/2");

    Ciphertext<Scheme::CKKS> scaled_x = x;
    {   
        Plaintext<Scheme::CKKS> scale_pt(ctx_);
        double scale_num = 4 * pow(2, delta_exp);
        encoder_.encode(scale_pt, std::vector<double>(slot_count, 1.0 / scale_num), scale_);
        evaluator_.multiply(scaled_x, scale_pt);
    }
    Ciphertext<Scheme::CKKS> exp_x;
    {
        exp_x = exp_thor(scaled_x);
        for(int i = 0; i < delta_exp; i++)
            evaluator_.square(exp_x);
        std::vector<double> mask_vector;
        mask_vector.reserve(slot_count);
        mask_vector.insert(mask_vector.end(), max_len, 1.0);  
        mask_vector.insert(mask_vector.end(), slot_count - max_len, 0.0);
        Plaintext<Scheme::CKKS> mask_pt(ctx_);
        encoder_.encode(mask_pt, mask_vector, scale_);
        evaluator_.multiply(exp_x, mask_pt);
    }
    Ciphertext<Scheme::CKKS> sum_exp_x = cipher_sum(exp_x, max_len, log_step);
    Plaintext<Scheme::CKKS> scaled_pt(ctx_);
    {
        encoder_.encode(scaled_pt, std::vector<double>(slot_count, 1.0 / 64), scale_);
        evaluator_.multiply(sum_exp_x, scaled_pt);
    }
    Ciphertext<Scheme::CKKS> sum_exp_x_inv;
    {
        sum_exp_x_inv = inverse_nexus(sum_exp_x);
        evaluator_.multiply(sum_exp_x_inv, scaled_pt);
    }
    Ciphertext<Scheme::CKKS> res;
    evaluator_.multiply(exp_x, sum_exp_x_inv, res);
    return res;
}


// ------------------- tanh ----------------------
Ciphertext<Scheme::CKKS> NonLinearExperi::tanh_thor(const Ciphertext<Scheme::CKKS>& x, double scale){
    std::vector<double> p1 = {
            -7.14529052e+03, -7.76519925e+01,  2.74279201e+04,  2.45150249e+02,
            -4.25793697e+04, -3.01953016e+02,  3.42189880e+04,  1.82989351e+02,
            -1.51158283e+04, -5.64098990e+01,  3.58757327e+03,  8.17596753e+00,
            -4.13341496e+02, -4.29024545e-01,  1.95056729e+01,  2.06201784e-03};
    std::vector<double> p2 = {
            -9.02573450e-03, -1.12320034e-04,  1.08762008e-01,  7.96793166e-04,
            -5.41327356e-01, -1.42873183e-03,  1.46476749e+00, -2.22416152e-03,
            -2.43259032e+00,  1.17381072e-02,  2.74974898e+00, -1.77631073e-02,
            -2.38934873e+00,  1.30194294e-02,  2.02874846e+00, -4.08442578e-03};
    std::reverse(p1.begin(), p1.end());
    std::reverse(p2.begin(), p2.end());

    Plaintext<Scheme::CKKS> scale_pt;
    // x scaled by 1/40
    encoder_.encode(scale_pt, std::vector<double>(slot_count, 1.0 / scale), scale_);
    Ciphertext<Scheme::CKKS> res = x;
    evaluator_.multiply(res, scale_pt);
    eval_poly_stockmeyer(p1, res, res);
    eval_poly_stockmeyer(p2, res, res);
    return res;
}

// ------------------- invert_sqrt -------------------
Ciphertext<Scheme::CKKS> NonLinearExperi::invSqrt_nexus(Ciphertext<Scheme::CKKS> x, int d_newt, int d_gold) {
    Ciphertext<Scheme::CKKS> res = initGuess(x);
    if( d_newt > 0)
        res = newtonIter(x, res, d_newt);
    if( d_gold > 0){
        auto pr = goldSchmidtIter(x, res, d_gold);
        res = pr.second;
    }
    return res;
}

Ciphertext<Scheme::CKKS> NonLinearExperi::initGuess(Ciphertext<Scheme::CKKS> x) {
    Ciphertext<Scheme::CKKS> res(ctx_);
    // 按理说高次多项式来初始化初值，近似速度会更快，但实测后，容易爆炸
    /*
    {
        const std::vector<double> isqrt_poly = {2.7506, -2.0362, 0.6242, -0.0782, 0.0033};
        Plaintext<Scheme::CKKS> plain(ctx_);
        Ciphertext<Scheme::CKKS> x2, x3, x4;
        encoder_.encode(plain, std::vector<double>(slot_count, isqrt_poly[1]), scale_);
        evaluator_.multiply(x, plain, res);

        encoder_.encode(plain, std::vector<double>(slot_count, isqrt_poly[0]), scale_);
        evaluator_.add(res, plain);

        evaluator_.square(x, x2);
        encoder_.encode(plain, std::vector<double>(slot_count, isqrt_poly[2]), scale_);
        evaluator_.multiply(x2, plain);
        evaluator_.add(res, x2);

        evaluator_.multiply(x, x2, x3);
        encoder_.encode(plain, std::vector<double>(slot_count, isqrt_poly[3]), scale_);
        evaluator_.multiply(x3, plain);
        evaluator_.add(res, x3);

        evaluator_.square(x2, x4);
        encoder_.encode(plain, std::vector<double>(slot_count, isqrt_poly[4]), scale_);
        evaluator_.multiply(x4, plain);
        evaluator_.add(res, x4);
    }
    */
    {
        Plaintext<Scheme::CKKS> A(ctx_), B(ctx_);
        encoder_.encode(A, std::vector<double>(slot_count, -0.000129054537), scale_);
        encoder_.encode(B, std::vector<double>(slot_count, 0.129054537), scale_);
        evaluator_.multiply(x, A, res);
        evaluator_.add(res, B);
    }
    return res;
}

Ciphertext<Scheme::CKKS> NonLinearExperi::newtonIter(Ciphertext<Scheme::CKKS> x, Ciphertext<Scheme::CKKS> res, int iter) {
    // y_n+1 = 1.5 * y_n - 0.5 * x * y_n^3
    for (int i = 0; i < iter; ++i) {
        Plaintext<Scheme::CKKS> three_half(ctx_), neg_half(ctx_);
        encoder_.encode(three_half, std::vector<double>(slot_count, 1.5), scale_);
        encoder_.encode(neg_half, std::vector<double>(slot_count, -0.5), scale_);

        // res_B = -0.5 * x * y^3
        Ciphertext<Scheme::CKKS> res_B;
        {
            Ciphertext<Scheme::CKKS> y2;
            evaluator_.square(res, y2);
            evaluator_.multiply(x, neg_half, res_B);
            evaluator_.multiply(res_B, res);
            evaluator_.multiply(res_B, y2);
        }
        // res_A = 1.5 * y
        Ciphertext<Scheme::CKKS> res_A;
        {
            evaluator_.multiply(res, three_half, res_A);
        }
        evaluator_.add(res_A, res_B, res);
    }
    return res;
}

std::pair<Ciphertext<Scheme::CKKS>, Ciphertext<Scheme::CKKS>> NonLinearExperi::goldSchmidtIter(
    Ciphertext<Scheme::CKKS> x_0, Ciphertext<Scheme::CKKS> y_0, int d) {
    Plaintext<Scheme::CKKS> const_pt(ctx_);
    encoder_.encode(const_pt, std::vector<double>(slot_count, 0.5), scale_);

    // x = x_0 * y, h = 0.5 * y
    Ciphertext<Scheme::CKKS> x, h;
    {
        evaluator_.multiply(x_0, y_0, x);
        evaluator_.multiply(y_0, const_pt, h);
    }
    
    Ciphertext<Scheme::CKKS> r, tmp;
    for( int i = 0; i < d; i++){
        // r_n = 0.5 - x_n * h_n
        encoder_.encode(const_pt, std::vector<double>(slot_count, 0.5), scale_);
        evaluator_.multiply(x, h, r);
        evaluator_.negate(r);
        evaluator_.add(r, const_pt);

        // x_n+1 = x_n + x_n * r_n = x_n * (1 + r_n)
        evaluator_.multiply(x, r, tmp);
        evaluator_.add(x, tmp);

        // h_n+1 = h_n + h_n * r_n = h_n * (1 + r_n)
        evaluator_.multiply(h, r, tmp);
        evaluator_.add(h, tmp);
    }
    encoder_.encode(const_pt, std::vector<double>(slot_count, 2.0), scale_);
    evaluator_.multiply(h, const_pt);
    // x ≈ ✓x, h * 2 ≈ 1/✓x
    return std::make_pair(x, h);
}

Ciphertext<Scheme::CKKS> NonLinearExperi::invSqrt_thor(const Ciphertext<Scheme::CKKS> x, double e, double alpha, int MAX_IT, bool relax_mode) {
    if( e < 0 || e >= 1 )
        throw std::invalid_argument("[invSqrt_thor]: e should be in [0,1)");
    Ciphertext<Scheme::CKKS> bn;
    {
        Plaintext<Scheme::CKKS> pt;
        encoder_.encode(pt, std::vector<double>(slot_count, 1.0), scale_);
        evaluator_.encrypt(pt, bn);
    }
    Ciphertext<Scheme::CKKS> an = x;
    double en = e;
    int it = 0;
    std::vector<double> relax_mode_coff = {2.6374, 2.1722, 1.5135, 1.0907};
    if(relax_mode){
        MAX_IT = relax_mode_coff.size();
    }
    while( en < 1 - alpha && it < MAX_IT){
        // compute the root of [（1-en^3）x^2 + (6en^2-6)x + (9-9en) = 0]
        double kn;
        if(relax_mode){
            kn = relax_mode_coff[it];
        }else{
            // as ax^2 + bx + c = 0
            double a = 1 - en * en * en;
            double b = 6 * en * en - 6;
            double c = 9 - 9 * en;
            
            std::complex<double> disc = std::complex<double>(b*b - 4.0*a*c, 0.0);
            std::complex<double> sd = std::sqrt(disc);
            std::complex<double> r2 = (std::complex<double>(-b, 0.0) - sd) / (2.0 * a);
            kn = std::real(r2);

            // Update en
            double t = 3.0 - kn * en;
            en = kn * en * (t * t) / 4.0;
        }
        // Update bn
        {
            Ciphertext<Scheme::CKKS> bn1, bn2;
            double kn_pow3_2 = kn * std::sqrt(kn);
            Plaintext<Scheme::CKKS> pt(ctx_);
            encoder_.encode(pt, std::vector<double>(slot_count, kn_pow3_2 / 2), scale_);
            evaluator_.multiply(bn, pt, bn1);
            evaluator_.negate(an, bn2);
            encoder_.encode(pt, std::vector<double>(slot_count, 3.0 / kn), scale_);
            evaluator_.add(bn2, pt);
            evaluator_.multiply(bn1, bn2, bn);
        }
        // Update an
        {
            double temp = kn * kn * kn / 4.0;
            Plaintext<Scheme::CKKS> pt(ctx_);
            encoder_.encode(pt, std::vector<double>(slot_count, temp), scale_);
            Ciphertext<Scheme::CKKS> an1, an2;
            evaluator_.multiply(an, pt, an1);
            evaluator_.negate(an, an2);
            encoder_.encode(pt, std::vector<double>(slot_count, 3.0 / kn), scale_);
            evaluator_.add(an2, pt);
            evaluator_.square(an2);
            evaluator_.multiply(an1, an2, an);
        }
        ++it;
    }
    if(!relax_mode && it == MAX_IT)
        throw std::invalid_argument("[invSqrt_thor]: it upto MAX_IT = " + std::to_string(MAX_IT));
    return bn;
}

// ------------------- sign evaluation -------------------
Ciphertext<Scheme::CKKS> NonLinearExperi::sgn_nexus(Ciphertext<Scheme::CKKS> x, int d_g, int d_f, double sgn_factor) {
    std::vector<double> f4 = {0, 315, 0, -420, 0, 378, 0, -180, 0, 35};
    const int f4_scale = (1 << 7);
    std::vector<double> g4 = {0, 5850, 0, -34974, 0, 97015, 0, -113492, 0, 46623};
    const int g4_scale = (1 << 10);
    
    std::vector<double> f4_last(10, 0.0), g4_last(10, 0.0);
    // Output range: [-sgn_factor, sgn_factor]
    for (int i = 0; i <= 9; ++i) {
        f4[i] /= (double)f4_scale;
        f4_last[i] = f4[i] * sgn_factor;
        g4[i] /= (double)g4_scale;
        g4_last[i] = g4[i] * sgn_factor;
    }

    Ciphertext<Scheme::CKKS> dest = x;
    for (int i = 0; i < d_g - 1; ++i)
        eval_odd_deg9_poly(g4, dest, dest);
    eval_odd_deg9_poly(g4_last, dest, dest);
    for (int i = 0; i < d_f - 1; ++i) 
        eval_odd_deg9_poly(f4, dest, dest);
    eval_odd_deg9_poly(f4_last, dest, dest);
    return dest;
}

Ciphertext<Scheme::CKKS> NonLinearExperi::sgn_rsqrt(Ciphertext<Scheme::CKKS> x, int d_newt, int d_gold){
    // sgn(x) ≈ x * invSqrt_nexus(x^2 + ε)
    Ciphertext<Scheme::CKKS> x2, res;
    evaluator_.square(x, x2);

    Plaintext<Scheme::CKKS> plain(ctx_);
    encoder_.encode(plain, std::vector<double>(slot_count, 0.00001), scale_);
    evaluator_.add(x2, plain);

    res = invSqrt_nexus(x2, d_newt, d_gold);
    evaluator_.multiply(res, x);
    return res;
}

Ciphertext<Scheme::CKKS> NonLinearExperi::abs_isqrt(Ciphertext<Scheme::CKKS> x, int d_newt, int d_gold){
    // |x| ≈ sqrt(x^2 + ε) = (x^2 + ε) * invSqrt_nexus(x^2 + ε)
    Ciphertext<Scheme::CKKS> x2, res;
    evaluator_.square(x, x2);

    Plaintext<Scheme::CKKS> plain(ctx_);
    encoder_.encode(plain, std::vector<double>(slot_count, 0.00001), scale_);
    evaluator_.add(x2, plain);

    res = invSqrt_nexus(x2, d_newt, d_gold);
    evaluator_.multiply(res, x2);
    return res;
}

// ------------------- polynomial eval -------------------
void NonLinearExperi::eval_odd_deg9_poly(const std::vector<double> &a, Ciphertext<Scheme::CKKS>& x, Ciphertext<Scheme::CKKS>& dest) {
    /*  
        P(x) = a9 x^9 + a7 x^7 + a5 x^5 + a3 x^3 + a1 x

        T1 = (a3 + a5 x^2) x^3
        T2 = (a7 x + a9 x^3) x^6
        T3 = a1 x
        P(x) = T1 + T2 + T3
    */

    Ciphertext<Scheme::CKKS> x2, x3, x6;
    evaluator_.square(x, x2); // x2 = x^2
    evaluator_.multiply(x2, x, x3); // x3 = x^3
    evaluator_.square(x3, x6); // x6 = x^6

    // T1 = (a3 + a5 * x2) * x3
    Ciphertext<Scheme::CKKS> T1;
    {
        Plaintext<Scheme::CKKS> a5(ctx_);
        encoder_.encode(a5, std::vector<double>(slot_count, a[5]), scale_);
        evaluator_.multiply(x2, a5, T1);

        Plaintext<Scheme::CKKS> a3(ctx_);
        encoder_.encode(a3, std::vector<double>(slot_count, a[3]), scale_);
        evaluator_.add(T1, a3);

        evaluator_.multiply(T1, x3);
    }

    // T2 = (a7 * x + a9 * x3) * x6
    Ciphertext<Scheme::CKKS> T2;
    {
        Plaintext<Scheme::CKKS> a9(ctx_);
        encoder_.encode(a9, std::vector<double>(slot_count, a[9]), scale_);
        evaluator_.multiply(x3, a9, T2);

        Plaintext<Scheme::CKKS> a7(ctx_);
        encoder_.encode(a7, std::vector<double>(slot_count, a[7]), scale_);
        Ciphertext<Scheme::CKKS> temp;
        evaluator_.multiply(x, a7, temp);

        evaluator_.add(T2, temp); 
        evaluator_.multiply(T2, x6);
    }

    // T3 = a1 * x
    Ciphertext<Scheme::CKKS> T3;
    {
        Plaintext<Scheme::CKKS> a1(ctx_);
        encoder_.encode(a1, std::vector<double>(slot_count, a[1]), scale_);
        evaluator_.multiply(x, a1, T3);
    }

    dest = T2;
    evaluator_.add(dest, T1);
    evaluator_.add(dest, T3);
}

void NonLinearExperi::eval_odd_deg7_poly(const std::vector<double> &a, Ciphertext<Scheme::CKKS>& x, Ciphertext<Scheme::CKKS>& dest) {
    /*  
        P(x) = a7 x^7 + a5 x^5 + a3 x^3 + a1 x = x [ a1 + x² ( a3 + x² ( a5 + a7 x² ) ) ]
    */

    Ciphertext<Scheme::CKKS> x2;
    evaluator_.square(x, x2);

    // term = a5 + a7 x²
    Ciphertext<Scheme::CKKS> term;
    {
        Plaintext<Scheme::CKKS> a7(ctx_);
        encoder_.encode(a7, std::vector<double>(slot_count, a[7]), scale_);
        evaluator_.multiply(x2, a7, term);

        Plaintext<Scheme::CKKS> a5(ctx_);
        encoder_.encode(a5, std::vector<double>(slot_count, a[5]), scale_);
        evaluator_.add(term, a5);
    }

    // term = a3 + x² (a5 + a7 x²)
    {
        evaluator_.multiply(term, x2);

        Plaintext<Scheme::CKKS> a3(ctx_);
        encoder_.encode(a3, std::vector<double>(slot_count, a[3]), scale_);
        evaluator_.add(term, a3);
    }

    // term = a1 + x² [a3 + x² (a5 + a7 x²)]
    {
        evaluator_.multiply(term, x2); 

        Plaintext<Scheme::CKKS> a1(ctx_);
        encoder_.encode(a1, std::vector<double>(slot_count, a[1]), scale_);
        evaluator_.add(term, a1);
    }

    // dest = x * term = P(x)
    evaluator_.multiply(term, x, dest);  
}

void NonLinearExperi::eval_poly_stockmeyer(const std::vector<double> &a, Ciphertext<Scheme::CKKS>& x, Ciphertext<Scheme::CKKS>& dest){
    //  - coefficients a are ordered increasing: a[0] + a[1] x + ... + a[n] x^n
    //  - we require degree in [16, 32], i.e. a.size() > 16 and (a.size()-1) <= 32
    //  - otherwise throw std::invalid_argument

    const int n_coeff = static_cast<int>(a.size());  // number of coefficients
    if (n_coeff < 16 || n_coeff > 32) {
        throw std::invalid_argument("eval_poly_stockmeyer: degree in (16, 32] required");
    }

    // 1) 预编码所有系数（避免重复编码）
    std::vector<Plaintext<Scheme::CKKS>> coeff_pts;
    coeff_pts.reserve(n_coeff);
    for (int i = 0; i < n_coeff; ++i) {
        Plaintext<Scheme::CKKS> pt(ctx_);
        encoder_.encode(pt, std::vector<double>(slot_count, a[i]), scale_);
        coeff_pts.push_back(std::move(pt));
    }

    // 2) 预计算幂：x^2, x^3, x^4, x^8, x^16
    Ciphertext<Scheme::CKKS> x2, x3, x4, x8, x16;
    evaluator_.square(x, x2);               // x2 = x^2
    evaluator_.multiply(x2, x, x3);         // x3 = x^3
    evaluator_.square(x2, x4);              // x4 = x^4
    evaluator_.square(x4, x8);              // x8 = x^8
    evaluator_.square(x8, x16);             // x16 = x^16

    // 3) zero_pt + zero_ct，作为所有 baby-block 的零密文模板
    Plaintext<Scheme::CKKS> zero_pt(ctx_);
    encoder_.encode(zero_pt, std::vector<double>(slot_count, 0.0), scale_);
    Ciphertext<Scheme::CKKS> zero_ct = x;   // 复制 x 的元数据/结构
    evaluator_.multiply(zero_ct, zero_pt);  // zero_ct <- 0

    // 4) x_powers: index 0..3 -> {zero_ct, x, x2, x3}
    std::array<Ciphertext<Scheme::CKKS>, 4> x_powers;
    x_powers[0] = zero_ct;
    x_powers[1] = x;  
    x_powers[2] = x2;
    x_powers[3] = x3;

    auto evaluate_baby = [&](int off, int len) -> Ciphertext<Scheme::CKKS> {
        Ciphertext<Scheme::CKKS> result = x_powers[0]; // copy zero_ct
        evaluator_.add(result, coeff_pts[off]);        // result += a[off]
        for (int i = 1; i < len; ++i) {
            Ciphertext<Scheme::CKKS> tmp;
            // tmp = x_powers[i] * coeff_pts[off+i]
            evaluator_.multiply(x_powers[i], coeff_pts[off + i], tmp);
            evaluator_.add(result, tmp);                // result += tmp
        }
        return result;
    };

    // 6) 划分 baby-blocks（低幂到高幂），每块最多 4 个系数
    std::vector<Ciphertext<Scheme::CKKS>> baby_results;
    baby_results.reserve((n_coeff + 3) / 4);
    for (int off = 0; off < n_coeff; off += 4) {
        int take = std::min(4, n_coeff - off);
        baby_results.push_back(evaluate_baby(off, take));
    }

    // 7) Giant step 1: pairwise combine with multiplier x^4
    std::vector<Ciphertext<Scheme::CKKS>> gs1;
    gs1.reserve((baby_results.size() + 1) / 2);
    for (size_t i = 0; i < baby_results.size(); i += 2) {
        if (i + 1 < baby_results.size()) {
            Ciphertext<Scheme::CKKS> right = std::move(baby_results[i + 1]);
            evaluator_.multiply(right, x4);        // right *= x^4
            Ciphertext<Scheme::CKKS> left = std::move(baby_results[i]);
            evaluator_.add(left, right);           // left += right
            gs1.push_back(std::move(left));
        } else {
            gs1.push_back(std::move(baby_results[i]));
        }
    }

    // 8) Giant step 2: pairwise combine gs1 with multiplier x^8
    std::vector<Ciphertext<Scheme::CKKS>> gs2;
    gs2.reserve((gs1.size() + 1) / 2);
    for (size_t i = 0; i < gs1.size(); i += 2) {
        if (i + 1 < gs1.size()) {
            Ciphertext<Scheme::CKKS> right = std::move(gs1[i + 1]);
            evaluator_.multiply(right, x8);        // right *= x^8
            Ciphertext<Scheme::CKKS> left = std::move(gs1[i]);
            evaluator_.add(left, right);           // left += right
            gs2.push_back(std::move(left));
        } else {
            gs2.push_back(std::move(gs1[i]));
        }
    }

    Ciphertext<Scheme::CKKS> result = std::move(gs2[0]);
    if (gs2.size() >= 2) {
        Ciphertext<Scheme::CKKS> right = std::move(gs2[1]);
        evaluator_.multiply(right, x16);          // right *= x^16
        evaluator_.add(result, right);             // result += right
    }

    dest = std::move(result);
}


// ------------------- Nexus Argmax -----------
Ciphertext<Scheme::CKKS> NonLinearExperi::ArgmaxNexus(Ciphertext<Scheme::CKKS>& x, int len){
    Ciphertext<Scheme::CKKS> res, x_max;
    x_max = cipher_max(x, len);
    evaluator_.sub(x, x_max, res);
    // res = sgn_nexus(res, 2, 2, 1.0);
    res = sgnEffi(res, 2, 2, 1.0, 0.0);

    Plaintext<Scheme::CKKS> plain_one(ctx_);
    encoder_.encode(plain_one, std::vector<double>(slot_count, 1.0), scale_);

    evaluator_.add(res, plain_one);
    return res;
}

Ciphertext<Scheme::CKKS> NonLinearExperi::cipher_bigger(Ciphertext<Scheme::CKKS>& cipher_a, Ciphertext<Scheme::CKKS>& cipher_b){
    Ciphertext<Scheme::CKKS> a_plus_b, a_minus_b, res;
    evaluator_.add(cipher_a, cipher_b, a_plus_b);
    evaluator_.sub(cipher_a, cipher_b, a_minus_b);

    // sgn(a - b) / 2
    // res = sgn_nexus(a_minus_b, 2, 2, 0.5);
    res = sgnEffi(a_minus_b, 2, 2, 0.5, 0.0);


    // (a - b) * sgn(a - b) / 2
    evaluator_.multiply(res, a_minus_b);

    // (a + b) / 2
    Plaintext<Scheme::CKKS> half(ctx_);
    encoder_.encode(half, std::vector<double>(slot_count, 0.5), scale_);
    evaluator_.multiply(a_plus_b, half);

    // res = (a + b) / 2 + (a - b) * sgn(a - b) / 2
    evaluator_.add(res, a_plus_b);

    return res;
}

Ciphertext<Scheme::CKKS> NonLinearExperi::cipher_max(Ciphertext<Scheme::CKKS>& x, int len){
    int log_step = 0;
    while ((1 << log_step) < len) {  // 1<<log_step 即 2^log_step
        log_step++;
    }
    int max_len = pow(2, log_step);
    if(max_len > slot_count/2){
        throw std::invalid_argument("len is larger than slot_count/2");
    }
    Ciphertext<Scheme::CKKS> tmp = x;
    evaluator_.rotate_rows(tmp, -max_len);
    Ciphertext<Scheme::CKKS> res = x;
    evaluator_.add(res, tmp);

    for(int i = 0; i < log_step; i++){
        tmp = res;
        evaluator_.rotate_rows(tmp, pow(2, i));
        res = cipher_bigger(res, tmp);
    }
    return res;
}

Ciphertext<Scheme::CKKS> NonLinearExperi::cipher_sum(Ciphertext<Scheme::CKKS>& x, int len, int log_step){
    Ciphertext<Scheme::CKKS> res, tmp;
    evaluator_.rotate_rows(x, -len, tmp);
    evaluator_.add(x, tmp, res);
    for(int i = 0; i < log_step; i++){
        tmp = res;
        evaluator_.rotate_rows(tmp, pow(2, i));
        evaluator_.add(res, tmp);
    }
    return res;
}

// -------------------- Efficient Argmax -------------------------
Ciphertext<Scheme::CKKS> NonLinearExperi::ArgmaxEffi(Ciphertext<Scheme::CKKS>& x, int len){
    int log2N = 0;
    while ((1 << log2N) < len)
        log2N++;
    int N = pow(2, log2N);
    if( N*N > slot_count)
        throw std::invalid_argument("len^2 is larger than slot_count");

    Ciphertext<Scheme::CKKS> left = x,right = x;
    right = transRow(right, N, log2N);

    left = replicateRow(left, N, log2N);
    right = replicateColumn(right, N, log2N);

    Ciphertext<Scheme::CKKS> res;
    evaluator_.sub(left, right, res);
    // std::vector<double> coefficients = chebyshev_.getCoefficients(
    //     [](double x)-> double{
    //         if(x >= 0 - 1e-10 ) return 1.0;
    //         else return 0.0;
    //     }, -1, 1, 11
    // );
    // res = chebyshev_.evalSeriesLinear(res, coefficients, -1, 1);
    res = sgnEffi(res, 2, 2, 0.5, 0.5);
    res = sumRows(res, N, log2N);

    std::vector<double> coefficients = chebyshev_.getCoefficients(
        [N](double x)-> double{
            if( x > N - 1 ) return 1.0;
            else return 0.0;
        }, 0, N, 11
    );
    res = chebyshev_.evalSeriesLinear(res, coefficients, 0, N);

    return res;
}

Ciphertext<Scheme::CKKS> NonLinearExperi::transRow(Ciphertext<Scheme::CKKS>& x, int N, int log2N){
    Ciphertext<Scheme::CKKS> res = x;
    Ciphertext<Scheme::CKKS> tmp;
    for(int i = 1; i <= log2N; i++){
        tmp = res;
        evaluator_.rotate_rows(tmp, -(N * (N-1) / pow(2, i)) );
        evaluator_.add(res, tmp);
    }
    
    std::vector<double> mask_vec(slot_count, 0.0);
    for(int i = 0; i < N; i++)
        mask_vec[i * N] = 1.0;
    Plaintext<Scheme::CKKS> mask_pt(ctx_);
    encoder_.encode(mask_pt, mask_vec, scale_);
    evaluator_.multiply(res, mask_pt);
    return res;
}

Ciphertext<Scheme::CKKS> NonLinearExperi::replicateRow(Ciphertext<Scheme::CKKS>& x, int N, int log2N){
    Ciphertext<Scheme::CKKS> res = x;
    Ciphertext<Scheme::CKKS> tmp;
    for(int i = 0; i < log2N; i++){
        tmp = res;
        evaluator_.rotate_rows(tmp, -(pow(2, log2N + i)) );
        evaluator_.add(res, tmp);
    }
    return res;
}

Ciphertext<Scheme::CKKS> NonLinearExperi::replicateColumn(Ciphertext<Scheme::CKKS>& x, int N, int log2N){
    Ciphertext<Scheme::CKKS> res = x;
    Ciphertext<Scheme::CKKS> tmp;
    for(int i = 0; i < log2N; i++){
        tmp = res;
        evaluator_.rotate_rows(tmp, -(pow(2, i)) );
        evaluator_.add(res, tmp);
    }
    return res;
}

Ciphertext<Scheme::CKKS> NonLinearExperi::sgnEffi(Ciphertext<Scheme::CKKS>& x, int d_g, int d_f,double A, double B){
    /*  
        The input need to limit to (-1,1)
        The output range Y = A * sgn(x) + B
    */
    // f3(x) = (35 x - 35 x^3 + 21 x^5 - 5 x^7) / 16
    std::vector<double> coeffF3 = {0, 35.0 / 16.0, 0, -35.0 / 16.0, 0, 21.0 / 16.0, 0, -5.0 / 16.0};
    
    // g3(x) = (4589 x - 16577 x^3 + 25614 x^5 - 12860 x^7) / 1024
    std::vector<double> coeffG3 = {0, 4589.0 / 1024.0, 0, -16577.0 / 1024.0, 0, 25614.0 / 1024.0, 0, -12860.0 / 1024.0};

    std::vector<double> coeff_final(coeffF3.size(), 0.0);
    for(int i=1; i < coeffF3.size(); i += 2){
        coeff_final[i] = coeffF3[i] * A;
    }

    Ciphertext<Scheme::CKKS> dest = x;
    for(int i = 0; i < d_g; i++){
        eval_odd_deg7_poly(coeffG3, dest, dest);
    }
    for(int i = 0; i < d_f - 1; i++){
        eval_odd_deg7_poly(coeffF3, dest, dest);
    }
    eval_odd_deg7_poly(coeff_final, dest, dest);
    if(B != 0.0){
        Plaintext<Scheme::CKKS> B_pt(ctx_);
        encoder_.encode(B_pt, std::vector<double>(slot_count, B), scale_);
        evaluator_.add(dest, B_pt);
    }
    return dest;
}

Ciphertext<Scheme::CKKS> NonLinearExperi::sumRows(Ciphertext<Scheme::CKKS>& x, int N, int log2N){
    Ciphertext<Scheme::CKKS> res = x;
    Ciphertext<Scheme::CKKS> tmp;
    for(int i = 0; i < log2N; i++){
        tmp = res;
        evaluator_.rotate_rows(tmp, pow(2, log2N + i) );
        evaluator_.add(res, tmp);
    }

    std::vector<double> mask_vec(slot_count, 0.0);
    for(int i = 0; i < N; i++)
        mask_vec[i] = 1.0;
    Plaintext<Scheme::CKKS> mask_pt(ctx_);
    encoder_.encode(mask_pt, mask_vec, scale_);
    evaluator_.multiply(res, mask_pt);

    return res;
}

// -------------------- Our Argmax -------------------------
Ciphertext<Scheme::CKKS> NonLinearExperi::ArgmaxOur(Ciphertext<Scheme::CKKS>& x, int len){
    int log2N = 0;
    while ((1 << log2N) < len)
        log2N++;
    int N = pow(2, log2N);
    if( N*N > slot_count)
        throw std::invalid_argument("len^2 is larger than slot_count");

    Ciphertext<Scheme::CKKS> left = x,right = x;
    right = transRow(right, N, log2N);

    left = replicateRow(left, N, log2N);
    right = replicateColumn(right, N, log2N);

    Ciphertext<Scheme::CKKS> res;
    evaluator_.sub(left, right, res);
    res = sgnEffi(res, 2, 2, 0.5, 0.5);

    for(int i = 0; i < log2N; i++){
        Ciphertext<Scheme::CKKS> tmp = res;
        evaluator_.rotate_rows(tmp, N * pow(2, i) );
        evaluator_.multiply(res, tmp);
    }
    Plaintext<Scheme::CKKS> scale_pt(ctx_);
    encoder_.encode(scale_pt, std::vector<double>(slot_count, 2.0), scale_);
    evaluator_.multiply(res, scale_pt);
    return res;
}

// -------------------- Our Grid Sample -------------------------
Ciphertext<Scheme::CKKS> NonLinearExperi::GridSampleOurs(const Ciphertext<Scheme::CKKS>& grid,
    const Ciphertext<Scheme::CKKS>& x_scale, const Ciphertext<Scheme::CKKS>& y_scale,
    int grid_width, int grid_height){
    // get the vec/grid mask from scale values
    Ciphertext<Scheme::CKKS> ct_x_mask = x_scale;
    {
        std::vector<double> target_Ind(slot_count, grid_width - 1);
        for(int i = 0; i < grid_height; i++){
            for(int j = 0; j < grid_width; j++){
                target_Ind[i * grid_width + j] = static_cast<double>(j);
            }
        }
        ct_x_mask = getGridMask(ct_x_mask, grid_width, target_Ind);
    }
    Ciphertext<Scheme::CKKS> ct_y_mask = y_scale;
    {
        std::vector<double> target_Ind(slot_count, grid_height - 1);
        for(int i = 0; i < grid_height; i++){
            for(int j = 0; j < grid_width; j++){
                target_Ind[i * grid_width + j] = static_cast<double>(i);
            }
        }
        ct_y_mask = getGridMask(ct_y_mask, grid_height, target_Ind);
    }
    evaluator_.multiply(ct_x_mask, ct_y_mask);
    Ciphertext<Scheme::CKKS> ct_result = grid;
    {
        evaluator_.multiply(ct_result, ct_x_mask);
        int Sum_Num = grid_width * grid_height;
        int log_step = 0;
        while ((1 << log_step) < Sum_Num) {
            log_step++;
        }
        int max_len = pow(2, log_step);
        ct_result = cipher_sum(ct_result, max_len, log_step);
    }
    return ct_result;
}

Ciphertext<Scheme::CKKS> NonLinearExperi::getGridMask(
    const Ciphertext<Scheme::CKKS>& scale, int grid_Size, std::vector<double>& target_Ind){
    Ciphertext<Scheme::CKKS> ct_scaled = scale;
    {
        // transpose scale from [-1, 1] to [0, grid_Num - 1]
        Plaintext<Scheme::CKKS> pt_one(ctx_), pt_half(ctx_), pt_grid_Size(ctx_);
        encoder_.encode(pt_one, std::vector<double>(slot_count, 1.0), scale_);
        encoder_.encode(pt_half, std::vector<double>(slot_count, 0.5), scale_);
        encoder_.encode(pt_grid_Size, std::vector<double>(slot_count, (double)(grid_Size - 1)), scale_);
        evaluator_.add(ct_scaled, pt_one);
        evaluator_.multiply(ct_scaled, pt_half);
        evaluator_.multiply(ct_scaled, pt_grid_Size);
    }
    std::vector<double> targed_Ind_plus_half(slot_count), targed_Ind_minus_half(slot_count);
    {
        std::transform(target_Ind.begin(), target_Ind.end(), targed_Ind_plus_half.begin(),
            [](double val) { return val + 0.5; });
        std::transform(target_Ind.begin(), target_Ind.end(), targed_Ind_minus_half.begin(),
            [](double val) { return val - 0.5; });
    }
    Ciphertext<Scheme::CKKS> ct_bigger_mask = ct_scaled;
    {
        Plaintext<Scheme::CKKS> pt_ind(ctx_);
        encoder_.encode(pt_ind, targed_Ind_minus_half, scale_);
        evaluator_.sub(ct_bigger_mask, pt_ind);

        Plaintext<Scheme::CKKS> pt_norm(ctx_);
        encoder_.encode(pt_norm, std::vector<double>(slot_count, 1.0 / ((double)(grid_Size - 1))), scale_);
        evaluator_.multiply(ct_bigger_mask, pt_norm);
        ct_bigger_mask = sgnEffi(ct_bigger_mask, 2, 2, 0.5, 0.5);
    }
    Ciphertext<Scheme::CKKS> ct_smaller_mask = ct_scaled;
    {
        evaluator_.negate(ct_smaller_mask);
        Plaintext<Scheme::CKKS> pt_ind(ctx_);
        encoder_.encode(pt_ind, targed_Ind_plus_half, scale_);
        evaluator_.add(ct_smaller_mask, pt_ind);

        Plaintext<Scheme::CKKS> pt_norm(ctx_);
        encoder_.encode(pt_norm, std::vector<double>(slot_count, 1.0 / ((double)(grid_Size - 1))), scale_);
        evaluator_.multiply(ct_smaller_mask, pt_norm);
        ct_smaller_mask = sgnEffi(ct_smaller_mask, 2, 2, 0.5, 0.5);
    }
    evaluator_.multiply(ct_bigger_mask, ct_smaller_mask);
    return ct_bigger_mask;
}

// -------------------- Our TopK Gather -------------------------
void NonLinearExperi::TopkGatherOurs(const Ciphertext<Scheme::CKKS>& query_scores, const std::vector<Ciphertext<Scheme::CKKS>>& query_coords,
    int Num, int Topk, std::vector<Ciphertext<Scheme::CKKS>>& ct_outputs){
    // Note: The res of getCipherRank is (Num-0.5) for top-1, (Num-1.5) for top-2, ..., (0.5) for top-Num
    Ciphertext<Scheme::CKKS> ct_rank = getCipherRank(query_scores, Num);
    std::vector<Ciphertext<Scheme::CKKS>> ct_temp_result(4);
    getRankCoord(ct_rank, query_coords, Num, 0, ct_temp_result);
    for(int coord_idx = 0; coord_idx < 4; coord_idx++){
        ct_outputs[coord_idx] = ct_temp_result[coord_idx];
    }
    for(int i = 1; i < Topk; i++){
        getRankCoord(ct_rank, query_coords, Num, i, ct_temp_result);
        for(int coord_idx = 0; coord_idx < 4; coord_idx++){
            evaluator_.add(ct_outputs[coord_idx], ct_temp_result[coord_idx]);
        }
    }
}

Ciphertext<Scheme::CKKS> NonLinearExperi::getCipherRank(const Ciphertext<Scheme::CKKS>& ct_input, int Num){
    int log2N = 0;
    while ((1 << log2N) < Num)
        log2N++;
    int N = pow(2, log2N);
    if( N*N > slot_count)
        throw std::invalid_argument("Num^2 is larger than slot_count");

    Ciphertext<Scheme::CKKS> ct_left = ct_input, ct_right = ct_input;
    {
        ct_right = transRow(ct_right, N, log2N);
        ct_left = replicateRow(ct_left, N, log2N);
        ct_right = replicateColumn(ct_right, N, log2N);
    }
    Ciphertext<Scheme::CKKS> res(ctx_);
    {
        evaluator_.sub(ct_left, ct_right, res);
        Plaintext<Scheme::CKKS> pt_norm(ctx_);
        encoder_.encode(pt_norm, std::vector<double>(slot_count, 1.0 / ((double)(9))), scale_);
        evaluator_.multiply(res, pt_norm);
        res = sgnEffi(res, 2, 2, 0.5, 0.5);
        res = sumRows(res, N, log2N);
    }
    return res;
}

void NonLinearExperi::getRankCoord(const Ciphertext<Scheme::CKKS>& ct_rank, const std::vector<Ciphertext<Scheme::CKKS>>& query_coords, 
    int Num, int getTopInd, std::vector<Ciphertext<Scheme::CKKS>>& ct_outputs){
    if( getTopInd > Num){
        throw std::invalid_argument("Topk index larger than Num");
    }
    std::vector<double> vec_ind_plus_half(slot_count, static_cast<double>(Num - getTopInd));
    std::vector<double> vec_ind_minus_half(slot_count, static_cast<double>(Num - getTopInd - 1));

    Ciphertext<Scheme::CKKS> ct_bigger_mask = ct_rank;
    {
        Plaintext<Scheme::CKKS> pt_ind(ctx_);
        encoder_.encode(pt_ind, vec_ind_minus_half, scale_);
        evaluator_.sub(ct_bigger_mask, pt_ind);

        Plaintext<Scheme::CKKS> pt_norm(ctx_);
        encoder_.encode(pt_norm, std::vector<double>(slot_count, 1.0 / ((double)(Num))), scale_);
        evaluator_.multiply(ct_bigger_mask, pt_norm);
        ct_bigger_mask = sgnEffi(ct_bigger_mask, 2, 2, 0.5, 0.5);
    }
    Ciphertext<Scheme::CKKS> ct_smaller_mask = ct_rank;
    {
        evaluator_.negate(ct_smaller_mask);
        Plaintext<Scheme::CKKS> pt_ind(ctx_);
        encoder_.encode(pt_ind, vec_ind_plus_half, scale_);
        evaluator_.add(ct_smaller_mask, pt_ind);

        Plaintext<Scheme::CKKS> pt_norm(ctx_);
        encoder_.encode(pt_norm, std::vector<double>(slot_count, 1.0 / ((double)(Num))), scale_);
        evaluator_.multiply(ct_smaller_mask, pt_norm);
        ct_smaller_mask = sgnEffi(ct_smaller_mask, 2, 2, 0.5, 0.5);
    }
    evaluator_.multiply(ct_bigger_mask, ct_smaller_mask);

    int log_step = 0;
    while ((1 << log_step) < Num) {
        log_step++;
    }
    int max_len = pow(2, log_step);
    std::vector<double> vec_mask(slot_count, 0.0);
    vec_mask[getTopInd] = 1.0;
    Plaintext<Scheme::CKKS> pt_mask(ctx_);

    for(int coord_idx = 0; coord_idx < 4; coord_idx++){
        Ciphertext<Scheme::CKKS> target_coord = query_coords[coord_idx];
        evaluator_.multiply(target_coord, ct_bigger_mask);
        target_coord = cipher_sum(target_coord, max_len, log_step);
        encoder_.encode(pt_mask, vec_mask, scale_);
        evaluator_.multiply(target_coord, pt_mask);
        ct_outputs[coord_idx] = target_coord;
    }
}
