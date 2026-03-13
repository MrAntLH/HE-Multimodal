#include "nonlinear.cuh"
#include <cmath>
#include <algorithm>
#include <iostream>
#include <string>

std::vector<Ciphertext<Scheme::CKKS>> NonLinear::Softmax(
    std::vector<Ciphertext<Scheme::CKKS>>& input_cts, const std::vector<std::vector<bool>>& atten_mask,
     int seq_len, int per_seq_len, int head_num, int head_size, 
     const double& soft_bias, const int& soft_delta_exp, const double& soft_inv_scale){

    const int hidden_dim = head_num * head_size;
    const int ct_num = input_cts.size();
    const double inv_scale_num = 1.0 / soft_inv_scale;

    // 保存中间结果以调试
    // file_.saveCtAttenScore(input_cts, "Ct_Softmax_Input_", 
    //     seq_len, head_num, head_size);

    // ==========  参数校验 ==========
    if( per_seq_len * hidden_dim > slot_count ){
        throw std::invalid_argument("[Softmax] error: 参数超出密文容量!");
    }
    std::vector<Ciphertext<Scheme::CKKS>> output_cts(ct_num);
    Ciphertext<Scheme::CKKS>temp_ct; // 公用的一个临时密文
    // ==========  固定平移 + 缩小区间  ==========
    {   
        // 固定平移，由于softmax具备平移不变性，故一般会减去最大值
            // 但此处为了密文下计算的高效性，选择减去一个固定的值
            // 为了适配性，此处固定值的计算与 seq_len 挂钩
        std::vector<double> vec_const_max(slot_count, 0.0);
        // 直接缩小区间，无需再手动还原，近似计算策略会自动还原
            // 缩小倍数 由两部分组成，4 是由于exp对e^(4x)的拟合，delta_exp是平方次数
        double scale_num = 1.0 / (4 * pow(2, soft_delta_exp));
        std::vector<double> vec_scale(slot_count, 0.0);
        for(int seq_ind = 0; seq_ind < per_seq_len; seq_ind++){
            for(int head_ind = 0; head_ind < head_num; head_ind++){
                for(int factor_ind = 0; factor_ind < per_seq_len; factor_ind++){
                    vec_scale[seq_ind * hidden_dim + head_ind * head_size + factor_ind] = scale_num;
                    vec_const_max[seq_ind * hidden_dim + head_ind * head_size + factor_ind] = soft_bias;
                }
            }
        }
        Plaintext<Scheme::CKKS> pt_const_max(ctx_), pt_scale(ctx_);
        encoder_.encode(pt_const_max, vec_const_max, scale_);
        encoder_.encode(pt_scale, vec_scale, scale_);
        for(int ct_ind = 0; ct_ind < ct_num; ct_ind++){
            Ciphertext<Scheme::CKKS>temp_ct = input_cts[ct_ind];
            evaluator_.sub(temp_ct, pt_const_max);
            evaluator_.multiply(temp_ct, pt_scale);
            output_cts[ct_ind] = temp_ct;
        }
    }
    // ==========  指数近似和应用mask  ==========
    for(int ct_ind = 0; ct_ind < ct_num; ct_ind++){
        temp_ct = exp_ct(output_cts[ct_ind]);
        for(int square_ind = 0; square_ind < soft_delta_exp; square_ind++){
            evaluator_.square(temp_ct);
        }
        std::vector<double> vec_mask(slot_count, 0.0);
        // 注意：此处需要注意，mask是 [per_seq_len, seq_len] 的，需要拆分，但多头共用一个mask
        for(int seq_ind = 0; seq_ind < per_seq_len; seq_ind++){
            for(int factor_ind = 0; factor_ind < per_seq_len; factor_ind++){
                double mask = 1.0;
                if(!atten_mask.empty())
                    mask = atten_mask[seq_ind][ct_ind * per_seq_len + factor_ind];
                for(int head_ind = 0; head_ind < head_num; head_ind++){
                    vec_mask[seq_ind * hidden_dim + head_ind * head_size + factor_ind] = mask;
                }
            }
        }
        Plaintext<Scheme::CKKS> pt_mask(ctx_);
        encoder_.encode(pt_mask, vec_mask, scale_);
        evaluator_.multiply(temp_ct, pt_mask);
        output_cts[ct_ind] = temp_ct;
    }
    // ==========  求和，并近似和的倒数  ==========
    Ciphertext<Scheme::CKKS> ct_inv_sum_exp;
    {
        Ciphertext<Scheme::CKKS> ct_sum_exp;
        for(int ct_ind = 0; ct_ind < ct_num; ct_ind++){
            temp_ct = ct_block_sum(output_cts[ct_ind], per_seq_len, head_num, head_size);
            if(ct_ind == 0){
                ct_sum_exp = temp_ct;
            }else{
                evaluator_.add(ct_sum_exp, temp_ct);
            }
        }
        // 保存中间结果以调试
        // file_.saveCtAttenScore(ct_sum_exp, "Ct_Softmax_exp_sum_",
        //     per_seq_len, head_num, head_size);

        // Note: 由于 inv 的近似需要范围在(0,2)以内，但是指数会放大区间，故此处需要额外的缩放
        //      考虑到 ct_sum_exp 的和与 seq_len 存在关系，故缩放大小通过 seq_len 进行计算
        std::vector<double> vec_mask(slot_count, 0.0);
        for(int seq_ind = 0; seq_ind < per_seq_len; seq_ind++)
            for(int head_ind = 0; head_ind < head_num; head_ind++)
                for(int factor_ind = 0; factor_ind < per_seq_len; factor_ind++)
                    vec_mask[seq_ind * hidden_dim + head_ind * head_size + factor_ind] = inv_scale_num;
        Plaintext<Scheme::CKKS> pt_mask(ctx_);
        encoder_.encode(pt_mask, vec_mask, scale_);

        evaluator_.multiply(ct_sum_exp, pt_mask);
        ct_inv_sum_exp = inverse_ct(ct_sum_exp, 8);
        // 此处的缩放需要还原
        evaluator_.multiply(ct_inv_sum_exp, pt_mask);
    }
    // ==========  依次乘以和的倒数  ==========
    
    for(int ct_ind = 0; ct_ind < ct_num; ct_ind++){
        evaluator_.multiply(output_cts[ct_ind], ct_inv_sum_exp);
    }
    // file_.saveCtAttenScore(output_cts, "Ct_Softmax_Output_", 
    //     seq_len, head_num, head_size);
    return output_cts;
}

Ciphertext<Scheme::CKKS> NonLinear::Softmax(Ciphertext<Scheme::CKKS>& input_ct,
    const std::vector<std::vector<bool>>& atten_mask,
    const int per_seq_len, const int head_num, const int head_size, 
    const double& soft_bias, const int& soft_delta_exp, const double& soft_inv_scale){

    const int hidden_dim = head_num * head_size;
    const double inv_scale_num = 1.0 / soft_inv_scale;

    // 保存中间结果以调试
    // file_.saveCtAttenScore(input_ct, "TextBackbone/Ct/Softmax_Input_", per_seq_len, head_num, head_size);

    // ==========  参数校验 ==========
    if( per_seq_len * hidden_dim > slot_count ){
        throw std::invalid_argument("[Softmax] error: 参数超出密文容量!");
    }
    Ciphertext<Scheme::CKKS> output_ct;
    Ciphertext<Scheme::CKKS> temp_ct; // 公用的一个临时密文
    // ==========  固定平移 + 缩小区间  ==========
    {   
        // 固定平移，由于softmax具备平移不变性，减去一个固定的值
        std::vector<double> vec_const_max(slot_count, 0.0);
        // 直接缩小区间，无需再手动还原，近似计算策略会自动还原
            // 缩小倍数 由两部分组成，4 是由于exp对e^(4x)的拟合，delta_exp是平方次数
        double scale_num = 1.0 / (4 * pow(2, soft_delta_exp));
        std::vector<double> vec_scale(slot_count, 0.0);
        for(int seq_ind = 0; seq_ind < per_seq_len; seq_ind++){
            for(int head_ind = 0; head_ind < head_num; head_ind++){
                for(int factor_ind = 0; factor_ind < per_seq_len; factor_ind++){
                    vec_scale[seq_ind * hidden_dim + head_ind * head_size + factor_ind] = scale_num;
                    vec_const_max[seq_ind * hidden_dim + head_ind * head_size + factor_ind] = soft_bias;
                }
            }
        }   
        Plaintext<Scheme::CKKS> pt_const_max(ctx_), pt_scale(ctx_);
        encoder_.encode(pt_const_max, vec_const_max, scale_);
        encoder_.encode(pt_scale, vec_scale, scale_);
        temp_ct = input_ct;
        evaluator_.sub(temp_ct, pt_const_max);
        evaluator_.multiply(temp_ct, pt_scale);
        output_ct = temp_ct;
    }
    // ==========  指数近似和应用mask  ==========
    {
        temp_ct = exp_ct(output_ct);
        for(int square_ind = 0; square_ind < soft_delta_exp; square_ind++){
            evaluator_.square(temp_ct);
        }
        std::vector<double> vec_mask(slot_count, 0.0);
        for(int seq_ind = 0; seq_ind < per_seq_len; seq_ind++){
            for(int factor_ind = 0; factor_ind < per_seq_len; factor_ind++){
                double mask = 1.0;
                if(!atten_mask.empty())
                    mask = atten_mask[seq_ind][factor_ind];;
                for(int head_ind = 0; head_ind < head_num; head_ind++){
                    vec_mask[seq_ind * hidden_dim + head_ind * head_size + factor_ind] = mask;
                }
            }
        }
        Plaintext<Scheme::CKKS> pt_mask(ctx_);
        encoder_.encode(pt_mask, vec_mask, scale_);
        evaluator_.multiply(temp_ct, pt_mask);
        output_ct = temp_ct;
    }
    // ==========  求和，并近似和的倒数  ==========
    Ciphertext<Scheme::CKKS> ct_inv_sum_exp;
    {
        temp_ct = ct_block_sum(output_ct, per_seq_len, head_num, head_size);

        // 保存中间结果以调试
        // file_.saveCtAttenScore(temp_ct, "TextBackbone/Ct/Softmax_exp_sum_", per_seq_len, head_num, head_size);
        
        // Note: 由于 inv 的近似需要范围在(0,2)以内，但是指数会放大区间，故此处需要额外的缩放
        std::vector<double> vec_mask(slot_count, 0.0);
        for(int seq_ind = 0; seq_ind < per_seq_len; seq_ind++)
            for(int head_ind = 0; head_ind < head_num; head_ind++)
                for(int factor_ind = 0; factor_ind < per_seq_len; factor_ind++)
                    vec_mask[seq_ind * hidden_dim + head_ind * head_size + factor_ind] = inv_scale_num;
        Plaintext<Scheme::CKKS> pt_mask(ctx_);
        encoder_.encode(pt_mask, vec_mask, scale_);

        evaluator_.multiply(temp_ct, pt_mask);
        ct_inv_sum_exp = inverse_ct(temp_ct, 8);
        // 此处的缩放需要还原
        evaluator_.multiply(ct_inv_sum_exp, pt_mask);
    }
    // ==========  依次乘以和的倒数  ==========
    evaluator_.multiply(output_ct, ct_inv_sum_exp);

    return output_ct;
}


Ciphertext<Scheme::CKKS> NonLinear::exp_ct(Ciphertext<Scheme::CKKS>& x){
    Ciphertext<Scheme::CKKS> res = x;
    // The coefficients is y = e^(4x) in [-1,1] 
    std::vector<double> p = { 
            0.9999999945, 3.9999999887, 8.0000007486, 10.6666672345, 
            10.6666501742, 8.5333250358, 5.6890263895, 3.2508477871, 
            1.6248375631, 0.7222123556, 0.2901854042, 0.1054371349, 
            0.0335527478, 0.0103825392, 0.0039797889, 0.0010451185};
    // std::reverse(p.begin(), p.end());
    eval_poly(p, res, res);
    return res;
}

void NonLinear::eval_poly(const std::vector<double> &a, Ciphertext<Scheme::CKKS>& x, 
    Ciphertext<Scheme::CKKS>& dest){

    //  - coefficients a are ordered increasing: a[0] + a[1] x + ... + a[n] x^n
    //  - require degree in [16, 32], i.e. a.size() > 16 and (a.size()-1) <= 32
    //  - otherwise throw std::invalid_argument

    const int n_coeff = static_cast<int>(a.size());  // number of coefficients
    if (n_coeff < 16 || n_coeff > 32) {
        throw std::invalid_argument("eval_poly: degree in (16, 32] required");
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

Ciphertext<Scheme::CKKS> NonLinear::ct_block_sum(Ciphertext<Scheme::CKKS>& input_ct, 
    int per_ct_seq_len, int head_num, int head_size){
    // Note：为进一步优化，此处按照2的整次幂批量操作
    int log_step = 0;
    while ((1 << log_step) < per_ct_seq_len) {
        log_step++;
    }
    int upper_per_ct_seq_len = 1 << log_step;
    if( upper_per_ct_seq_len > head_size )
        throw std::invalid_argument("[ct_block_sum] error : Not support the size!");

    Ciphertext<Scheme::CKKS> output_ct = input_ct;
    const int hidden_dim = head_num * head_size;
    const int total_valid_len = per_ct_seq_len * hidden_dim;
    const int idle_len = slot_count - total_valid_len;
    if( idle_len < 0 )
        throw std::invalid_argument("[ct_block_sum] error : 参数有误");

    // 当富裕一半的空间可进一步优化，利用空位
    if(upper_per_ct_seq_len * 2 <= head_size){
        Ciphertext<Scheme::CKKS> tmp_ct = input_ct;
        // 先旋转，填充后面 upper_per_ct_seq_len 闲置位
        evaluator_.rotate_rows(tmp_ct, -upper_per_ct_seq_len);
        evaluator_.add(output_ct, tmp_ct);
        // 然后向前累加
        for(int i = 0; i < log_step; i++){
            tmp_ct = output_ct;
            int shift = 1 << i;
            evaluator_.rotate_rows(tmp_ct, shift);
            if(shift > idle_len){
                std::vector<double> shift_vec_mask(slot_count, 1.0);
                std::fill(shift_vec_mask.end() - shift, shift_vec_mask.end(), 0.0);
                Plaintext<Scheme::CKKS> shift_pt_mask(ctx_);
                encoder_.encode(shift_pt_mask, shift_vec_mask, scale_);
                evaluator_.multiply(tmp_ct, shift_pt_mask);
            }
            evaluator_.add(output_ct, tmp_ct);
        }
    }else{
        // 1. 先旋转累加至第一个位置
        Ciphertext<Scheme::CKKS> tmp_ct;
        for(int i = 0; i < log_step; i++){
            int shift = 1 << i;
            tmp_ct = output_ct;
            evaluator_.rotate_rows(tmp_ct, shift);
            if(shift > idle_len){
                std::vector<double> shift_vec_mask(slot_count, 1.0);
                std::fill(shift_vec_mask.end() - shift, shift_vec_mask.end(), 0.0);
                Plaintext<Scheme::CKKS> shift_pt_mask(ctx_);
                encoder_.encode(shift_pt_mask, shift_vec_mask, scale_);
                evaluator_.multiply(tmp_ct, shift_pt_mask);
            }
            evaluator_.add(output_ct, tmp_ct);
        }
        // 2. 应用掩码，让其余位置至0
        std::vector<double> vec_mask(slot_count, 0.0);
        for(int seq_ind = 0; seq_ind < per_ct_seq_len; seq_ind++)
            for(int head_ind = 0; head_ind < head_num; head_ind++)
                vec_mask[seq_ind * hidden_dim + head_ind * head_size] = 1.0;
        Plaintext<Scheme::CKKS> pt_mask(ctx_);
        encoder_.encode(pt_mask, vec_mask, scale_);
        evaluator_.multiply(output_ct, pt_mask);
        // 3. 将结果旋转累加后重复至后 upper_per_ct_seq_len
        for(int i = 0; i < log_step; i++){
            int shift = 1 << i;
            tmp_ct = output_ct;
            evaluator_.rotate_rows(tmp_ct, -shift);
            if(shift > idle_len){
                std::vector<double> shift_vec_mask(slot_count, 1.0);
                std::fill(shift_vec_mask.begin(), shift_vec_mask.begin() + shift, 0.0);
                Plaintext<Scheme::CKKS> shift_pt_mask(ctx_);
                encoder_.encode(shift_pt_mask, shift_vec_mask, scale_);
                evaluator_.multiply(tmp_ct, shift_pt_mask);
            }
            evaluator_.add(output_ct, tmp_ct);
        }
    }
    return output_ct;
}

Ciphertext<Scheme::CKKS> NonLinear::inverse_ct(Ciphertext<Scheme::CKKS> x, int iter) {
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
