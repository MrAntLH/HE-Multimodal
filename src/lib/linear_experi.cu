#include "linear_experi.cuh"
#include <cmath>
#include <algorithm>
#include <iostream>
#include <string>

int LinearExperi::floor_power_of_two_loop(int input_num) {
    if (input_num < 1) {
        throw std::invalid_argument("input_num must ≥1");
    }
    int result = 1;
    while (result * 2 <= input_num) {
        result <<= 1;
    }
    return result;
}

Ciphertext<Scheme::CKKS> LinearExperi::pt_ct_matmul(Ciphertext<Scheme::CKKS>& ct, const std::vector<std::vector<double>>& weights,
    const std::vector<double>& bias, int input_dim, int hidden_dim, int output_dim){
    // ========== 1. 参数校验  ==================
    if (input_dim * hidden_dim >= slot_count) {
        throw std::invalid_argument("pt_ct_matmul error: input_dim * hidden_dim > slot_count ");
    }
    if (weights.empty() || weights.size() < hidden_dim || weights[0].size() < output_dim) {
        throw std::invalid_argument("pt_ct_matmul error: Invalid weights matrix dimensions ");
    }
    if (bias.empty() || bias.size() < output_dim) {
        throw std::invalid_argument("pt_ct_matmul error: Invalid bias dimensions ");
    }
    // =========== 2. 矩阵乘法 ===================
    // 为了防止爆显存，此处线性计算，后续可优化并行
    Ciphertext<Scheme::CKKS> result_ct;
    for(int output_index = 0; output_index < output_dim; output_index++){
        // =========== 2.1 重复打包每列，然后相乘 ===================
        std::vector<double> vec_weight(slot_count, 0.0);
        for(int input_index = 0; input_index < input_dim; input_index++){
            for(int hidden_index = 0; hidden_index < hidden_dim; hidden_index++){
                vec_weight[input_index * hidden_dim + hidden_index] = weights[hidden_index][output_index];
            }
        }
        Plaintext<Scheme::CKKS> pt_diag_weight;
        encoder_.encode(pt_diag_weight, vec_weight, scale_);
        Ciphertext<Scheme::CKKS> temp_mul_ct;
        evaluator_.multiply(ct, pt_diag_weight, temp_mul_ct);
        // =========== 2.2 局部旋转累加 ===================
        if(hidden_dim > 1){
            Ciphertext<Scheme::CKKS> temp_rot_ct;
            int exp_rot_step = floor_power_of_two_loop(hidden_dim);
            if(exp_rot_step < hidden_dim){
                Ciphertext<Scheme::CKKS> temp_add_ct = temp_mul_ct;
                std::vector<double> vec_mask(slot_count, 0.0);
                for(int i = 0; i < input_dim; i++){
                    for(int j = exp_rot_step; j < hidden_dim; j++){
                        vec_mask[i * hidden_dim + j] = 1.0;
                    }
                }
                Plaintext<Scheme::CKKS> pt_mask;
                encoder_.encode(pt_mask, vec_mask, scale_);
                evaluator_.multiply(temp_add_ct, pt_mask);
                evaluator_.rotate_rows(temp_add_ct, hidden_dim - exp_rot_step);
                evaluator_.add(temp_mul_ct, temp_add_ct);
            }
            while(exp_rot_step > 1){
                evaluator_.rotate_rows(temp_mul_ct, exp_rot_step / 2, temp_rot_ct);
                evaluator_.add(temp_mul_ct, temp_rot_ct);
                exp_rot_step /= 2;
            }
        }
        // =========== 2.3 维度不同时，变形匹配 ===================
        Ciphertext<Scheme::CKKS> temp_align_ct;
        if( hidden_dim == output_dim){
            std::vector<double> vec_mask(slot_count, 0.0);
            for(int i = 0; i < input_dim; i++){
                vec_mask[i * hidden_dim] = 1.0;
            }
            Plaintext<Scheme::CKKS> pt_mask;
            encoder_.encode(pt_mask, vec_mask, scale_);
            evaluator_.multiply(temp_mul_ct, pt_mask, temp_align_ct);
        }else{
            std::vector<double> vec_mask(slot_count, 0.0);
            vec_mask[0] = 1.0;
            Plaintext<Scheme::CKKS> pt_mask;
            encoder_.encode(pt_mask, vec_mask, scale_);
            vec_mask[0] = 0.0;
            evaluator_.multiply(temp_mul_ct, pt_mask, temp_align_ct);
            for(int i = 1; i < input_dim; i++){
                Ciphertext<Scheme::CKKS> temp_rot_ct;
                evaluator_.rotate_rows(temp_mul_ct, i * (hidden_dim - output_dim), temp_rot_ct);
                vec_mask[i * output_dim] = 1.0;
                encoder_.encode(pt_mask, vec_mask, scale_);
                evaluator_.multiply(temp_rot_ct, pt_mask);
                vec_mask[i * output_dim] = 0.0;
                evaluator_.add(temp_align_ct, temp_rot_ct);
            }
        }
        // =========== 2.4 最终求和 ===================
        if(output_index == 0){
            result_ct = temp_align_ct;
        } else {
            Ciphertext<Scheme::CKKS> rotated_ct;
            evaluator_.rotate_rows(temp_align_ct, -output_index, rotated_ct);
            evaluator_.add(result_ct, rotated_ct);
        }
    }
    // =========== 3. 加偏置 ===================
    std::vector<double> vec_bias(slot_count, 0.0);
    for(int i = 0; i < input_dim; i++)
        for(int j = 0; j < output_dim; j++)
            vec_bias[i * output_dim + j] = bias[j];
    Plaintext<Scheme::CKKS> pt_bias;
    encoder_.encode(pt_bias, vec_bias, scale_);
    evaluator_.add(result_ct, pt_bias);

    return result_ct;
}

Ciphertext<Scheme::CKKS> LinearExperi::ct_ct_attenscore(Ciphertext<Scheme::CKKS>& ct_q, 
    Ciphertext<Scheme::CKKS>& ct_k_t, int seq_dim, int hidden_dim){
    // ========== 1. 参数校验  ==================
    if (seq_dim * hidden_dim >= slot_count) {
        throw std::invalid_argument("ct_ct_attenscore error: seq_dim * hidden_dim > slot_count ");
    }
    // =========== 2. 矩阵乘法 ===================
    Ciphertext<Scheme::CKKS> result_ct;
    // 为了防止爆显存，此处线性计算，后续可优化并行
    for(int seq_index = 0; seq_index < seq_dim; seq_index++){
        // =========== 2.1 K 矩阵按列打包构造 ===================
        Ciphertext<Scheme::CKKS> ct_ind_col = ct_k_t;
        {
            if(seq_index > 0){
                evaluator_.rotate_rows(ct_ind_col, seq_index * hidden_dim);
            }
            std::vector<double> vec_mask(slot_count, 0.0);
            for(int i = 0; i < hidden_dim; i++)
                vec_mask[i] = 1.0;
            Plaintext<Scheme::CKKS> pt_mask;
            encoder_.encode(pt_mask, vec_mask, scale_);
            evaluator_.multiply(ct_ind_col, pt_mask);
            // Repeat for seq_dim times
            int exp_repeate = floor_power_of_two_loop(seq_dim);
            int normal_repeate = seq_dim - exp_repeate;
            while(exp_repeate > 1){
                Ciphertext<Scheme::CKKS> temp_rot_ct;
                evaluator_.rotate_rows(ct_ind_col, -exp_repeate / 2 * hidden_dim, temp_rot_ct);
                evaluator_.add(ct_ind_col, temp_rot_ct);
                exp_repeate /= 2;
            }
            if(normal_repeate > 0){
                std::vector<double> temp_mask(slot_count, 0.0);
                for(int i = 0; i < normal_repeate * hidden_dim; i++)
                    temp_mask[i] = 1.0;
                Plaintext<Scheme::CKKS> pt_temp_mask;
                encoder_.encode(pt_temp_mask, temp_mask, scale_);
                Ciphertext<Scheme::CKKS> ct_temp = ct_ind_col;
                evaluator_.multiply(ct_temp, pt_temp_mask);
                evaluator_.rotate_rows(ct_temp, -exp_repeate * hidden_dim);
                evaluator_.add(ct_ind_col, ct_temp);
            }
        }
        // =========== 2.2 Q 与 K该列 相乘 ===================
        Ciphertext<Scheme::CKKS> ct_temp_mul = ct_q;
        evaluator_.multiply(ct_temp_mul, ct_ind_col);
        // =========== 2.3 局部旋转累加 ===================
        if(hidden_dim > 1){
            Ciphertext<Scheme::CKKS> temp_rot_ct;
            int exp_rot_step = floor_power_of_two_loop(hidden_dim);
            if(exp_rot_step < hidden_dim){
                Ciphertext<Scheme::CKKS> temp_add_ct = ct_temp_mul;
                std::vector<double> vec_mask(slot_count, 0.0);
                for(int i = 0; i < seq_dim; i++){
                    for(int j = exp_rot_step; j < hidden_dim; j++){
                        vec_mask[i * hidden_dim + j] = 1.0;
                    }
                }
                Plaintext<Scheme::CKKS> pt_mask;
                encoder_.encode(pt_mask, vec_mask, scale_);
                evaluator_.multiply(temp_add_ct, pt_mask);
                evaluator_.rotate_rows(temp_add_ct, hidden_dim - exp_rot_step);
                evaluator_.add(ct_temp_mul, temp_add_ct);
            }
            while(exp_rot_step > 1){
                evaluator_.rotate_rows(ct_temp_mul, exp_rot_step / 2, temp_rot_ct);
                evaluator_.add(ct_temp_mul, temp_rot_ct);
                exp_rot_step /= 2;
            }
        }
        // =========== 2.4 维度不同时，变形匹配 ===================
        Ciphertext<Scheme::CKKS> temp_align_ct;
        if( hidden_dim == seq_dim){
            std::vector<double> vec_mask(slot_count, 0.0);
            for(int i = 0; i < seq_dim; i++){
                vec_mask[i * hidden_dim] = 1.0;
            }
            Plaintext<Scheme::CKKS> pt_mask;
            encoder_.encode(pt_mask, vec_mask, scale_);
            evaluator_.multiply(ct_temp_mul, pt_mask, temp_align_ct);
        }else{
            std::vector<double> vec_mask(slot_count, 0.0);
            vec_mask[0] = 1.0;
            Plaintext<Scheme::CKKS> pt_mask;
            encoder_.encode(pt_mask, vec_mask, scale_);
            vec_mask[0] = 0.0;
            evaluator_.multiply(ct_temp_mul, pt_mask, temp_align_ct);
            for(int i = 1; i < seq_dim; i++){
                Ciphertext<Scheme::CKKS> temp_rot_ct;
                evaluator_.rotate_rows(ct_temp_mul, i * (hidden_dim - seq_dim), temp_rot_ct);
                vec_mask[i * seq_dim] = 1.0;
                encoder_.encode(pt_mask, vec_mask, scale_);
                evaluator_.multiply(temp_rot_ct, pt_mask);
                vec_mask[i * seq_dim] = 0.0;
                evaluator_.add(temp_align_ct, temp_rot_ct);
            }
        }
        // =========== 2.4 最终求和 ===================
        if(seq_index == 0){
            result_ct = temp_align_ct;
        } else {
            Ciphertext<Scheme::CKKS> rotated_ct;
            evaluator_.rotate_rows(temp_align_ct, -seq_index, rotated_ct);
            evaluator_.add(result_ct, rotated_ct);
        }
    }
    return result_ct;
}

Plaintext<Scheme::CKKS> LinearExperi::pack_matrix_by_row(const std::vector<std::vector<double>>& mat,
    int target_rows, int target_cols){
    if(mat.empty() || mat[0].empty()){
        throw std::invalid_argument("pack_matrix_by_row error: 输入矩阵不能为空");
    }
    if (mat.size() > target_rows || mat[0].size() > target_cols) {
        throw std::invalid_argument("pack_matrix_by_row error: 输入矩阵维度大于目标打包维度");
    }

    if( target_rows * target_cols > slot_count ){
        throw std::invalid_argument("pack_matrix_by_row error: 目标打包维度乘积大于slot_count，无法存储");
    }
    std::vector<double> packed_vec(slot_count, 0.0);
    for(int i = 0; i < mat.size(); i++){
        for(int j = 0; j < mat[0].size(); j++){
            packed_vec[i * target_cols + j] = mat[i][j];
        }
    }
    Plaintext<Scheme::CKKS> pt(ctx_);
    encoder_.encode(pt, packed_vec, scale_);
    return pt;
}

std::vector<std::vector<double>> LinearExperi::unpack_matrix_by_row(Plaintext<Scheme::CKKS> pack,
    int pack_rows, int pack_cols, int target_rows, int target_cols){
    if(target_rows > pack_rows || target_cols > pack_cols || pack_rows * pack_cols > slot_count){
        throw std::invalid_argument("unpack_matrix_by_row error: 输入参数有误");
    }

    std::vector<double> packed_vec(slot_count, 0.0);
    encoder_.decode(packed_vec, pack);
    std::vector<std::vector<double>> packed_mat(target_rows, std::vector<double>(target_cols, 0.0));
    for(int i = 0; i < target_rows; i++){
        for(int j = 0; j < target_cols; j++){
            packed_mat[i][j] = packed_vec[i * pack_cols + j];
        }
    }

    return packed_mat;
}

