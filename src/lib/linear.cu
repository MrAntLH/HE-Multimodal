#include "linear.cuh"

Ciphertext<Scheme::CKKS> Linear::ctpt_matmul_to_row(Ciphertext<Scheme::CKKS>& ct, 
    const std::vector<std::vector<double>>& weights, const std::vector<double>& bias, 
    int input_dim, int hidden_dim){
    // ========== 1. 参数校验  ==================
    if (input_dim * hidden_dim > slot_count) {
        throw std::invalid_argument("pt_ct_matmul error: input_dim * hidden_dim > slot_count ");
    }
    if (weights.empty() || weights.size() < hidden_dim || weights[0].size() < hidden_dim) {
        throw std::invalid_argument("pt_ct_matmul error: Invalid weights matrix dimensions ");
    }
    if (bias.empty() || bias.size() < hidden_dim) {
        throw std::invalid_argument("pt_ct_matmul error: Invalid bias dimensions ");
    }
    // =========== 2. 矩阵乘法 ===================
    // 为了防止爆显存，此处线性计算，后续可优化并行
    Ciphertext<Scheme::CKKS> result_ct;
    for(int output_index = 0; output_index < hidden_dim; output_index++){
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
            Ciphertext<Scheme::CKKS> temp_rot_ct;
            while(exp_rot_step > 1){
                evaluator_.rotate_rows(temp_mul_ct, exp_rot_step / 2, temp_rot_ct);
                evaluator_.add(temp_mul_ct, temp_rot_ct);
                exp_rot_step /= 2;
            }
        }
        // =========== 2.3 添加 mask ===================
        Ciphertext<Scheme::CKKS> temp_align_ct;
        {
            std::vector<double> vec_mask(slot_count, 0.0);
            for(int i = 0; i < input_dim; i++){
                vec_mask[i * hidden_dim] = 1.0;
            }
            Plaintext<Scheme::CKKS> pt_mask;
            encoder_.encode(pt_mask, vec_mask, scale_);
            evaluator_.multiply(temp_mul_ct, pt_mask, temp_align_ct);
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
        for(int j = 0; j < hidden_dim; j++)
            vec_bias[i * hidden_dim + j] = bias[j];
    Plaintext<Scheme::CKKS> pt_bias;
    encoder_.encode(pt_bias, vec_bias, scale_);
    evaluator_.add(result_ct, pt_bias);

    return result_ct;
}

Ciphertext<Scheme::CKKS> Linear::ctct_attenscore(Ciphertext<Scheme::CKKS>& ct_q, 
    Ciphertext<Scheme::CKKS>& ct_kt, int seq_dim, int hidden_dim, int head_num){
    const int head_size = hidden_dim / head_num;
    // ========== 1. 参数校验  ==================
    if (hidden_dim % head_num != 0) {
        throw std::invalid_argument("[ct_ct_attenscore] error: hidden_dim % head_num != 0.");
    }
    if (seq_dim * hidden_dim > slot_count) {
        throw std::invalid_argument("[ct_ct_attenscore] error: seq_dim * hidden_dim > slot_count ");
    }
    if (seq_dim > head_size) {
        throw std::invalid_argument("[ct_ct_attenscore] error: seq_dim > hidden_dim/head_num ");
    }
    // =========== 2. 矩阵乘法 ===================
    Ciphertext<Scheme::CKKS> result_ct;
    // 为了防止爆显存，此处线性计算，后续可优化并行
    for(int seq_index = 0; seq_index < seq_dim; seq_index++){
        // =========== 2.1 KT 矩阵按列打包构造 ===================
        Ciphertext<Scheme::CKKS> ct_ind_col = ct_kt;
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
            const int normal_repeate = seq_dim - exp_repeate;
            const int exp_repeate_len = exp_repeate * hidden_dim;
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
                evaluator_.rotate_rows(ct_temp, -exp_repeate_len);
                evaluator_.add(ct_ind_col, ct_temp);
            }
        }
        // =========== 2.2 Q 与 K 该列 相乘 ===================
        Ciphertext<Scheme::CKKS> ct_temp_mul = ct_q;
        evaluator_.multiply(ct_temp_mul, ct_ind_col);
        // =========== 2.3 旋转局部累加 ===================
        if(head_size > 1){
            Ciphertext<Scheme::CKKS> temp_rot_ct;
            int exp_rot_step = floor_power_of_two_loop(head_size);
            // 不足指数的部分，先旋转并求和
            if(exp_rot_step < head_size){
                Ciphertext<Scheme::CKKS> temp_add_ct = ct_temp_mul;
                std::vector<double> vec_mask(slot_count, 0.0);
                for(int seq_ind = 0; seq_ind < seq_dim; seq_dim++)
                    for(int head_ind = 0; head_ind < head_size; head_ind++)
                        for(int j = exp_rot_step; j < head_size; j++)
                            vec_mask[seq_ind * hidden_dim + head_ind * head_size + j] = 1.0;
                Plaintext<Scheme::CKKS> pt_mask;
                encoder_.encode(pt_mask, vec_mask, scale_);
                evaluator_.multiply(temp_add_ct, pt_mask);
                evaluator_.rotate_rows(temp_add_ct, head_size - exp_rot_step);
                evaluator_.add(ct_temp_mul, temp_add_ct);
            }
            while(exp_rot_step > 1){
                evaluator_.rotate_rows(ct_temp_mul, exp_rot_step / 2, temp_rot_ct);
                evaluator_.add(ct_temp_mul, temp_rot_ct);
                exp_rot_step /= 2;
            }
        }
        // =========== 2.4 块内累加后应用mask，仅保留每块头部元素 ============
        Ciphertext<Scheme::CKKS> temp_align_ct;
        {
            std::vector<double> vec_mask(slot_count, 0.0);
            for(int seq_ind = 0; seq_ind < seq_dim; seq_ind++){
                for(int head_ind = 0; head_ind < head_size; head_ind++){
                    vec_mask[seq_ind * hidden_dim + head_ind * head_size] = 1.0;
                }
            }
            Plaintext<Scheme::CKKS> pt_mask;
            encoder_.encode(pt_mask, vec_mask, scale_);
            evaluator_.multiply(ct_temp_mul, pt_mask, temp_align_ct);
        }
        // =========== 2.5 最终求和 ===================
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

std::vector<Ciphertext<Scheme::CKKS>> Linear::ctpt_matmul_to_col(
    Ciphertext<Scheme::CKKS>& input_ct, const std::vector<std::vector<double>>& weights, 
    const std::vector<double>& bias, const int seq_len, const int head_num, const int head_size){
    // ========== 1. 参数校验  ==================
    if( seq_len > head_size )
        throw std::invalid_argument("[ctpt_matmul_to_col] error: Not support seq_len > head_size");
    const int hidden_dim = head_num * head_size;
    const int per_ct_cols = std::gcd(seq_len, head_size); // 列打包，每个密文中存 per_ct_cols 列, 即均分存储
    const int out_ct_nums = head_size / per_ct_cols;
    // =========== 2. 矩阵乘法 ===================
    // 为了防止爆显存，此处线性计算，后续可优化并行
    std::vector<Ciphertext<Scheme::CKKS>> res_cts(out_ct_nums);
    for(int seq_ind = 0; seq_ind < seq_len; seq_ind++){
        // 2.1 先构造 行重复打包
        Ciphertext<Scheme::CKKS> ct_ind_row = input_ct;
        {
            std::vector<double> vec_mask(slot_count, 0.0);
            for(int hid_ind = 0; hid_ind < hidden_dim; hid_ind++)
                vec_mask[seq_ind * hidden_dim + hid_ind] = 1.0;
            Plaintext<Scheme::CKKS> pt_mask;
            encoder_.encode(pt_mask, vec_mask, scale_);
            evaluator_.multiply(ct_ind_row, pt_mask);
            Ciphertext<Scheme::CKKS> masked_ct_ind_row = ct_ind_row;
            for(int repead_ind = 0; repead_ind < seq_len; repead_ind++){
                if(repead_ind != seq_ind){
                    Ciphertext<Scheme::CKKS> temp_rot_ct;
                    evaluator_.rotate_rows(masked_ct_ind_row, (seq_ind - repead_ind) * hidden_dim, temp_rot_ct);
                    evaluator_.add(ct_ind_row, temp_rot_ct);
                }
            }
        }
        // 2.2 遍历结果密文，即 每个密文中存储 head_num * seq_len 列
        for(int out_ct_ind = 0; out_ct_ind < out_ct_nums; out_ct_ind++){
            Ciphertext<Scheme::CKKS> per_out_ct;
            // 2.2.1 遍历每个头
            for(int head_ind = 0; head_ind < head_num; head_ind++){
                Ciphertext<Scheme::CKKS> per_head_ct;
                // 乘以按列编码权重
                {
                    std::vector<double> vec_weight(slot_count, 0.0);
                    for(int col_ind = 0; col_ind < per_ct_cols; col_ind++){
                        int hidden_index = head_ind * head_size + out_ct_ind * per_ct_cols + col_ind;
                        for(int seq_ind = 0; seq_ind < hidden_dim; seq_ind++){
                            vec_weight[col_ind * hidden_dim + seq_ind] = weights[seq_ind][hidden_index];
                        }
                    }
                    Plaintext<Scheme::CKKS> pt_weight;
                    encoder_.encode(pt_weight, vec_weight, scale_);
                    evaluator_.multiply(ct_ind_row, pt_weight, per_head_ct);
                }
                // 旋转累加
                if(hidden_dim > 1){
                    Ciphertext<Scheme::CKKS> temp_rot_ct;
                    int exp_rot_step = floor_power_of_two_loop(hidden_dim);
                    // 不足指数的部分，先旋转并求和
                    if(exp_rot_step < hidden_dim){
                        Ciphertext<Scheme::CKKS> temp_add_ct = per_head_ct;
                        std::vector<double> vec_mask(slot_count, 0.0);
                        for(int seq_ind = 0; seq_ind < seq_len; seq_ind++)
                            for(int j = exp_rot_step; j < hidden_dim; j++)
                                vec_mask[seq_ind * hidden_dim + j] = 1.0;
                        Plaintext<Scheme::CKKS> pt_mask;
                        encoder_.encode(pt_mask, vec_mask, scale_);
                        evaluator_.multiply(temp_add_ct, pt_mask);
                        evaluator_.rotate_rows(temp_add_ct, hidden_dim - exp_rot_step);
                        evaluator_.add(per_head_ct, temp_add_ct);
                    }
                    while(exp_rot_step > 1){
                        evaluator_.rotate_rows(per_head_ct, exp_rot_step / 2, temp_rot_ct);
                        evaluator_.add(per_head_ct, temp_rot_ct);
                        exp_rot_step /= 2;
                    }
                }
                // 作用掩码
                {
                    std::vector<double> vec_mask(slot_count, 0.0);
                    for(int col_ind = 0; col_ind < per_ct_cols; col_ind++)
                        vec_mask[col_ind * hidden_dim] = 1.0;
                    Plaintext<Scheme::CKKS> pt_mask;
                    encoder_.encode(pt_mask, vec_mask, scale_);
                    evaluator_.multiply(per_head_ct, pt_mask);
                }
                // 添加 Bias
                {
                    std::vector<double> vec_bias(slot_count, 0.0);
                    for(int col_ind = 0; col_ind < per_ct_cols; col_ind++){
                        int hidden_index = head_ind * head_size + out_ct_ind * per_ct_cols + col_ind;
                        vec_bias[col_ind * hidden_dim] = bias[hidden_index];
                    }
                    Plaintext<Scheme::CKKS> pt_bias;
                    encoder_.encode(pt_bias, vec_bias, scale_);
                    evaluator_.add(per_head_ct, pt_bias);
                }
                // 旋转对齐
                // if(seq_ind == 0 && head_ind == 0){
                //     res_cts[out_ct_ind] = per_head_ct;
                // }else{
                //     evaluator_.rotate_rows(per_head_ct, - (head_ind * head_size + seq_ind) );
                //     evaluator_.add(res_cts[out_ct_ind], per_head_ct);
                // }
                if(head_ind == 0){
                    per_out_ct = per_head_ct;
                }else{
                    evaluator_.rotate_rows(per_head_ct, -head_ind * head_size);
                    evaluator_.add(per_out_ct, per_head_ct);
                }
            }
            if(seq_ind == 0){
                res_cts[out_ct_ind] = per_out_ct;
            }else{
                evaluator_.rotate_rows(per_out_ct, -seq_ind);
                evaluator_.add(res_cts[out_ct_ind], per_out_ct);
            }
        }
    }
    return res_cts;
}

Ciphertext<Scheme::CKKS> Linear::ctct_attenOutLinear(
    std::vector<Ciphertext<Scheme::CKKS>>& cts_atten_score, 
    std::vector<std::vector<Ciphertext<Scheme::CKKS>>>& v_cts, 
    const int input_ct_nums, const int per_seq_len, const int head_num, const int head_size){
    const int hidden_dim = head_num * head_size;
    const int v_block_num = v_cts[0].size();
    const int per_ct_cols = head_size / v_block_num;
    
    // ========== 1. 参数校验  ==================
    if( cts_atten_score.size() != input_ct_nums || v_cts.size() != input_ct_nums)
        throw std::invalid_argument("[ctct_attenOutLinear] error: 密文维度有误!");

    // ========== 2. 受矩阵分块影响，累加 input_ct_nums 次的矩阵 ==================
    Ciphertext<Scheme::CKKS> output_ct;
    for(int block_ind = 0; block_ind < input_ct_nums; block_ind++){
        Ciphertext<Scheme::CKKS> res_ct_block;
        for(int v_block_ind = 0; v_block_ind < v_block_num; v_block_ind++){
            for(int col_ind = 0; col_ind < per_ct_cols; col_ind++){
                // 2.1 V 矩阵列重复
                Ciphertext<Scheme::CKKS> per_col_ct = cts_atten_score[block_ind];
                {
                    Ciphertext<Scheme::CKKS> v_col_ct = v_cts[block_ind][v_block_ind];
                    std::vector<double> vec_mask(slot_count, 0.0);
                    for(int hid_ind = 0; hid_ind < hidden_dim; hid_ind++)
                        vec_mask[col_ind * hidden_dim + hid_ind] = 1.0;
                    Plaintext<Scheme::CKKS> pt_mask;
                    encoder_.encode(pt_mask, vec_mask, scale_);
                    evaluator_.multiply(v_col_ct, pt_mask);
                    Ciphertext<Scheme::CKKS> repeated_ct = v_col_ct;
                    // 重复 per_seq_len 次
                    for(int repead_ind = 0; repead_ind < per_seq_len; repead_ind++){
                        if(repead_ind != col_ind){
                            Ciphertext<Scheme::CKKS> temp_rot_ct;
                            evaluator_.rotate_rows(v_col_ct, (col_ind - repead_ind) * hidden_dim, temp_rot_ct);
                            evaluator_.add(repeated_ct, temp_rot_ct);
                        }
                    }
                    evaluator_.multiply(per_col_ct, repeated_ct);
                }
                // 2.2 块内旋转累加
                if(per_seq_len > 1){
                    int exp_rot_step = floor_power_of_two_loop(per_seq_len);
                    if(exp_rot_step < per_seq_len){
                        Ciphertext<Scheme::CKKS> temp_add_ct = per_col_ct;
                        std::vector<double> vec_mask(slot_count, 0.0);
                        for(int seq_ind = 0; seq_ind < per_seq_len; seq_ind++)
                            for(int head_ind = 0; head_ind < head_num; head_ind++)
                                for(int j = exp_rot_step; j < per_seq_len; j++)
                                    vec_mask[seq_ind * hidden_dim + head_ind * head_size + j ] = 1.0;
                        Plaintext<Scheme::CKKS> pt_mask;
                        encoder_.encode(pt_mask, vec_mask, scale_);
                        evaluator_.multiply(temp_add_ct, pt_mask);
                        evaluator_.rotate_rows(temp_add_ct, per_seq_len - exp_rot_step);
                        evaluator_.add(per_col_ct, temp_add_ct);
                    }
                    Ciphertext<Scheme::CKKS> temp_rot_ct;
                    while(exp_rot_step > 1){
                        evaluator_.rotate_rows(per_col_ct, exp_rot_step / 2, temp_rot_ct);
                        evaluator_.add(per_col_ct, temp_rot_ct);
                        exp_rot_step /= 2;
                    }
                }
                // 2.3 应用掩码
                {
                    std::vector<double> vec_mask(slot_count, 0.0);
                    for(int seq_ind = 0; seq_ind < per_seq_len; seq_ind++)
                        for(int head_ind = 0; head_ind < head_num; head_ind++)
                            vec_mask[seq_ind * hidden_dim + head_ind * head_size] = 1.0;
                    Plaintext<Scheme::CKKS> pt_mask;
                    encoder_.encode(pt_mask, vec_mask, scale_);
                    evaluator_.multiply(per_col_ct, pt_mask);
                }
                // 2.4 旋转对齐
                if(v_block_ind == 0 && col_ind == 0){
                    res_ct_block = per_col_ct;
                }else{
                    evaluator_.rotate_rows(per_col_ct, - (v_block_ind * per_ct_cols + col_ind) );
                    evaluator_.add(res_ct_block, per_col_ct);
                }
            }
        }
        if(block_ind == 0){
            output_ct = res_ct_block;
        }else{
            evaluator_.add(output_ct, res_ct_block);
        }
    }
    return output_ct;
}

Ciphertext<Scheme::CKKS> Linear::ctct_attenOutLinear(
    Ciphertext<Scheme::CKKS>& cts_atten_score, 
    std::vector<Ciphertext<Scheme::CKKS>>& v_cts, 
    const int per_seq_len, const int head_num, const int head_size){
    const int hidden_dim = head_num * head_size;
    const int v_block_num = v_cts.size();
    const int per_ct_cols = per_seq_len / v_block_num;
    // ========== 1. 参数校验  ==================
    if( v_cts.size() != head_size / per_seq_len )
        throw std::invalid_argument("[ctct_attenOutLinear] error: V密文打包维度有误!");
    // ========== 2. 矩阵分块影响，累加 input_ct_nums 次的矩阵 ==================
    Ciphertext<Scheme::CKKS> output_ct;
    for(int v_block_ind = 0; v_block_ind < v_block_num; v_block_ind++){
        for(int col_ind = 0; col_ind < per_ct_cols; col_ind++){
            // 2.1 V 矩阵列重复
            Ciphertext<Scheme::CKKS> per_col_ct = cts_atten_score;
            {
                Ciphertext<Scheme::CKKS> v_col_ct = v_cts[v_block_ind];
                std::vector<double> vec_mask(slot_count, 0.0);
                for(int hid_ind = 0; hid_ind < hidden_dim; hid_ind++)
                    vec_mask[col_ind * hidden_dim + hid_ind] = 1.0;
                Plaintext<Scheme::CKKS> pt_mask;
                encoder_.encode(pt_mask, vec_mask, scale_);
                evaluator_.multiply(v_col_ct, pt_mask);
                Ciphertext<Scheme::CKKS> repeated_ct = v_col_ct;
                // 重复 per_seq_len 次
                for(int repead_ind = 0; repead_ind < per_seq_len; repead_ind++){
                    if(repead_ind != col_ind){
                        Ciphertext<Scheme::CKKS> temp_rot_ct;
                        evaluator_.rotate_rows(v_col_ct, (col_ind - repead_ind) * hidden_dim, temp_rot_ct);
                        evaluator_.add(repeated_ct, temp_rot_ct);
                    }
                }
                evaluator_.multiply(per_col_ct, repeated_ct);
            }
            // 2.2 块内旋转累加
            if(per_seq_len > 1){
                int exp_rot_step = floor_power_of_two_loop(per_seq_len);
                if(exp_rot_step < per_seq_len){
                    Ciphertext<Scheme::CKKS> temp_add_ct = per_col_ct;
                    std::vector<double> vec_mask(slot_count, 0.0);
                    for(int seq_ind = 0; seq_ind < per_seq_len; seq_ind++)
                        for(int head_ind = 0; head_ind < head_num; head_ind++)
                            for(int j = exp_rot_step; j < per_seq_len; j++)
                                vec_mask[seq_ind * hidden_dim + head_ind * head_size + j ] = 1.0;
                    Plaintext<Scheme::CKKS> pt_mask;
                    encoder_.encode(pt_mask, vec_mask, scale_);
                    evaluator_.multiply(temp_add_ct, pt_mask);
                    evaluator_.rotate_rows(temp_add_ct, per_seq_len - exp_rot_step);
                    evaluator_.add(per_col_ct, temp_add_ct);
                }
                Ciphertext<Scheme::CKKS> temp_rot_ct;
                while(exp_rot_step > 1){
                    evaluator_.rotate_rows(per_col_ct, exp_rot_step / 2, temp_rot_ct);
                    evaluator_.add(per_col_ct, temp_rot_ct);
                    exp_rot_step /= 2;
                }
            }
            // 2.3 应用掩码
            {
                std::vector<double> vec_mask(slot_count, 0.0);
                for(int seq_ind = 0; seq_ind < per_seq_len; seq_ind++)
                    for(int head_ind = 0; head_ind < head_num; head_ind++)
                        vec_mask[seq_ind * hidden_dim + head_ind * head_size] = 1.0;
                Plaintext<Scheme::CKKS> pt_mask;
                encoder_.encode(pt_mask, vec_mask, scale_);
                evaluator_.multiply(per_col_ct, pt_mask);
            }
            // 2.4 旋转对齐
            if(v_block_ind == 0 && col_ind == 0){
                output_ct = per_col_ct;
            }else{
                evaluator_.rotate_rows(per_col_ct, - (v_block_ind * per_ct_cols + col_ind) );
                evaluator_.add(output_ct, per_col_ct);
            }
        }
    }
    return output_ct;
}


Plaintext<Scheme::CKKS> Linear::pack_matrix_by_row(
    const std::vector<std::vector<double>>& mat, int target_rows, int target_cols){
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

std::vector<std::vector<double>> Linear::unpack_matrix_by_row(Plaintext<Scheme::CKKS> pack,
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

int Linear::floor_power_of_two_loop(int input_num) {
    if (input_num < 1) {
        throw std::invalid_argument("input_num must ≥1");
    }
    int result = 1;
    while (result * 2 <= input_num) {
        result <<= 1;
    }
    return result;
}

std::vector<std::vector<double>> Linear::matMul(
    const std::vector<std::vector<double>>& A, const std::vector<std::vector<double>>& B){
    int m = A.size();
    int n = B.size();
    int p = B[0].size();

    if (A.empty() || B.empty() || A[0].size() != n) {
        throw std::invalid_argument("矩阵乘法维度不匹配");
    }

    std::vector<std::vector<double>> result(m, std::vector<double>(p, 0.0));

    for (int i = 0; i < m; ++i) {
        for (int k = 0; k < n; ++k) {
            if (A[i][k] == 0) continue; // 优化：跳过0元素
            for (int j = 0; j < p; ++j) {
                result[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    return result;
}

std::vector<std::vector<double>> Linear::matAddBias(
    const std::vector<std::vector<double>>& A, const std::vector<double>& bias) {
    int m = A.size();
    int d = A[0].size();
    if (d != bias.size()) {
        throw std::invalid_argument("矩阵列数与偏置长度不匹配");
    }
    std::vector<std::vector<double>> result = A;
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < d; ++j) {
            result[i][j] += bias[j];
        }
    }
    return result;
}


