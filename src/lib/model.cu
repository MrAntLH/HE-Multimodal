#include "model.cuh"

std::vector<Ciphertext<Scheme::CKKS>> Model::bert_ct_opti(
    std::vector<Ciphertext<Scheme::CKKS>>& input_cts,
    const int seq_len, const int hidden_dim, const int num_heads,
    const std::vector<std::vector<bool>>& atten_mask,
    const std::vector<std::vector<double>>& Wq, const std::vector<double>& Bq,
    const std::vector<std::vector<double>>& Wk, const std::vector<double>& Bk,
    const std::vector<std::vector<double>>& Wv, const std::vector<double>& Bv,
    const double soft_bias, const int soft_delta_exp, const double soft_inv_scale,
    std::atomic<long long>* q_time, std::atomic<long long>* k_time, std::atomic<long long>* score_time, 
    std::atomic<long long>* softmax_time, std::atomic<long long>* v_time, std::atomic<long long>* mulV_time){

    const int input_ct_nums = input_cts.size();
    const int per_input_seq_len = seq_len / input_ct_nums;
    const int head_size = hidden_dim / num_heads;

    // 此处利用贪心算法，拼凑多个小块为整块，提高 利用率
    std::vector<std::vector<std::vector<int>>> cut_blocks;
    std::cout << "  ==> 0. 输入拆分 ..." <<  std::endl;
    {
        std::vector<std::vector<int>> input_blocks;
        for( int begin_ind = 0; begin_ind < seq_len; begin_ind++){
            if(!atten_mask[begin_ind][begin_ind])
                break;
            int end_ind = begin_ind;
            while(end_ind + 1 < seq_len && atten_mask[begin_ind][end_ind+1]){
                end_ind++;
            }
            const int part_size = end_ind - begin_ind + 1;
            // 当前实现不支持分块大于单个密文情况，若有需求，补充二次分块逻辑即可
            if(part_size > per_input_seq_len)
                throw std::invalid_argument("[bert_ct_opti] error: 目前不支持当前分块尺寸.");
            input_blocks.push_back(std::vector<int>{part_size, begin_ind, end_ind});
            begin_ind = end_ind;
        }
        std::sort(input_blocks.begin(), input_blocks.end(), 
                [](const std::vector<int>& a, const std::vector<int>& b) {
                    return a[0] > b[0]; 
                });
        while(!input_blocks.empty()){
            std::vector<std::vector<int>> using_block;
            int cap = per_input_seq_len;
            for(auto it = input_blocks.begin(); it != input_blocks.end();){
                int block_len = (*it)[0];
                if(cap >= block_len){
                    using_block.emplace_back(*it);
                    it = input_blocks.erase(it);
                    cap -= block_len;
                }else{
                    ++it;
                }
            }
            cut_blocks.emplace_back(using_block);
        }
    }
    std::vector<Ciphertext<Scheme::CKKS>> cut_cts;
    std::vector<std::vector<std::vector<bool>>> cut_mask;
    // 1. 密文拆分+紧凑组合 + 2.形成 mask
    std::cout << "   => ";
    for(const auto& using_block : cut_blocks){
        int block_ind = 0;
        Ciphertext<Scheme::CKKS> part_ct;
        std::cout << "[ ";
        for(auto it = using_block.begin(); it != using_block.end(); it++){
            const int part_size = (*it)[0], begin_ind = (*it)[1], end_ind = (*it)[2];
            std::cout << begin_ind << "-" << end_ind << " ";
            const int begin_block = begin_ind / per_input_seq_len;
            const int begin_seq = begin_ind % per_input_seq_len;
            const int end_block = end_ind / per_input_seq_len;
            const int end_seq = end_ind % per_input_seq_len;
            int first_ct_len;
            if(end_block > begin_block){
                first_ct_len = per_input_seq_len - begin_seq;
            }else{
                first_ct_len = end_seq - begin_seq + 1;
            }
            Ciphertext<Scheme::CKKS> block_ct = input_cts[begin_block];

            if(begin_seq > 0)
                evaluator_.rotate_rows(block_ct, begin_seq * hidden_dim);
            {
                std::vector<double> vec_mask(slot_count, 0.0);
                for(int seq_ind = 0; seq_ind < first_ct_len; seq_ind++)
                    for(int dim_ind = 0; dim_ind < hidden_dim; dim_ind++)
                        vec_mask[seq_ind * hidden_dim + dim_ind] = 1.0;
                Plaintext<Scheme::CKKS> pt_mask(ctx_);
                encoder_.encode(pt_mask, vec_mask, scale_);
                evaluator_.multiply(block_ct, pt_mask);
            }
            // 如果涉及两块密文，则对齐累加
            if( end_block > begin_block ){
                Ciphertext<Scheme::CKKS> temp_ct = input_cts[end_block];
                std::vector<double> vec_mask(slot_count, 0.0);
                for(int seq_ind = 0; seq_ind <= end_seq; seq_ind++)
                    for(int dim_ind = 0; dim_ind < hidden_dim; dim_ind++)
                        vec_mask[seq_ind * hidden_dim + dim_ind] = 1.0;
                Plaintext<Scheme::CKKS> pt_mask;
                encoder_.encode(pt_mask, vec_mask, scale_);
                evaluator_.multiply(temp_ct, pt_mask);
                evaluator_.rotate_rows(temp_ct, -first_ct_len * hidden_dim);
                evaluator_.add(block_ct, temp_ct);
            }
            if(it == using_block.begin()){
                part_ct = block_ct;
            }else{
                evaluator_.rotate_rows(block_ct, -block_ind * hidden_dim);
                evaluator_.add(part_ct, block_ct);
            }
            block_ind += part_size;
        }
        cut_cts.emplace_back(part_ct);
        std::cout << " ] ";
        std::vector<std::vector<bool>> part_mask(block_ind, std::vector<bool>(block_ind, true));
        if(using_block.size() > 1){
            int temp = 0;
            for(auto it = using_block.begin(); it != using_block.end(); it++){
                const int part_size = (*it)[0];
                std::vector<bool> temp_vec(block_ind, false);
                for(int ind = temp; ind < temp + part_size; ind++)
                    temp_vec[ind] = true;
                for(int ind = temp; ind < temp + part_size; ind++)
                    part_mask[ind] = temp_vec;
                temp += part_size;
            }
        }
        cut_mask.emplace_back(part_mask);
    }
    std::cout << " <= " <<  std::endl;
    // 3. 组合后 执行推理
        // 注：此处可以并行化处理 cut_cts 中的各个密文块，但目前串行实验，暂不实现
        // 同时为了同普通推理过程相同，统一处理Q，K，V的计算
    {
        const int cut_ct_nums = cut_cts.size();
        std::vector<Ciphertext<Scheme::CKKS>> atten_cts(cut_ct_nums);
        {
            // 3.1 计算 Q，K
            std::cout << "  ==> 1. 计算 Q, K 矩阵..." << std::endl;
            std::vector<Ciphertext<Scheme::CKKS>> q_cts(cut_ct_nums);
            {
                CppTimer timer(q_time);
                for(int ct_ind = 0; ct_ind < cut_ct_nums; ct_ind++){
                    q_cts[ct_ind] = linear_.ctpt_matmul_to_row(
                        cut_cts[ct_ind], Wq, Bq, per_input_seq_len, hidden_dim);
                }
            }
            std::vector<Ciphertext<Scheme::CKKS>> k_cts(cut_ct_nums);
            {
                CppTimer timer(k_time);
                for(int ct_ind = 0; ct_ind < cut_ct_nums; ct_ind++){
                    k_cts[ct_ind] = linear_.ctpt_matmul_to_row(
                        cut_cts[ct_ind], Wk, Bk, per_input_seq_len, hidden_dim);
                }
            }
            {
                // 3.2 计算 注意力得分
                CppTimer timer(score_time);
                for(int ct_ind = 0; ct_ind < cut_ct_nums; ct_ind++){
                    atten_cts[ct_ind] = linear_.ctct_attenscore(
                        q_cts[ct_ind], k_cts[ct_ind], per_input_seq_len, hidden_dim, num_heads);
                }
            }
        }
        {
            // 3.3 scale + Softmax
            std::cout << "  ==> 2. 计算 Softmax..." <<  std::endl;
            std::vector<double> vec_mask(slot_count, 0.0);
            double scale_factor = 1.0 / std::sqrt(static_cast<double>(head_size));
            for(int seq_ind = 0; seq_ind < per_input_seq_len; seq_ind++){
                for(int head_ind = 0; head_ind < num_heads; head_ind++){
                    for(int factor_ind = 0; factor_ind < head_size; factor_ind++){
                        vec_mask[seq_ind * hidden_dim + head_ind * head_size + factor_ind] = scale_factor;
                    }
                }
            }
            Plaintext<Scheme::CKKS> pt_scale;
            encoder_.encode(pt_scale, vec_mask, scale_);
            CppTimer timer(softmax_time);
            for(int ct_ind = 0; ct_ind < cut_ct_nums; ct_ind++){
                evaluator_.multiply(atten_cts[ct_ind], pt_scale);
                atten_cts[ct_ind] = nonlinear_.Softmax(atten_cts[ct_ind], 
                    cut_mask[ct_ind], per_input_seq_len, num_heads, head_size, 
                    soft_bias, soft_delta_exp, soft_inv_scale);
            }
        }
        std::vector<std::vector<Ciphertext<Scheme::CKKS>>> v_cts(cut_ct_nums);
        {
            std::cout << "  ==> 3. 计算 V " << std::endl;
            CppTimer timer(v_time);
            for(int ct_ind = 0; ct_ind < cut_ct_nums; ct_ind++)
                v_cts[ct_ind] = linear_.ctpt_matmul_to_col(cut_cts[ct_ind], 
                    Wv, Bv, per_input_seq_len, num_heads, head_size);
        }
        {
            std::cout << "  ==> 4. 计算 atten_score * V " << std::endl; 
            CppTimer timer(mulV_time);
            for(int input_ind = 0; input_ind < cut_ct_nums; input_ind++)
                cut_cts[input_ind] = linear_.ctct_attenOutLinear(
                    atten_cts[input_ind], v_cts[input_ind],
                    per_input_seq_len, num_heads, head_size
                );
        }
    }
    // 5. 推理后解包 并 对齐结果
    std::cout << "  ==> 5. 计算完成，解包对齐 ..." <<  std::endl;
    std::vector<Ciphertext<Scheme::CKKS>> output_cts(input_ct_nums);
    std::vector<bool> output_ct_isEmpty(input_ct_nums, true);
    for(int ct_ind = 0; ct_ind < cut_cts.size(); ct_ind++){
        std::vector<std::vector<int>> using_block = cut_blocks[ct_ind];
        int block_ind = 0;
        for(auto it = using_block.begin(); it != using_block.end(); it++){
            const int part_size = (*it)[0], begin_ind = (*it)[1], end_ind = (*it)[2];
            const int begin_block = begin_ind / per_input_seq_len;
            const int begin_seq = begin_ind % per_input_seq_len;
            const int end_block = end_ind / per_input_seq_len;
            const int end_seq = end_ind % per_input_seq_len;
            int first_ct_len;
            if(end_block > begin_block){
                first_ct_len = per_input_seq_len - begin_seq;
            }else{
                first_ct_len = end_seq - begin_seq + 1;
            }

            Ciphertext<Scheme::CKKS> block_ct = cut_cts[ct_ind];
            {
                std::vector<double> vec_mask(slot_count, 0.0);
                for(int seq_ind = block_ind; seq_ind < block_ind + first_ct_len; seq_ind++)
                    for(int dim_ind = 0; dim_ind < hidden_dim; dim_ind++)
                        vec_mask[seq_ind * hidden_dim + dim_ind] = 1.0; 
                Plaintext<Scheme::CKKS> pt_mask;
                encoder_.encode(pt_mask, vec_mask, scale_);
                evaluator_.multiply(block_ct, pt_mask);
            }
            if(block_ind != begin_seq)
                evaluator_.rotate_rows(block_ct, (block_ind - begin_seq) * hidden_dim);
            if(output_ct_isEmpty[begin_block]){
                output_cts[begin_block] = block_ct;
                output_ct_isEmpty[begin_block] = false;
            }else{
                evaluator_.add(output_cts[begin_block], block_ct);
            }
            block_ind += first_ct_len;
            // 如果当前片段涉及两个分块，则继续采集
            if(end_block > begin_block){
                Ciphertext<Scheme::CKKS> block_ct = cut_cts[ct_ind];
                int second_ct_len = part_size - first_ct_len;
                std::vector<double> vec_mask(slot_count, 0.0);
                for(int seq_ind = block_ind; seq_ind < block_ind + second_ct_len; seq_ind++)
                    for(int dim_ind = 0; dim_ind < hidden_dim; dim_ind++)
                        vec_mask[seq_ind * hidden_dim + dim_ind] = 1.0; 
                Plaintext<Scheme::CKKS> pt_mask;
                encoder_.encode(pt_mask, vec_mask, scale_);
                evaluator_.multiply(block_ct, pt_mask);
                evaluator_.rotate_rows(block_ct, block_ind * hidden_dim);
                if(output_ct_isEmpty[end_block]){
                    output_cts[end_block] = block_ct;
                    output_ct_isEmpty[end_block] = false;
                }else{
                    evaluator_.add(output_cts[end_block], block_ct);
                }
                block_ind += second_ct_len;
            }
        }
    }
    return output_cts;
}

std::vector<Ciphertext<Scheme::CKKS>> Model::swin_ct_norm(
    std::vector<Ciphertext<Scheme::CKKS>>& input_cts,
    const int per_seq_len, const int hidden_dim, const int num_heads, 
    const int input_row, const int input_col, const int window_size, const int shift_size,
    const std::vector<std::vector<double>>& Wq, const std::vector<double>& Bq,
    const std::vector<std::vector<double>>& Wk, const std::vector<double>& Bk,
    const std::vector<std::vector<double>>& Wv, const std::vector<double>& Bv,
    const std::vector<std::vector<double>>& position_bias_table, 
    const double soft_bias, const int soft_delta_exp, const double soft_inv_scale,
    std::atomic<long long>* cyclic_shift_time, std::atomic<long long>* shift_reverse_time,
    std::atomic<long long>* window_partition_time, std::atomic<long long>* window_reverse_time,
    std::atomic<long long>* q_time, std::atomic<long long>* k_time, std::atomic<long long>* score_time, 
    std::atomic<long long>* softmax_time, std::atomic<long long>* v_time, std::atomic<long long>* mulV_time){
    const int input_ct_nums = input_cts.size();
    const int input_total_nums = input_row * input_col;
    if (input_total_nums != input_ct_nums * per_seq_len) {
        throw std::invalid_argument("[swin_ct_norm] error : 输入密文维度有误");
    }
    std::cout << " >>> 1. 循环移位 ..." << std::endl;
    std::vector<Ciphertext<Scheme::CKKS>> shifted_cts(input_ct_nums);
    {
        CppTimer timer(cyclic_shift_time);
        // 注：由于 per_seq_len, input_row, input_col 未必存在特定的倍数关系，故无法批量处理密文，只能逐个计算
        Ciphertext<Scheme::CKKS> temp_ct;
        std::vector<bool> is_cts_visited(input_ct_nums, false);
        for(int row_ind = 0; row_ind < input_row; ++row_ind){
            int new_row_ind = (row_ind - shift_size + input_row) % input_row;
            for(int col_ind = 0; col_ind < input_col; ++col_ind){
                int new_col_ind = (col_ind - shift_size + input_col) % input_col;
                int old_idx = row_ind * input_col + col_ind;
                int old_ct_ind = old_idx / per_seq_len;
                int old_ct_inner_ind = old_idx % per_seq_len;
                temp_ct = input_cts[old_ct_ind];
                {
                    std::vector<double> vec_mask(slot_count, 0.0);
                    std::fill(vec_mask.begin() + old_ct_inner_ind * hidden_dim, 
                        vec_mask.begin() + (old_ct_inner_ind + 1) * hidden_dim, 1.0);
                    Plaintext<Scheme::CKKS> pt_mask(ctx_);
                    encoder_.encode(pt_mask, vec_mask, scale_);
                    evaluator_.multiply(temp_ct, pt_mask);
                }
                int new_idx = new_row_ind * input_col + new_col_ind;
                int new_ct_ind = new_idx / per_seq_len;
                int new_ct_inner_ind = new_idx % per_seq_len;
                evaluator_.rotate_rows(temp_ct, (old_ct_inner_ind - new_ct_inner_ind) * hidden_dim);
                if(is_cts_visited[new_ct_ind]){
                    evaluator_.add(shifted_cts[new_ct_ind], temp_ct);
                }else{
                    shifted_cts[new_ct_ind] = temp_ct;
                    is_cts_visited[new_ct_ind] = true;
                }
            }
        }
    }
    std::cout << " >>> 2. 分窗口推理 ..." << std::endl;
    std::vector<Ciphertext<Scheme::CKKS>> infer_cts(input_ct_nums);
    {
        const int window_num = input_row / window_size * input_col / window_size;
        const int paded_per_window_ct_num = (window_size * window_size + per_seq_len - 1) / per_seq_len;
        const int paded_window_size = paded_per_window_ct_num * per_seq_len;
        // heads_pos_bias : [head_num, paded_window_size, paded_window_size]
        std::vector<std::vector<std::vector<double>>> heads_pos_bias = get_swin_pos_bias(window_size, position_bias_table);
        // windows_mask : [window_num, paded_window_size, paded_window_size]
        std::vector<std::vector<std::vector<bool>>> windows_mask = get_attn_mask(input_row, input_col, window_size, shift_size);
        
        if(paded_window_size > window_size * window_size){
            for(auto& head_bias : heads_pos_bias){
                head_bias.resize(paded_window_size);
                for(auto& line : head_bias)
                    line.resize(paded_window_size, 0);
            }
            for(auto& window_mask : windows_mask){
                window_mask.resize(paded_window_size);
                for(auto& line : window_mask)
                    line.resize(paded_window_size, false);
            }
        }
        std::vector<bool> is_visited_infer_cts(input_ct_nums, false);
        int window_idx = 0;
        for(int row_ind = 0; row_ind < input_row; row_ind += window_size){
            for(int col_ind = 0; col_ind < input_col; col_ind += window_size){
                std::cout << "   >> No." + std::to_string(window_idx + 1) + " / " + std::to_string(window_num)
                    << " , From: ( " << row_ind << ", " << col_ind << " ) , Window: [" 
                    << window_size << ", " << window_size << "] << " << std::endl;
                std::vector<Ciphertext<Scheme::CKKS>> window_cts = window_partition_cts(shifted_cts, input_row, input_col, 
                    per_seq_len, hidden_dim, row_ind, col_ind, window_size, window_size, window_partition_time);

                // file_.saveCtsMat(window_cts, "Ct_Input_Norm", per_seq_len, hidden_dim);

                window_cts = window_attn_ct(window_cts, per_seq_len, hidden_dim, num_heads, 
                    windows_mask[window_idx], Wq, Bq, Wk, Bk, Wv, Bv, heads_pos_bias, soft_bias, soft_delta_exp, soft_inv_scale,
                    q_time, k_time, score_time, softmax_time, v_time, mulV_time);
                    
                // file_.saveCtsMat(window_cts, "Ct_Output_Norm", per_seq_len, hidden_dim);

                window_reverse_cts(window_cts, input_row, input_col, per_seq_len, hidden_dim, infer_cts, is_visited_infer_cts, 
                    row_ind, col_ind, window_size, window_size, window_reverse_time);
                window_idx++;
            }
        }
    }
    std::cout << " >>> 3. 反转移位 ..." << std::endl;
    std::vector<Ciphertext<Scheme::CKKS>> output_cts(input_ct_nums);
    {
        CppTimer timer(shift_reverse_time);
        Ciphertext<Scheme::CKKS> temp_ct;
        std::vector<bool> is_cts_visited(input_ct_nums, false);
        for(int row_ind = 0; row_ind < input_row; ++row_ind){
            int new_row_ind = (row_ind + shift_size) % input_row;
            for(int col_ind = 0; col_ind < input_col; ++col_ind){
                int new_col_ind = (col_ind + shift_size) % input_col;
                int old_idx = row_ind * input_col + col_ind;
                int old_ct_ind = old_idx / per_seq_len;
                int old_ct_inner_ind = old_idx % per_seq_len;
                temp_ct = infer_cts[old_ct_ind];
                {
                    std::vector<double> vec_mask(slot_count, 0.0);
                    std::fill(vec_mask.begin() + old_ct_inner_ind * hidden_dim, 
                        vec_mask.begin() + (old_ct_inner_ind + 1) * hidden_dim, 1.0);
                    Plaintext<Scheme::CKKS> pt_mask(ctx_);
                    encoder_.encode(pt_mask, vec_mask, scale_);
                    evaluator_.multiply(temp_ct, pt_mask);
                }
                int new_idx = new_row_ind * input_col + new_col_ind;
                int new_ct_ind = new_idx / per_seq_len;
                int new_ct_inner_ind = new_idx % per_seq_len;
                evaluator_.rotate_rows(temp_ct, (old_ct_inner_ind - new_ct_inner_ind) * hidden_dim);
                if(is_cts_visited[new_ct_ind]){
                    evaluator_.add(output_cts[new_ct_ind], temp_ct);
                }else{
                    output_cts[new_ct_ind] = temp_ct;
                    is_cts_visited[new_ct_ind] = true;
                }
            }
        }
    }
    return output_cts;
}

std::vector<Ciphertext<Scheme::CKKS>> Model::swin_ct_opti(
    std::vector<Ciphertext<Scheme::CKKS>>& input_cts,
    const int per_seq_len, const int hidden_dim, const int num_heads,
    const int input_row, const int input_col, const int window_size, const int shift_size,
    const std::vector<std::vector<double>>& Wq, const std::vector<double>& Bq,
    const std::vector<std::vector<double>>& Wk, const std::vector<double>& Bk,
    const std::vector<std::vector<double>>& Wv, const std::vector<double>& Bv,
    const std::vector<std::vector<double>>& position_bias_table,
    const double soft_bias, const int soft_delta_exp, const double soft_inv_scale,
    std::atomic<long long>* window_partition_time, std::atomic<long long>* window_reverse_time,
    std::atomic<long long>* q_time, std::atomic<long long>* k_time, std::atomic<long long>* score_time, 
    std::atomic<long long>* softmax_time, std::atomic<long long>* v_time, std::atomic<long long>* mulV_time){
    const int input_ct_nums = input_cts.size();
    const int shift_remain_size = window_size - shift_size;
    
    const std::vector<std::vector<std::vector<double>>> heads_pos_bias = get_swin_pos_bias(window_size, position_bias_table);
    std::vector<Ciphertext<Scheme::CKKS>> output_cts(input_ct_nums);
    std::vector<bool> is_cts_visited(input_ct_nums, false);
    // 采用分割后，不需要mask，此处创建一个空的，供函数调用
    std::vector<std::vector<bool>> empty_mask;

    // 此处为避免窗口滑动，同时减少计算量，将原尺寸分为9个部分依次计算，从属分成三个分区
    int row_ind = 0, col_ind = 0;
    std::cout << " >>> 1. 开始处理上分区: " << std::endl;
    {
        // 1.1 input 左上角部分 => pos_bias 右下部分
        {
            int block_size = shift_size * shift_size;
            int paded_block_size = (block_size + per_seq_len - 1) / per_seq_len * per_seq_len;

            std::vector<std::vector<std::vector<double>>> part_heads_pos_bias = get_part_pos_bias(
                heads_pos_bias, window_size, window_size, 
                shift_remain_size, shift_remain_size, window_size - 1, window_size - 1, paded_block_size);
            
            std::cout << "   >> From: ( " << row_ind << ", " << col_ind << " ) , Window: [" 
                << shift_size << ", " << shift_size << "] << " << std::endl;

            std::vector<Ciphertext<Scheme::CKKS>> window_cts = window_partition_cts(
                input_cts, input_row, input_col, per_seq_len, hidden_dim, row_ind, col_ind, 
                shift_size, shift_size, window_partition_time);

            window_cts = window_attn_ct(window_cts, per_seq_len, hidden_dim, num_heads, 
                empty_mask, Wq, Bq, Wk, Bk, Wv, Bv, part_heads_pos_bias, soft_bias, soft_delta_exp, soft_inv_scale,
                q_time, k_time, score_time, softmax_time, v_time, mulV_time);
            
            window_reverse_cts(window_cts, input_row, input_col, per_seq_len, hidden_dim, output_cts, is_cts_visited, row_ind, col_ind, 
                shift_size, shift_size, window_reverse_time);

            col_ind += shift_size;
        }
        // 1.2 input 中上部分 => pos_bias 下半部分
        {
            int block_size = shift_size * window_size;
            int paded_block_size = (block_size + per_seq_len - 1) / per_seq_len * per_seq_len;

            std::vector<std::vector<std::vector<double>>> part_heads_pos_bias = get_part_pos_bias(
                heads_pos_bias, window_size, window_size, 
                shift_remain_size, 0, window_size - 1, window_size - 1, paded_block_size);

            for( ; col_ind < input_col - shift_remain_size; col_ind += window_size){
                std::cout << "   >> From: ( " << row_ind << ", " << col_ind << " ) , Window: [" 
                    << shift_size << ", " << window_size << "] << " << std::endl;

                std::vector<Ciphertext<Scheme::CKKS>> window_cts = window_partition_cts(
                    input_cts, input_row, input_col, per_seq_len, hidden_dim, row_ind, col_ind, 
                    shift_size, window_size, window_partition_time);

                window_cts = window_attn_ct(window_cts, per_seq_len, hidden_dim, num_heads, 
                    empty_mask, Wq, Bq, Wk, Bk, Wv, Bv, part_heads_pos_bias, soft_bias, soft_delta_exp, soft_inv_scale,
                    q_time, k_time, score_time, softmax_time, v_time, mulV_time);

                window_reverse_cts(window_cts, input_row, input_col, per_seq_len, hidden_dim, output_cts, is_cts_visited, row_ind, col_ind, 
                    shift_size, window_size, window_reverse_time);
            }
        }
        // 1.3 input 右上角部分 => pos_bias 左下部分
        {
            int block_size = shift_size * shift_remain_size;
            int paded_block_size = (block_size + per_seq_len - 1) / per_seq_len * per_seq_len;

            std::vector<std::vector<std::vector<double>>> part_heads_pos_bias = get_part_pos_bias(
                heads_pos_bias, window_size, window_size, 
                shift_remain_size, 0, window_size - 1, shift_remain_size - 1, paded_block_size);
            

            std::cout << "   >> From: ( " << row_ind << ", " << col_ind << " ) , Window: [" 
                << shift_size << ", " << shift_remain_size << "] << " << std::endl;

            std::vector<Ciphertext<Scheme::CKKS>> window_cts = window_partition_cts(
                input_cts, input_row, input_col, per_seq_len, hidden_dim, row_ind, col_ind, 
                shift_size, shift_remain_size, window_partition_time);

            window_cts = window_attn_ct(window_cts, per_seq_len, hidden_dim, num_heads, 
                empty_mask, Wq, Bq, Wk, Bk, Wv, Bv, part_heads_pos_bias, soft_bias, soft_delta_exp, soft_inv_scale,
                q_time, k_time, score_time, softmax_time, v_time, mulV_time);

            window_reverse_cts(window_cts, input_row, input_col, per_seq_len, hidden_dim, output_cts, is_cts_visited, row_ind, col_ind, 
                shift_size, shift_remain_size, window_reverse_time);
        }
    }

    row_ind += shift_size;
    col_ind = 0;
    std::cout << " >> 2. 开始处理中分区: " << std::endl;
    for( ; row_ind < input_row - shift_remain_size; row_ind += window_size){
        // 2.1 input 左中部分 => pos_bias 右半部分
        {
            int block_size = window_size * shift_size;
            int paded_block_size = (block_size + per_seq_len - 1) / per_seq_len * per_seq_len;

            std::vector<std::vector<std::vector<double>>> part_heads_pos_bias = get_part_pos_bias(
                heads_pos_bias, window_size, window_size, 
                0, shift_remain_size, window_size - 1, window_size - 1, paded_block_size);
            
            std::cout << "   >> From: ( " << row_ind << ", " << col_ind << " ) , Window: [" 
                << window_size << ", " << shift_size << "] << " << std::endl;

            std::vector<Ciphertext<Scheme::CKKS>> window_cts = window_partition_cts(
                input_cts, input_row, input_col, per_seq_len, hidden_dim, row_ind, col_ind, 
                window_size, shift_size, window_partition_time);

            window_cts = window_attn_ct(window_cts, per_seq_len, hidden_dim, num_heads, 
                empty_mask, Wq, Bq, Wk, Bk, Wv, Bv, part_heads_pos_bias, soft_bias, soft_delta_exp, soft_inv_scale,
                q_time, k_time, score_time, softmax_time, v_time, mulV_time);

            window_reverse_cts(window_cts, input_row, input_col, per_seq_len, hidden_dim, output_cts, is_cts_visited, row_ind, col_ind, 
                window_size, shift_size, window_reverse_time);

            col_ind += shift_size;
        }
        // 2.2 input 中间部分 => pos_bias 完整
        for( ; col_ind < input_col - shift_remain_size; col_ind += window_size){
            std::cout << "   >> From: ( " << row_ind << ", " << col_ind << " ) , Window: [" 
                << window_size << ", " << window_size << "] << " << std::endl;

            std::vector<Ciphertext<Scheme::CKKS>> window_cts = window_partition_cts(
                input_cts, input_row, input_col, per_seq_len, hidden_dim, row_ind, col_ind, 
                window_size, window_size, window_partition_time);

            window_cts = window_attn_ct(window_cts, per_seq_len, hidden_dim, num_heads, 
                empty_mask, Wq, Bq, Wk, Bk, Wv, Bv, heads_pos_bias, soft_bias, soft_delta_exp, soft_inv_scale,
                q_time, k_time, score_time, softmax_time, v_time, mulV_time);

            window_reverse_cts(window_cts, input_row, input_col, per_seq_len, hidden_dim, output_cts, is_cts_visited, row_ind, col_ind, 
                window_size, window_size, window_reverse_time);
        }
        // 2.3 input 右中部分 => pos_bias 左半部分
        {
            int block_size = window_size * shift_remain_size;
            int paded_block_size = (block_size + per_seq_len - 1) / per_seq_len * per_seq_len;

            std::vector<std::vector<std::vector<double>>> part_heads_pos_bias = get_part_pos_bias(
                heads_pos_bias, window_size, window_size, 
                0, 0, window_size - 1, shift_remain_size - 1, paded_block_size);

            std::cout << "   >> From: ( " << row_ind << ", " << col_ind << " ) , Window: [" 
                << window_size << ", " << shift_remain_size << "] << " << std::endl;

            std::vector<Ciphertext<Scheme::CKKS>> window_cts = window_partition_cts(
                input_cts, input_row, input_col, per_seq_len, hidden_dim, row_ind, col_ind, 
                window_size, shift_remain_size, window_partition_time);

            window_cts = window_attn_ct(window_cts, per_seq_len, hidden_dim, num_heads, 
                empty_mask, Wq, Bq, Wk, Bk, Wv, Bv, part_heads_pos_bias, soft_bias, soft_delta_exp, soft_inv_scale,
                q_time, k_time, score_time, softmax_time, v_time, mulV_time);

            window_reverse_cts(window_cts, input_row, input_col, per_seq_len, hidden_dim, output_cts, is_cts_visited, row_ind, col_ind, 
                window_size, shift_remain_size, window_reverse_time);
        }
        col_ind = 0;
    }
    col_ind = 0;
    std::cout << " >> 3. 开始处理下分区: " << std::endl;
    {
        // 3.1 input 左下角部分 => pos_bias 右上部分
        {
            int block_size = shift_remain_size * shift_size;
            int paded_block_size = (block_size + per_seq_len - 1) / per_seq_len * per_seq_len;

            std::vector<std::vector<std::vector<double>>> part_heads_pos_bias = get_part_pos_bias(
                heads_pos_bias, window_size, window_size, 
                0, shift_remain_size, shift_remain_size - 1, window_size - 1, paded_block_size);
            
            std::cout << "   >> From: ( " << row_ind << ", " << col_ind << " ) , Window: [" 
                << shift_remain_size << ", " << shift_size << "] << " << std::endl;

            std::vector<Ciphertext<Scheme::CKKS>> window_cts = window_partition_cts(
                input_cts, input_row, input_col, per_seq_len, hidden_dim, row_ind, col_ind, 
                shift_remain_size, shift_size, window_partition_time);

            window_cts = window_attn_ct(window_cts, per_seq_len, hidden_dim, num_heads, 
                empty_mask, Wq, Bq, Wk, Bk, Wv, Bv, part_heads_pos_bias, soft_bias, soft_delta_exp, soft_inv_scale,
                q_time, k_time, score_time, softmax_time, v_time, mulV_time);

            window_reverse_cts(window_cts, input_row, input_col, per_seq_len, hidden_dim, output_cts, is_cts_visited, row_ind, col_ind, 
                shift_remain_size, shift_size, window_reverse_time);
            
            col_ind += shift_size;
        }
        // 3.2 input 中下部分 => pos_bias 上半部分
        {
            int block_size = shift_remain_size * window_size;
            int paded_block_size = (block_size + per_seq_len - 1) / per_seq_len * per_seq_len;

            std::vector<std::vector<std::vector<double>>> part_heads_pos_bias = get_part_pos_bias(
                heads_pos_bias, window_size, window_size, 
                0, 0, shift_remain_size - 1, window_size - 1, paded_block_size);
            

            for( ; col_ind < input_col - shift_remain_size; col_ind += window_size){
                std::cout << "   >> From: ( " << row_ind << ", " << col_ind << " ) , Window: [" 
                    << shift_remain_size << ", " << window_size << "] << " << std::endl;

                std::vector<Ciphertext<Scheme::CKKS>> window_cts = window_partition_cts(
                    input_cts, input_row, input_col, per_seq_len, hidden_dim, row_ind, col_ind, 
                    shift_remain_size, window_size, window_partition_time);

                window_cts = window_attn_ct(window_cts, per_seq_len, hidden_dim, num_heads, 
                    empty_mask, Wq, Bq, Wk, Bk, Wv, Bv, part_heads_pos_bias, soft_bias, soft_delta_exp, soft_inv_scale,
                    q_time, k_time, score_time, softmax_time, v_time, mulV_time);

                window_reverse_cts(window_cts, input_row, input_col, per_seq_len, hidden_dim, output_cts, is_cts_visited, row_ind, col_ind, 
                    shift_remain_size, window_size, window_reverse_time);
            }
        }
        // 3.3 input 右下角部分 => pos_bias 左上部分
        {
            int block_size = shift_remain_size * shift_remain_size;
            int paded_block_size = (block_size + per_seq_len - 1) / per_seq_len * per_seq_len;

            std::vector<std::vector<std::vector<double>>> part_heads_pos_bias = get_part_pos_bias(
                heads_pos_bias, window_size, window_size, 
                0, 0, shift_remain_size - 1, shift_remain_size - 1, paded_block_size);
            
            std::cout << "   >> From: ( " << row_ind << ", " << col_ind << " ) , Window: [" 
                << shift_remain_size << ", " << shift_remain_size << "] << " << std::endl;

            std::vector<Ciphertext<Scheme::CKKS>> window_cts = window_partition_cts(
                input_cts, input_row, input_col, per_seq_len, hidden_dim, row_ind, col_ind, 
                shift_remain_size, shift_remain_size, window_partition_time);

            window_cts = window_attn_ct(window_cts, per_seq_len, hidden_dim, num_heads, 
                empty_mask, Wq, Bq, Wk, Bk, Wv, Bv, part_heads_pos_bias, soft_bias, soft_delta_exp, soft_inv_scale,
                q_time, k_time, score_time, softmax_time, v_time, mulV_time);

            window_reverse_cts(window_cts, input_row, input_col, per_seq_len, hidden_dim, output_cts, is_cts_visited, row_ind, col_ind, 
                shift_remain_size, shift_remain_size, window_reverse_time);
        }
    }
    return output_cts;
}

std::vector<Ciphertext<Scheme::CKKS>> Model::multiHeadAtten_ct(
    std::vector<Ciphertext<Scheme::CKKS>>& input_cts,
    const int seq_len, const int hidden_dim, const int num_heads, 
    const std::vector<std::vector<bool>>& atten_mask,
    const std::vector<std::vector<double>>& Wq, const std::vector<double>& Bq,
    const std::vector<std::vector<double>>& Wk, const std::vector<double>& Bk,
    const std::vector<std::vector<double>>& Wv,const std::vector<double>& Bv,
    const double soft_bias, const int soft_delta_exp, const double soft_inv_scale,
    std::atomic<long long>* q_time, std::atomic<long long>* k_time, std::atomic<long long>* score_time, 
    std::atomic<long long>* softmax_time, std::atomic<long long>* v_time, std::atomic<long long>* mulV_time){
    const int head_size = hidden_dim / num_heads;
    const int input_ct_nums = input_cts.size();
    const int per_input_seq_len = seq_len / input_ct_nums;
    // 1. 参数校验
    if (hidden_dim % num_heads != 0) {
        throw std::invalid_argument("[multiHeadAtten_ct] error: 头数不能整除.");
    }
    if( per_input_seq_len * hidden_dim > slot_count ){
        throw std::invalid_argument("[multiHeadAtten_ct] error: 分块参数大于槽空间.");
    }
    // 2. 计算 Q * K.T
        // 此处为了代码的可读性，将 atten_scores_cts 定义为二维数组，由于数量不多，故性能无差异
    std::vector<std::vector<Ciphertext<Scheme::CKKS>>> atten_scores_cts(
        input_ct_nums, std::vector<Ciphertext<Scheme::CKKS>>(input_ct_nums));
    {
        std::cout << " >>>>> [密文推理] 1. 计算 Q 和 K " << std::endl;
        // 2.1 计算 Q 和 K 矩阵，仅逻辑上视为多头，仍存储在单个密文中
        std::vector<Ciphertext<Scheme::CKKS>> q_cts(input_ct_nums);
        {
            CppTimer timer(q_time);
            for(int input_ind = 0; input_ind < input_ct_nums; input_ind++){
                q_cts[input_ind] = linear_.ctpt_matmul_to_row(
                    input_cts[input_ind], Wq, Bq, per_input_seq_len, hidden_dim);
            }
        }
        std::vector<Ciphertext<Scheme::CKKS>> k_cts(input_ct_nums);
        {
            CppTimer timer(k_time);
            for(int input_ind = 0; input_ind < input_ct_nums; input_ind++){
                k_cts[input_ind] = linear_.ctpt_matmul_to_row(
                    input_cts[input_ind], Wk, Bk, per_input_seq_len, hidden_dim);
            }
        }
            
        std::cout << " >>>>> [密文推理] 2. 计算 Q @ K.T " << std::endl;
        // 2.2 计算注意力得分矩阵 Q * K.T
        {
            CppTimer timer(score_time);
            for(int q_ind = 0; q_ind < input_ct_nums; q_ind++)
                for(int k_ind = 0; k_ind < input_ct_nums; k_ind++)
                    atten_scores_cts[q_ind][k_ind] = linear_.ctct_attenscore(
                        q_cts[q_ind], k_cts[k_ind], per_input_seq_len, hidden_dim, num_heads);
        }
        
    }
    
    std::cout << " >>>>> [密文推理] 3. 近似 Softmax " << std::endl;
    // 3. 除以 sqrt(head_size) 并进行 softmax
    {
        std::vector<double> vec_mask(slot_count, 0.0);
        double scale_factor = 1.0 / std::sqrt(static_cast<double>(head_size));
        for(int seq_ind = 0; seq_ind < per_input_seq_len; seq_ind++)
            for(int head_ind = 0; head_ind < num_heads; head_ind++)
                for(int factor_ind = 0; factor_ind < per_input_seq_len; factor_ind++)
                    vec_mask[seq_ind * hidden_dim + head_ind * head_size + factor_ind] = scale_factor;
        Plaintext<Scheme::CKKS> pt_scale;
        encoder_.encode(pt_scale, vec_mask, scale_);
        {
            CppTimer timer(softmax_time);
            for(int q_ind = 0; q_ind < input_ct_nums; q_ind++){
                for(int k_ind = 0; k_ind < input_ct_nums; k_ind++){
                    // Scale the attention scores
                    evaluator_.multiply(atten_scores_cts[q_ind][k_ind], pt_scale);
                }
                std::vector<std::vector<bool>> part_mask;
                for(int seq_ind = 0; seq_ind < per_input_seq_len; seq_ind++){
                    part_mask.emplace_back(atten_mask[q_ind * per_input_seq_len + seq_ind]);
                }
                // // 保存中间结果以调试
                // {
                //     file_.saveCtAttenScore(atten_scores_cts[q_ind], 
                //         "Ct/Before_Softmax_L_" + std::to_string(q_ind + 1), 
                //         seq_len, num_heads, head_size);
                // }
                atten_scores_cts[q_ind] = nonlinear_.Softmax(atten_scores_cts[q_ind], 
                    part_mask, seq_len, per_input_seq_len, num_heads, head_size,
                    soft_bias, soft_delta_exp, soft_inv_scale);
            }
        }
    }
    // 4. 计算 Attention 输出：Softmax(QK.T) * V
    std::vector<Ciphertext<Scheme::CKKS>> output_cts(input_ct_nums);
    {
        // 4.1 计算 V, 为便于后续进行乘法，此处直接每列存在一个密文内，不再紧凑对齐
            // 此处 v_cts 维度为 [input_ct_nums, head_size / per_input_seq_len]
        std::cout << " >>>>> [密文推理] 4. 计算 V " << std::endl;
        std::vector<std::vector<Ciphertext<Scheme::CKKS>>> v_cts(input_ct_nums);
        {
            CppTimer timer(v_time);
            for(int ct_ind = 0; ct_ind < input_ct_nums; ct_ind++)
                v_cts[ct_ind] = linear_.ctpt_matmul_to_col(input_cts[ct_ind], 
                    Wv, Bv, per_input_seq_len, num_heads, head_size);
        }
        
        std::cout << " >>>>> [密文推理] 5. 计算 atten_score * V " << std::endl; 
        // 4.2 计算 atten_scores_cts * v_cts，按照分块思想: V[i] = atten_scores_cts[i,:] * v_cts[:]
        {
            CppTimer timer(mulV_time);
            for(int input_ind = 0; input_ind < input_ct_nums; input_ind++)
                output_cts[input_ind] = linear_.ctct_attenOutLinear(
                    atten_scores_cts[input_ind], v_cts,
                    input_ct_nums, per_input_seq_len, num_heads, head_size
                );
        }
    }
    // 保存中间结果以调试
    // {
    //     for(int ct_ind = 0; ct_ind < input_ct_nums; ct_ind++){
    //         file_.saveCtMat(output_cts[ct_ind], 
    //             "Ct/12.Output_L_" + std::to_string(ct_ind + 1), 
    //             per_input_seq_len, hidden_dim);
    //     }
    // }
    return output_cts;
}

Ciphertext<Scheme::CKKS> Model::multiHeadAtten_ct(
    Ciphertext<Scheme::CKKS>& input_ct,
    const int per_input_seq_len, const int hidden_dim, const int num_heads, 
    const std::vector<std::vector<bool>>& atten_mask,
    const std::vector<std::vector<double>>& Wq, const std::vector<double>& Bq,
    const std::vector<std::vector<double>>& Wk, const std::vector<double>& Bk,
    const std::vector<std::vector<double>>& Wv,const std::vector<double>& Bv,
    const double soft_bias, const int soft_delta_exp, const double soft_inv_scale,
    std::atomic<long long>* q_time, std::atomic<long long>* k_time, std::atomic<long long>* score_time, 
    std::atomic<long long>* softmax_time, std::atomic<long long>* v_time, std::atomic<long long>* mulV_time){
    const int head_size = hidden_dim / num_heads;

    // 1. 计算 Q * K.T
    Ciphertext<Scheme::CKKS> atten_scores_ct;
    {
        // 1.1 计算 Q 和 K 矩阵，仅逻辑上视为多头，仍存储在单个密文中
        Ciphertext<Scheme::CKKS> q_ct, k_ct;
        {
            CppTimer timer(q_time);
            q_ct = linear_.ctpt_matmul_to_row(
                input_ct, Wq, Bq, per_input_seq_len, hidden_dim);
        }
        {
            CppTimer timer(k_time);
            k_ct = linear_.ctpt_matmul_to_row(
                input_ct, Wk, Bk, per_input_seq_len, hidden_dim);
        }
        // 1.2 计算注意力得分矩阵 Q * K.T
        {
            CppTimer timer(score_time);
            atten_scores_ct = linear_.ctct_attenscore(
                q_ct, k_ct, per_input_seq_len, hidden_dim, num_heads);
        }
        
    }
    // 2. 除以 sqrt(head_size) 并进行 softmax
    {
        std::vector<double> vec_mask(slot_count, 0.0);
        double scale_factor = 1.0 / std::sqrt(static_cast<double>(head_size));
        for(int seq_ind = 0; seq_ind < per_input_seq_len; seq_ind++){
            for(int head_ind = 0; head_ind < num_heads; head_ind++){
                for(int factor_ind = 0; factor_ind < per_input_seq_len; factor_ind++){
                    vec_mask[seq_ind * hidden_dim + head_ind * head_size + factor_ind] = scale_factor;
                }
            }
        }
        Plaintext<Scheme::CKKS> pt_scale;
        encoder_.encode(pt_scale, vec_mask, scale_);
        // Scale the attention scores
        evaluator_.multiply(atten_scores_ct, pt_scale);
        CppTimer timer(softmax_time);
        atten_scores_ct = nonlinear_.Softmax(atten_scores_ct, 
            atten_mask, per_input_seq_len, num_heads, head_size, 
            soft_bias, soft_delta_exp, soft_inv_scale);
    }
    // 3. 计算 Attention 输出：Softmax(QK.T) * V
    Ciphertext<Scheme::CKKS> output_ct;
    {
        std::vector<Ciphertext<Scheme::CKKS>> v_cts;
        {
            CppTimer timer(v_time);
            v_cts = linear_.ctpt_matmul_to_col(input_ct, 
                    Wv, Bv, per_input_seq_len, num_heads, head_size);
        }
        {
            CppTimer timer(mulV_time);
            output_ct = linear_.ctct_attenOutLinear(
                    atten_scores_ct, v_cts, per_input_seq_len, num_heads, head_size);
        }
    }
    
    return output_ct;
}

std::vector<Ciphertext<Scheme::CKKS>> Model::window_attn_ct(
    std::vector<Ciphertext<Scheme::CKKS>>& input_cts,
    const int per_input_seq_len, const int hidden_dim, const int num_heads, 
    const std::vector<std::vector<bool>>& atten_mask,
    const std::vector<std::vector<double>>& Wq, const std::vector<double>& Bq,
    const std::vector<std::vector<double>>& Wk, const std::vector<double>& Bk,
    const std::vector<std::vector<double>>& Wv,const std::vector<double>& Bv,
    const std::vector<std::vector<std::vector<double>>>& head_pos_bias,
    const double soft_bias, const int soft_delta_exp, const double soft_inv_scale,
    std::atomic<long long>* q_time, std::atomic<long long>* k_time, std::atomic<long long>* score_time, 
    std::atomic<long long>* softmax_time, std::atomic<long long>* v_time, std::atomic<long long>* mulV_time){

    // 1. 参数准备
    const int head_size = hidden_dim / num_heads;
    const int input_ct_nums = input_cts.size();
    const int seq_len = per_input_seq_len * input_ct_nums;

    if (num_heads != head_pos_bias.size() || seq_len != head_pos_bias[0].size() || seq_len != head_pos_bias[0][0].size())
        throw std::invalid_argument("[window_attn_ct] error : 输入 Position Bias 有误");
    // 2. 计算 Q * K.T
    std::vector<std::vector<Ciphertext<Scheme::CKKS>>> atten_scores_cts(
        input_ct_nums, std::vector<Ciphertext<Scheme::CKKS>>(input_ct_nums));
    {
        // 2.1 计算 Q 和 K 矩阵，仅逻辑上视为多头，仍存储在单个密文中
        std::vector<Ciphertext<Scheme::CKKS>> q_cts(input_ct_nums);
        {
            CppTimer timer(q_time);
            for(int input_ind = 0; input_ind < input_ct_nums; input_ind++){
                q_cts[input_ind] = linear_.ctpt_matmul_to_row(
                    input_cts[input_ind], Wq, Bq, per_input_seq_len, hidden_dim);
            }
        }
        std::vector<Ciphertext<Scheme::CKKS>> k_cts(input_ct_nums);
        {
            CppTimer timer(k_time);
            for(int input_ind = 0; input_ind < input_ct_nums; input_ind++){
                k_cts[input_ind] = linear_.ctpt_matmul_to_row(
                    input_cts[input_ind], Wk, Bk, per_input_seq_len, hidden_dim);
            }
        }
        // 2.2 计算注意力得分矩阵 Q * K.T
        {
            CppTimer timer(score_time);
            for(int q_ind = 0; q_ind < input_ct_nums; q_ind++)
                for(int k_ind = 0; k_ind < input_ct_nums; k_ind++)
                    atten_scores_cts[q_ind][k_ind] = linear_.ctct_attenscore(
                        q_cts[q_ind], k_cts[k_ind], per_input_seq_len, hidden_dim, num_heads);
        }
    }
    // 3. 除以 sqrt(head_size)
    {
        // 3.1 对 scale 进行编码构造
        Plaintext<Scheme::CKKS> pt_scale;
        {
            std::vector<double> vec_mask(slot_count, 0.0);
            double scale_factor = 1.0 / std::sqrt(static_cast<double>(head_size));
            for(int seq_ind = 0; seq_ind < per_input_seq_len; seq_ind++)
                for(int head_ind = 0; head_ind < num_heads; head_ind++)
                    for(int factor_ind = 0; factor_ind < per_input_seq_len; factor_ind++)
                        vec_mask[seq_ind * hidden_dim + head_ind * head_size + factor_ind] = scale_factor;
            encoder_.encode(pt_scale, vec_mask, scale_);
        }
        for(int q_ind = 0; q_ind < input_ct_nums; q_ind++){
            for(int k_ind = 0; k_ind < input_ct_nums; k_ind++){
                evaluator_.multiply(atten_scores_cts[q_ind][k_ind], pt_scale);
                // 3.2 提取每个密文对应的 bias 并进行编码构造
                Plaintext<Scheme::CKKS> pt_bias;
                {
                    std::vector<double> vec_bias(slot_count, 0.0);
                    for(int row_ind = 0; row_ind < per_input_seq_len; ++row_ind)
                        for(int head_ind = 0; head_ind < num_heads; ++head_ind)
                            for(int col_ind = 0; col_ind < per_input_seq_len; ++col_ind)
                                vec_bias[row_ind * hidden_dim + head_ind * head_size + col_ind] = 
                                    head_pos_bias[head_ind][q_ind * per_input_seq_len + row_ind][k_ind * per_input_seq_len + col_ind];
                    encoder_.encode(pt_bias, vec_bias, scale_);
                }
                evaluator_.add(atten_scores_cts[q_ind][k_ind], pt_bias);
            }
        }
    }
    // 4. 近似计算 Softmax
    {
        CppTimer timer(softmax_time);
        for(int q_ind = 0; q_ind < input_ct_nums; q_ind++){
            std::vector<std::vector<bool>> part_mask;
            if(!atten_mask.empty()){
                part_mask.reserve(per_input_seq_len);
                for(int seq_ind = 0; seq_ind < per_input_seq_len; seq_ind++){
                    part_mask.emplace_back(atten_mask[q_ind * per_input_seq_len + seq_ind]);
                }
            }
            atten_scores_cts[q_ind] = nonlinear_.Softmax(atten_scores_cts[q_ind], 
                part_mask, seq_len, per_input_seq_len, num_heads, head_size, 
                soft_bias, soft_delta_exp, soft_inv_scale);
        }
    }
    // 5. 计算 Attention 输出：Softmax(QK.T) * V
    std::vector<Ciphertext<Scheme::CKKS>> output_cts(input_ct_nums);
    {
        // 5.1 计算 V, 为便于后续进行乘法，此处直接每列存在一个密文内，不再紧凑对齐
        std::vector<std::vector<Ciphertext<Scheme::CKKS>>> v_cts(input_ct_nums);
        {
            CppTimer timer(v_time);
            for(int ct_ind = 0; ct_ind < input_ct_nums; ct_ind++)
                v_cts[ct_ind] = linear_.ctpt_matmul_to_col(input_cts[ct_ind], 
                    Wv, Bv, per_input_seq_len, num_heads, head_size);
        }
        // 5.2 计算 atten_scores_cts * v_cts，按照分块思想: V[i] = atten_scores_cts[i,:] * v_cts[:]
        {
            CppTimer timer(mulV_time);
            for(int input_ind = 0; input_ind < input_ct_nums; input_ind++)
                output_cts[input_ind] = linear_.ctct_attenOutLinear(
                    atten_scores_cts[input_ind], v_cts,
                    input_ct_nums, per_input_seq_len, num_heads, head_size
                );
        }
    }
    return output_cts;
}

std::vector<Ciphertext<Scheme::CKKS>> Model::window_partition_cts(
    const std::vector<Ciphertext<Scheme::CKKS>>& input_cts,
    const int input_row_len, const int input_col_len, const int per_seq_len, const int hidden_dim,
    const int begin_row_ind, const int begin_col_ind, const int window_row_len, const int window_col_len,
    std::atomic<long long>* duration_time){
    CppTimer timer(duration_time);
    const int window_tol_size = window_row_len * window_col_len;
    // 注意此处为向上取整，即自动 Pad 至整数块密文
    const int ct_num = (window_tol_size + per_seq_len - 1) / per_seq_len;
    std::vector<Ciphertext<Scheme::CKKS>> output_cts(ct_num);
    std::vector<bool> is_cts_visited(ct_num, false);
    for(int row_ind = 0; row_ind < window_row_len; ++row_ind){
        for(int col_ind = 0; col_ind < window_col_len; ++col_ind){
            int old_ind = (begin_row_ind + row_ind) * input_col_len + begin_col_ind + col_ind;
            int old_ct_ind = old_ind / per_seq_len;
            int old_ct_inner_ind = old_ind % per_seq_len;
            Ciphertext<Scheme::CKKS>temp_ct = input_cts[old_ct_ind];
            {
                std::vector<double> vec_mask(slot_count, 0.0);
                std::fill(vec_mask.begin() + old_ct_inner_ind * hidden_dim, 
                    vec_mask.begin() + (old_ct_inner_ind + 1) * hidden_dim, 1.0);
                Plaintext<Scheme::CKKS> pt_mask(ctx_);
                encoder_.encode(pt_mask, vec_mask, scale_);
                evaluator_.multiply(temp_ct, pt_mask);
            }
            int target_ind = row_ind * window_col_len + col_ind;
            int target_ct_ind = target_ind / per_seq_len;
            int target_ct_inner_ind = target_ind % per_seq_len;
            evaluator_.rotate_rows(temp_ct, (old_ct_inner_ind - target_ct_inner_ind) * hidden_dim);
            if(is_cts_visited[target_ct_ind]){
                evaluator_.add(output_cts[target_ct_ind], temp_ct);
            }else{
                output_cts[target_ct_ind] = temp_ct;
                is_cts_visited[target_ct_ind] = true;
            }
        }
    }
    return output_cts;
}

void Model::window_reverse_cts(const std::vector<Ciphertext<Scheme::CKKS>>& window_cts, 
    const int input_row_len, const int input_col_len, const int per_seq_len, const int hidden_dim,
    std::vector<Ciphertext<Scheme::CKKS>>& output_cts, std::vector<bool>& is_output_visited,
    const int begin_row_ind, const int begin_col_ind, const int window_row_len, const int window_col_len,
    std::atomic<long long>* duration_time){
    CppTimer timer(duration_time);
    for(int row_ind = 0; row_ind < window_row_len; ++row_ind){
        for(int col_ind = 0; col_ind < window_col_len; ++col_ind){
            int old_ind = row_ind * window_col_len + col_ind;
            int old_ct_ind = old_ind / per_seq_len;
            int old_ct_inner_ind = old_ind % per_seq_len;
            Ciphertext<Scheme::CKKS>temp_ct = window_cts[old_ct_ind];
            {
                std::vector<double> vec_mask(slot_count, 0.0);
                std::fill(vec_mask.begin() + old_ct_inner_ind * hidden_dim, 
                    vec_mask.begin() + (old_ct_inner_ind + 1) * hidden_dim, 1.0);
                Plaintext<Scheme::CKKS> pt_mask(ctx_);
                encoder_.encode(pt_mask, vec_mask, scale_);
                evaluator_.multiply(temp_ct, pt_mask);
            }
            int target_ind = (begin_row_ind + row_ind) * input_col_len + begin_col_ind + col_ind;;
            int target_ct_ind = target_ind / per_seq_len;
            int target_ct_inner_ind = target_ind % per_seq_len;
            evaluator_.rotate_rows(temp_ct, (old_ct_inner_ind - target_ct_inner_ind) * hidden_dim);
            if(is_output_visited[target_ct_ind]){
                evaluator_.add(output_cts[target_ct_ind], temp_ct);
            }else{
                output_cts[target_ct_ind] = temp_ct;
                is_output_visited[target_ct_ind] = true;
            }
        }
    }
}

std::vector<std::vector<std::vector<double>>> Model::get_part_pos_bias(
    const std::vector<std::vector<std::vector<double>>>& heads_pos_bias, const int window_row, const int window_col,
    const int begin_row_ind, const int begin_col_ind, const int end_row_ind, const int end_col_ind, const int paded_size){
    const int window_tol_size = window_row * window_col;
    // 基础参数校验
    if (begin_row_ind > end_row_ind || begin_col_ind > end_col_ind ||
        begin_row_ind < 0 || begin_row_ind >= window_row ||begin_col_ind < 0 || begin_col_ind >= window_col ||
        end_row_ind < 0 || end_row_ind >= window_row || end_col_ind < 0 || end_col_ind >= window_col) {
        throw std::invalid_argument("[get_part_pos_bias] 坐标参数非法");
    }
    if (heads_pos_bias.empty()) 
        throw std::invalid_argument("[get_part_pos_bias] 输入三维矩阵为空");

    // 计算子窗口尺寸
    const int sub_row = end_row_ind - begin_row_ind + 1;
    const int sub_col = end_col_ind - begin_col_ind + 1;
    const int sub_size = sub_row * sub_col;

    if (paded_size < sub_size)
        throw std::invalid_argument("[get_part_pos_bias] 填充尺寸有误（小于子窗口尺寸）");

    // 预生成目标索引列表（只需生成一次，复用）
    std::vector<int> target_indices;
    target_indices.reserve(sub_size);
    for (int i = begin_row_ind; i <= end_row_ind; ++i) 
        for (int j = begin_col_ind; j <= end_col_ind; ++j) 
            target_indices.push_back(i * window_col + j);

    // 遍历第一个维度，批量处理每个二维矩阵
    std::vector<std::vector<std::vector<double>>> sub_bias;
    sub_bias.reserve(heads_pos_bias.size()); 
    for (const auto& mat_2d : heads_pos_bias) {
        // 校验当前二维矩阵维度
        if (mat_2d.size() != window_tol_size || mat_2d[0].size() != window_tol_size) 
            throw std::invalid_argument("[get_part_pos_bias] 二维矩阵维度不匹配");
        
        // 1. 创建填充后的矩阵，初始值全为0（尺寸为paded_size×paded_size）
        std::vector<std::vector<double>> paded_mat(paded_size, std::vector<double>(paded_size, 0.0));
        // 2. 将提取的子矩阵内容填充到paded_mat的左上角（前sub_size行/列）
        for (int p = 0; p < sub_size; ++p) {
            int orig_row = target_indices[p];
            for (int q = 0; q < sub_size; ++q) {
                int orig_col = target_indices[q];
                paded_mat[p][q] = mat_2d[orig_row][orig_col];
            }
        }
        // 3. 将填充后的矩阵加入结果（替换原有sub_mat）
        sub_bias.push_back(std::move(paded_mat));
    }
    return sub_bias;
}

std::vector<std::vector<double>> Model::multiHeadAtten_pt(const std::vector<std::vector<double>>& input,
    int seq_len, int hidden_dim, int num_heads,
    const std::vector<std::vector<bool>>& atten_mask,
    const std::vector<std::vector<double>>& Wq, const std::vector<double>& Bq,
    const std::vector<std::vector<double>>& Wk, const std::vector<double>& Bk,
    const std::vector<std::vector<double>>& Wv,const std::vector<double>& Bv){
    // 前置校验：确保维度合法
    if (input.size() != seq_len || input[0].size() != hidden_dim) {
        throw std::invalid_argument("input维度与seq_len/embed_dim不匹配");
    }
    if (hidden_dim % num_heads != 0) {
        throw std::invalid_argument("embed_dim必须能被num_heads整除");
    }
    int head_size = hidden_dim / num_heads;
    // ==========  计算 Q 和 K，并拆分多头 ==========
    std::vector<std::vector<double>> Q = linear_.matAddBias(linear_.matMul(input, Wq), Bq);
    std::vector<std::vector<double>> K = linear_.matAddBias(linear_.matMul(input, Wk), Bk);
    std::vector<std::vector<double>> V = linear_.matAddBias(linear_.matMul(input, Wv), Bv);

    std::vector<std::vector<std::vector<double>>> heads_Q(num_heads, 
        std::vector<std::vector<double>>(seq_len, std::vector<double>(head_size, 0.0)));
    std::vector<std::vector<std::vector<double>>> heads_K(num_heads, 
        std::vector<std::vector<double>>(seq_len, std::vector<double>(head_size, 0.0)));
    std::vector<std::vector<std::vector<double>>> heads_V(num_heads, 
        std::vector<std::vector<double>>(seq_len, std::vector<double>(head_size, 0.0)));

    for (int head_idx = 0; head_idx < num_heads; ++head_idx) {
        int start_dim = head_idx * head_size;
        for (int seq_idx = 0; seq_idx < seq_len; ++seq_idx) {
            for (int dim_idx = 0; dim_idx < head_size; ++dim_idx) {
                heads_Q[head_idx][seq_idx][dim_idx] = Q[seq_idx][start_dim + dim_idx];
                heads_K[head_idx][seq_idx][dim_idx] = K[seq_idx][start_dim + dim_idx];
                heads_V[head_idx][seq_idx][dim_idx] = V[seq_idx][start_dim + dim_idx];
            }
        }
    }

    // ==========  遍历每个头计算注意力 ==========
    std::vector<std::vector<std::vector<double>>> heads_output(num_heads, 
        std::vector<std::vector<double>>(seq_len, std::vector<double>(head_size, 0.0)));
    for (int head_idx = 0; head_idx < num_heads; ++head_idx) {
        const auto& Q_h = heads_Q[head_idx];
        const auto& K_h = heads_K[head_idx];
        const auto& V_h = heads_V[head_idx];

        int k_rows = K_h.size();
        int k_cols = K_h[0].size();
        std::vector<std::vector<double>> Kt_h(k_cols, std::vector<double>(k_rows, 0.0));
        for (int i = 0; i < k_rows; ++i)
            for (int j = 0; j < k_cols; ++j) 
                Kt_h[j][i] = K_h[i][j];
        std::vector<std::vector<double>> QKt_h = linear_.matMul(Q_h, Kt_h);

        std::vector<std::vector<double>> scaled_QKt_h(QKt_h.size(), std::vector<double>(QKt_h[0].size(), 0.0));
        double scale_factor = sqrt(head_size);
        for (int i = 0; i < QKt_h.size(); ++i) 
            for (int j = 0; j < QKt_h[0].size(); ++j) 
                scaled_QKt_h[i][j] = QKt_h[i][j] / scale_factor;

        // 应用Attention Mask + 计算 Softmax
        std::vector<std::vector<double>> masked_QKt_h(scaled_QKt_h.size(), std::vector<double>(scaled_QKt_h[0].size(), 0.0));
        if (scaled_QKt_h.size() != atten_mask.size() || scaled_QKt_h[0].size() != atten_mask[0].size()) {
            throw std::invalid_argument("得分矩阵与mask维度不匹配");
        }
        for (int i = 0; i < scaled_QKt_h.size(); ++i){
            for (int j = 0; j < scaled_QKt_h[0].size(); ++j){
                if (atten_mask[i][j]) { // 等价于mask=1
                    masked_QKt_h[i][j] = scaled_QKt_h[i][j];
                } else { // 等价于mask=0（屏蔽，设置为极小负数）
                    masked_QKt_h[i][j] = -1e9;
                }
            }
        }

        std::vector<std::vector<double>> atten_weights_h(masked_QKt_h.size(), std::vector<double>(masked_QKt_h[0].size(), 0.0));
        for (int i = 0; i < masked_QKt_h.size(); ++i) {
            double row_max = *std::max_element(masked_QKt_h[i].begin(), masked_QKt_h[i].end());
            double exp_sum = 0.0;
            std::vector<double> exp_vals(masked_QKt_h[0].size());
            for (int j = 0; j < masked_QKt_h[0].size(); ++j) {
                exp_vals[j] = exp(masked_QKt_h[i][j] - row_max);
                exp_sum += exp_vals[j];
            }

            for (int j = 0; j < masked_QKt_h[0].size(); ++j) {
                atten_weights_h[i][j] = exp_vals[j] / exp_sum;
            }
        }

        // 5. 计算当前头的Attention输出 (维度：seq_len × head_size)
        heads_output[head_idx] = linear_.matMul(atten_weights_h, V_h);
    }

    // ========== 拼接所有头的输出 ==========
    std::vector<std::vector<double>> atten_output(seq_len, std::vector<double>(hidden_dim, 0.0));
    for (int head_idx = 0; head_idx < num_heads; ++head_idx) {
        int start_dim = head_idx * head_size;
        for (int seq_idx = 0; seq_idx < seq_len; ++seq_idx) {
            for (int dim_idx = 0; dim_idx < head_size; ++dim_idx) {
                atten_output[seq_idx][start_dim + dim_idx] = heads_output[head_idx][seq_idx][dim_idx];
            }
        }
    }
    // 保存中间结果以调试
    // file_.writeMat(atten_output, "Pt/Output");

    return atten_output;
}

std::vector<std::vector<double>> Model::swin_pt(const std::vector<std::vector<double>>& input_mat, 
    const int input_row, const int input_col, const int num_heads, const int window_size, const int shift_size,
    const std::vector<std::vector<double>>& Wq, const std::vector<double>& Bq,
    const std::vector<std::vector<double>>& Wk, const std::vector<double>& Bk,
    const std::vector<std::vector<double>>& Wv, const std::vector<double>& Bv,
    const std::vector<std::vector<double>>& position_bias_table){
    const int input_total = input_mat.size(), hidden_dim = input_mat[0].size();
    const int head_size = hidden_dim / num_heads;
    const int window_total_size = window_size * window_size;
    const int window_num = input_total / window_total_size;
    // double max_input = -10, min_input = 10, max_sum = -10, min_sum = 10;

    if (input_total != input_row * input_col) {
        throw std::invalid_argument("[swin_pt] error : Input total size mismatch with row*col!");
    }
    if (input_total % window_total_size != 0) {
        throw std::invalid_argument("[swin_pt] error : Input total size not divisible by window size!");
    }
    // 1. Cyclic Shift
    std::vector<std::vector<double>> shifted_mat(input_total, std::vector<double>(hidden_dim));
    for (int idx = 0; idx < input_total; ++idx) {
        int row = idx / input_col;
        int col = idx % input_col;
        int new_row = (row - shift_size + input_row) % input_row;
        int new_col = (col - shift_size + input_col) % input_col;
        int new_idx = new_row * input_col + new_col;
        shifted_mat[new_idx] = input_mat[idx];
    }
    // 2. Position Bias
    std::vector<std::vector<std::vector<double>>> heads_pos_bias;
    heads_pos_bias = get_swin_pos_bias(window_size, position_bias_table);
    if (heads_pos_bias.size() != num_heads || 
        heads_pos_bias[0].size() != window_total_size || 
        heads_pos_bias[0][0].size() != window_total_size) {
        throw std::invalid_argument("[swin_pt] error : Position bias dimension mismatch!");
    }
    // 3. window partition
    std::vector<std::vector<std::vector<double>>> windows_input_mat = window_partition_pt(
        shifted_mat, input_row, input_col, hidden_dim, window_size);
    std::vector<std::vector<std::vector<bool>>> windows_input_mask = get_attn_mask(
        input_row, input_col, window_size, shift_size);
    
    // 4. windows atten
    for(int window_ind = 0; window_ind < window_num; window_ind++){
        // file_.writeMat(windows_input_mat[window_ind], "Pt_Input_Window_" + std::to_string(window_ind));
        // ==========  计算 Q 和 K，并拆分多头 ==========
        std::vector<std::vector<double>> Q = linear_.matAddBias(linear_.matMul(windows_input_mat[window_ind], Wq), Bq);
        std::vector<std::vector<double>> K = linear_.matAddBias(linear_.matMul(windows_input_mat[window_ind], Wk), Bk);
        std::vector<std::vector<double>> V = linear_.matAddBias(linear_.matMul(windows_input_mat[window_ind], Wv), Bv);

        std::vector<std::vector<std::vector<double>>> heads_Q(num_heads, 
            std::vector<std::vector<double>>(window_total_size, std::vector<double>(head_size, 0.0)));
        std::vector<std::vector<std::vector<double>>> heads_K(num_heads, 
            std::vector<std::vector<double>>(window_total_size, std::vector<double>(head_size, 0.0)));
        std::vector<std::vector<std::vector<double>>> heads_V(num_heads, 
            std::vector<std::vector<double>>(window_total_size, std::vector<double>(head_size, 0.0)));
        for (int head_idx = 0; head_idx < num_heads; ++head_idx) {
            int start_dim = head_idx * head_size;
            for (int seq_idx = 0; seq_idx < window_total_size; ++seq_idx) {
                for (int dim_idx = 0; dim_idx < head_size; ++dim_idx) {
                    heads_Q[head_idx][seq_idx][dim_idx] = Q[seq_idx][start_dim + dim_idx];
                    heads_K[head_idx][seq_idx][dim_idx] = K[seq_idx][start_dim + dim_idx];
                    heads_V[head_idx][seq_idx][dim_idx] = V[seq_idx][start_dim + dim_idx];
                }
            }
        }
        // ==========  遍历每个头计算注意力 ==========
        std::vector<std::vector<std::vector<double>>> heads_output(num_heads, 
            std::vector<std::vector<double>>(window_total_size, std::vector<double>(head_size, 0.0)));
        for (int head_idx = 0; head_idx < num_heads; ++head_idx) {
            const auto& Q_h = heads_Q[head_idx];
            const auto& K_h = heads_K[head_idx];
            const auto& V_h = heads_V[head_idx];
            int k_rows = K_h.size();
            int k_cols = K_h[0].size();
            std::vector<std::vector<double>> Kt_h(k_cols, std::vector<double>(k_rows, 0.0));
            for (int i = 0; i < k_rows; ++i)
                for (int j = 0; j < k_cols; ++j) 
                    Kt_h[j][i] = K_h[i][j];
            std::vector<std::vector<double>> QKt_h = linear_.matMul(Q_h, Kt_h);
            std::vector<std::vector<double>> scaled_QKt_h(QKt_h.size(), std::vector<double>(QKt_h[0].size(), 0.0));
            double scale_factor = sqrt(head_size);
            for (int i = 0; i < QKt_h.size(); ++i){
                for (int j = 0; j < QKt_h[0].size(); ++j){
                    scaled_QKt_h[i][j] = QKt_h[i][j] / scale_factor;
                    scaled_QKt_h[i][j] += heads_pos_bias[head_idx][i][j];
                }
            }
            // 应用Attention Mask
            std::vector<std::vector<double>> masked_QKt_h(scaled_QKt_h.size(), std::vector<double>(scaled_QKt_h[0].size(), 0.0));
            for (int i = 0; i < scaled_QKt_h.size(); ++i){
                for (int j = 0; j < scaled_QKt_h[0].size(); ++j){
                    if (windows_input_mask[window_ind][i][j]) {
                        masked_QKt_h[i][j] = scaled_QKt_h[i][j];
                        // masked_QKt_h[i][j] -= 6;
                        // max_input = std::max(max_input, masked_QKt_h[i][j]);
                        // min_input = std::min(min_input, masked_QKt_h[i][j]);
                    } else {
                        masked_QKt_h[i][j] = -100; // 对齐源码设计
                    }
                }
            }
            //  计算 Softmax
            std::vector<std::vector<double>> atten_weights_h(masked_QKt_h.size(), std::vector<double>(masked_QKt_h[0].size(), 0.0));
            for (int i = 0; i < masked_QKt_h.size(); ++i) {
                double row_max = *std::max_element(masked_QKt_h[i].begin(), masked_QKt_h[i].end());
                double exp_sum = 0.0;
                std::vector<double> exp_vals(masked_QKt_h[0].size());
                for (int j = 0; j < masked_QKt_h[0].size(); ++j) {
                    exp_vals[j] = exp(masked_QKt_h[i][j] - row_max);
                    // exp_vals[j] = exp(masked_QKt_h[i][j]);
                    exp_sum += exp_vals[j];
                }
                // max_sum = std::max(max_sum, exp_sum);
                // min_sum = std::min(min_sum, exp_sum);
                for (int j = 0; j < masked_QKt_h[0].size(); ++j) {
                    atten_weights_h[i][j] = exp_vals[j] / exp_sum;
                }
            }
            // 计算当前头的Attention输出 
            heads_output[head_idx] = linear_.matMul(atten_weights_h, V_h);
        }
        // ========== 拼接所有头的输出 ==========
        std::vector<std::vector<double>> atten_output(window_total_size, std::vector<double>(hidden_dim, 0.0));
        for (int head_idx = 0; head_idx < num_heads; ++head_idx) {
            int start_dim = head_idx * head_size;
            for (int seq_idx = 0; seq_idx < window_total_size; ++seq_idx) {
                for (int dim_idx = 0; dim_idx < head_size; ++dim_idx) {
                    atten_output[seq_idx][start_dim + dim_idx] = heads_output[head_idx][seq_idx][dim_idx];
                }
            }
        }
        windows_input_mat[window_ind] = atten_output;
    }

    // 5. Windows Reverse
    std::vector<std::vector<double>> attn_output_mat = window_reverse_pt(
        windows_input_mat, input_row, input_col, hidden_dim, window_size);
    // 6. Reverse Shift
    std::vector<std::vector<double>> output_mat(input_total);
    for (int idx = 0; idx < input_total; ++idx) {
        int shifted_row = idx / input_col;
        int shifted_col = idx % input_col;
        int orig_row = (shifted_row + shift_size) % input_row;
        int orig_col = (shifted_col + shift_size) % input_col;
        int orig_idx = orig_row * input_col + orig_col;
        output_mat[orig_idx] = attn_output_mat[idx];
    }
    // std::cout << "  >> max_input = " << max_input << std::endl;
    // std::cout << "  >> min_input = " << min_input << std::endl;
    // std::cout << "  >> max_sum = " << max_sum << std::endl;
    // std::cout << "  >> min_sum = " << min_sum << std::endl;
    return output_mat;
}

std::vector<std::vector<std::vector<double>>> Model::get_swin_pos_bias(
    const int window_size, const std::vector<std::vector<double>>& pos_bias_table){
    const int bias_len = window_size * window_size;
    const int head_num = pos_bias_table[0].size();
    const int rel_pos_num = (2 * window_size - 1) * (2 * window_size - 1);
    if (pos_bias_table.size() != rel_pos_num) {
        throw std::invalid_argument("[get_swin_pos_bias] error : pos_bias_table的行数必须等于(2*window_size-1)^2");
    }
    std::vector<std::vector<int>> pos_index(bias_len, std::vector<int>(bias_len));
    // 1. 计算 pos_index
    std::vector<std::pair<int, int>> pos2coord(bias_len);
    for (int pos = 0; pos < bias_len; ++pos) {
        int i = pos / window_size;
        int j = pos % window_size;
        pos2coord[pos] = {i, j};
    }
    for (int pos1 = 0; pos1 < bias_len; ++pos1) {
        auto [i1, j1] = pos2coord[pos1];
        for (int pos2 = 0; pos2 < bias_len; ++pos2) {
            auto [i2, j2] = pos2coord[pos2];
            int dx = i1 - i2;
            int dy = j1 - j2;
            dx += (window_size - 1);
            dy += (window_size - 1);
            pos_index[pos1][pos2] = dx * (2 * window_size - 1) + dy;
        }
    }
    // 2. 每个头查表
    std::vector<std::vector<std::vector<double>>> output(head_num, 
        std::vector<std::vector<double>>(bias_len, std::vector<double>(bias_len)));
    for(int h = 0; h < head_num; ++h) {
        for(int pos1 = 0; pos1 < bias_len; ++pos1) {
            for(int pos2 = 0; pos2 < bias_len; ++pos2) {
                int index = pos_index[pos1][pos2];
                output[h][pos1][pos2] = pos_bias_table[index][h];
            }
        }
    }
    return output;
}

std::vector<std::vector<std::vector<double>>> Model::window_partition_pt(std::vector<std::vector<double>> input, 
    const int row_len, const int col_len, const int hidden_dim, const int window_size){
    // 1. 参数合法性校验
    if (input.size() != row_len * col_len)
        throw std::invalid_argument("[window_partition_pt] error: input size must equal row_len * col_len");
    if (input[0].size() != hidden_dim) 
        throw std::invalid_argument("[window_partition_pt] error: each row of input must have hidden_dim elements");
    if (row_len % window_size != 0 || col_len % window_size != 0)
        throw std::invalid_argument("[window_partition_pt] error: row_len/col_len must be divisible by window_size");

    // 2. 计算窗口数量
    const int num_windows_row = row_len / window_size;
    const int num_windows_col = col_len / window_size;
    const int window_num = num_windows_row * num_windows_col;

    // 3. 初始化结果：[window_num, window_size*window_size, hidden_dim]
    std::vector<std::vector<std::vector<double>>> result(window_num,
        std::vector<std::vector<double>>(window_size * window_size, std::vector<double>(hidden_dim, 0.0)));
    // 4. 遍历每个窗口，填充数据
    int window_idx = 0;
    for (int i = 0; i < num_windows_row; ++i) {
        for (int j = 0; j < num_windows_col; ++j) { 
            int elem_idx_in_window = 0;
            for (int y = 0; y < window_size; ++y) {
                const int row = i * window_size + y;
                for (int x = 0; x < window_size; ++x) {
                    const int col = j * window_size + x;
                    const int input_idx = row * col_len + col;
                    result[window_idx][elem_idx_in_window] = input[input_idx];
                    elem_idx_in_window++;
                }
            }
            window_idx++;
        }
    }
    return result;
}

std::vector<std::vector<double>> Model::window_reverse_pt(
    std::vector<std::vector<std::vector<double>>> input_windows, 
    const int row_len, const int col_len, const int hidden_dim, const int window_size){
    // 计算理论窗口数量：行方向窗口数 × 列方向窗口数
    const int num_windows_row = row_len / window_size;
    const int num_windows_col = col_len / window_size;
    const int expected_window_num = num_windows_row * num_windows_col;
    const int expected_window_elem_num = window_size * window_size;
    
    // 校验窗口数量是否匹配
    if (input_windows.size() != expected_window_num) {
        throw std::invalid_argument(
            "[window_reverse_pt] error : input_windows size(" + std::to_string(input_windows.size()) + 
            ") must equal " + std::to_string(expected_window_num)
        );
    }
    // 校验原始特征图尺寸是否能被窗口大小整除
    if (row_len % window_size != 0 || col_len % window_size != 0) {
        throw std::invalid_argument("[window_reverse_pt] error : row_len/col_len must be divisible by window_size");
    }
    // 2. 初始化输出：[row_len*col_len, hidden_dim]（原始特征图展平格式）
    std::vector<std::vector<double>> output(
        row_len * col_len, 
        std::vector<double>(hidden_dim, 0.0)
    );
    // 3. 逆向窗口划分：将每个窗口的元素放回原始位置
    int window_idx = 0; 
    for (int i = 0; i < num_windows_row; ++i) {
        for (int j = 0; j < num_windows_col; ++j) {
            for (int elem_idx_in_window = 0; elem_idx_in_window < expected_window_elem_num; ++elem_idx_in_window) {
                const int y = elem_idx_in_window / window_size;
                const int x = elem_idx_in_window % window_size;

                const int global_row = i * window_size + y;
                const int global_col = j * window_size + x;

                const int output_idx = global_row * col_len + global_col;

                output[output_idx] = input_windows[window_idx][elem_idx_in_window];
            }
            window_idx++;
        }
    }
    return output;
}

std::vector<std::vector<std::vector<bool>>> Model::get_attn_mask(const int row_len, const int col_len, 
    const int window_size, const int shift_size){
    // 1. 偏移量≤0时，无需掩码，返回空数组
    if (shift_size <= 0) {
        return std::vector<std::vector<std::vector<bool>>>();
    }

    // 2. 初始化img_mask：[row_len][col_len]，用数值标记不同区域
    std::vector<std::vector<int>> img_mask(row_len, std::vector<int>(col_len, 0));

    // 3. 生成行/列的3个切片（对应Python的height_slices/width_slices）
    // 切片规则：[0, row_len-window_size), [row_len-window_size, row_len-shift_size), [row_len-shift_size, row_len)
    auto make_slices = [](int len, int window_size, int shift_size) -> std::vector<std::pair<int, int>> {
        std::vector<std::pair<int, int>> slices;
        slices.emplace_back(0, len - window_size);
        slices.emplace_back(len - window_size, len - shift_size);
        slices.emplace_back(len - shift_size, len);
        return slices;
    };
    const auto row_slices = make_slices(row_len, window_size, shift_size);
    const auto col_slices = make_slices(col_len, window_size, shift_size);

    // 4. 遍历3x3切片，给每个区域分配唯一标记值（0~8）
    int count = 0;
    for (const auto& row_slice : row_slices) {
        const int row_start = row_slice.first;
        const int row_end = row_slice.second;
        for (const auto& col_slice : col_slices) {
            const int col_start = col_slice.first;
            const int col_end = col_slice.second;
            // 给当前区域赋值count
            for (int r = row_start; r < row_end; ++r) {
                for (int c = col_start; c < col_end; ++c) {
                    img_mask[r][c] = count;
                }
            }
            count++;
        }
    }

    // 5. 将img_mask展平为 [row_len*col_len, 1] 格式，适配window_partition输入
    std::vector<std::vector<double>> img_mask_flat(row_len * col_len, std::vector<double>(1, 0.0));
    for (int r = 0; r < row_len; ++r) {
        for (int c = 0; c < col_len; ++c) {
            const int idx = r * col_len + c;
            img_mask_flat[idx][0] = static_cast<double>(img_mask[r][c]);
        }
    }

    // 6. 调用window_partition切割img_mask，得到 [window_num, window_size*window_size, 1]
    auto mask_windows = window_partition_pt(img_mask_flat, row_len, col_len, 1, window_size);
    const int window_num = mask_windows.size();
    const int window_elem_num = window_size * window_size;

    // 7. 生成布尔型注意力掩码
    std::vector<std::vector<std::vector<bool>>> attn_mask(
        window_num,
        std::vector<std::vector<bool>>(window_elem_num, std::vector<bool>(window_elem_num, false))
    );

    for (int w = 0; w < window_num; ++w)
        for (int i = 0; i < window_elem_num; ++i) 
            for (int j = 0; j < window_elem_num; ++j) 
                attn_mask[w][i][j] = (mask_windows[w][i][0] == mask_windows[w][j][0]);

    return attn_mask;
}

