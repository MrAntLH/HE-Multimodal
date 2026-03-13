#include "ckks_evaluator.cuh"
#include "nonlinear.cuh"
#include "linear.cuh"
#include "model.cuh"
#include "file_handler.cuh"
#include "cpp_timer.cuh"

#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>

#include <cstdlib>
#include <ctime>

int main() {
    try {
        std::cout << "[main] start\n";
        cudaSetDevice(0);

        // ------------------- context & keys -------------------
        std::cout << "[main] creating context\n";
        heongpu::HEContext<heongpu::Scheme::CKKS> ctx(
            heongpu::keyswitching_type::KEYSWITCHING_METHOD_II,
            heongpu::sec_level_type::none);
        ctx.set_poly_modulus_degree(16384);
        ctx.set_coeff_modulus_bit_sizes(
            {60, 50, 50, 50, 50, 50, 50, 50, 50, 50,
                50, 50, 50, 50, 50, 50, 50, 50, 50, 50,
                50, 50, 50, 50, 50, 50, 50, 50, 50, 50},
            {60, 60, 60});
        ctx.generate();

        double scale = std::pow(2.0, 50);
        const int slot_count = ctx.get_poly_modulus_degree() / 2;

        std::cout << "[main] generating keys\n";
        heongpu::HEKeyGenerator<heongpu::Scheme::CKKS> keygen(ctx);
        heongpu::Secretkey<heongpu::Scheme::CKKS> secret_key(ctx, 16);
        keygen.generate_secret_key(secret_key);

        heongpu::Publickey<heongpu::Scheme::CKKS> public_key(ctx);
        keygen.generate_public_key(public_key, secret_key);

        heongpu::Relinkey<heongpu::Scheme::CKKS> relin_key(ctx);
        keygen.generate_relin_key(relin_key, secret_key);
        std::cout << "[main] keys generated\n";

        std::cout << "[main] create encoder/encryptor/decryptor/operators\n";
        heongpu::HEEncoder<heongpu::Scheme::CKKS> encoder(ctx);
        heongpu::HEEncryptor<heongpu::Scheme::CKKS> encryptor(ctx, public_key);
        heongpu::HEDecryptor<heongpu::Scheme::CKKS> decryptor(ctx, secret_key);
        heongpu::HEArithmeticOperator<heongpu::Scheme::CKKS> operators(ctx, encoder);

        std::cout << "[main] generates galois key for bootstrapping\n";
        // CtoS_piece_ = [2,5]
        // StoC_piece_ = [2,5]
        // taylor_number_ = [6,15]
        // less_key_mode_ = true or false
        heongpu::BootstrappingConfig boot_config(3, 3, 7, false);
        operators.generate_bootstrapping_params(
            scale, boot_config,
            heongpu::arithmetic_bootstrapping_type::SLIM_BOOTSTRAPPING);
        std::vector<int> key_index = operators.bootstrapping_key_indexs();
        heongpu::Galoiskey<heongpu::Scheme::CKKS> galois_key(
            ctx, key_index);
        keygen.generate_galois_key(galois_key,secret_key);

        // ------------------- CKKSEvaluator -------------------
        std::cout << "[main] constructing CKKSEvaluator \n\n";

        CKKSEvaluator evaluator_(ctx, operators, encoder, encryptor, decryptor, relin_key, galois_key, scale, boot_config);
        fileHandler file_(ctx, encoder, evaluator_, scale);
        NonLinear nonLinear_(ctx, encoder, evaluator_, file_, scale);
        Linear linear_(ctx, encoder, evaluator_, scale);
        Model model_(ctx, encoder, evaluator_, linear_, nonLinear_, file_, scale);

        // ------------------- Test Option -------------------------
        const std::vector<bool> test_option = {false, false, true, true};

        // ------------------- [M1] Bert Cipher Norm Version -------------------
        if( test_option[0] ){
            std::cout << "[M1] [Model] run bert_ct_norm()\n";
            try{
                const int seq_len = 128, embed_dim = 768, num_heads = 12;
                // const int per_block_seq = slot_count / embed_dim;
                const int per_block_seq = 8;
                const int padded_seq_len = ((seq_len + per_block_seq - 1) / per_block_seq) * per_block_seq;
                const int block_num = padded_seq_len / per_block_seq;
                const int head_size = embed_dim / num_heads;
                std::cout << "=============== 配置说明 =================" << std::endl;
                std::cout << " >> 输入长度 : " << seq_len << std::endl;
                std::cout << " >> 特征维度 : " << embed_dim << std::endl;
                std::cout << " >> 多头个数 : " << num_heads << std::endl;
                std::cout << " >> 密文空间 : " << slot_count << std::endl;
                std::cout << "=============== 准备数据 =================" << std::endl;
                // 1. 准备输入数据
                std::vector<std::vector<double>> input_mat(seq_len, std::vector<double>(embed_dim, 0));
                std::vector<std::vector<double>> weight_query, weight_key, weight_value;
                std::vector<double> bias_query, bias_key, bias_value;
                std::vector<std::vector<bool>> atten_mask(seq_len, std::vector<bool>(seq_len, false));
                {
                    input_mat = file_.readMat("TextBackbone/input/input_mat_128");
                    atten_mask = file_.readMask("TextBackbone/input/input_mask_128");

                    weight_query = file_.readMat("TextBackbone/weight/query_weight");
                    weight_key = file_.readMat("TextBackbone/weight/key_weight");
                    weight_value = file_.readMat("TextBackbone/weight/value_weight");

                    bias_query = file_.readVec("TextBackbone/weight/query_bias");
                    bias_key = file_.readVec("TextBackbone/weight/key_bias");
                    bias_value = file_.readVec("TextBackbone/weight/value_bias");
                }
                if(padded_seq_len > seq_len)
                    std::cout << " >> 输入填充后为 : " << padded_seq_len << std::endl;
                std::cout << " >> 密文容纳长度 : " << per_block_seq << std::endl;
                std::cout << " >> 分块后密文数 : " << block_num << std::endl;
                if( padded_seq_len > seq_len ){
                    input_mat.resize(padded_seq_len, std::vector<double>(embed_dim, 0.0));
                    atten_mask.resize(padded_seq_len, std::vector<bool>(padded_seq_len, false));
                    for (auto& row : atten_mask)
                        row.resize(padded_seq_len, false);
                }
                // 展示输入
                {
                    evaluator_.print_matrix(input_mat, 
                        " Input Matrix [ " + std::to_string(input_mat.size()) + 
                        " x " + std::to_string(input_mat[0].size()) + " ]:", 
                        input_mat.size(), input_mat[0].size(), 4, 8, 3);
                    evaluator_.print_matrix(atten_mask, 
                        " Input Mask [ " + std::to_string(atten_mask.size()) + 
                        " x " + std::to_string(atten_mask[0].size()) + " ]:", 
                        atten_mask.size(), atten_mask[0].size(), 4, 8);
                    
                }
                // 2. 分块，并打包加密输入
                std::vector<heongpu::Ciphertext<heongpu::Scheme::CKKS>> input_cts(block_num);
                {
                    heongpu::Plaintext<heongpu::Scheme::CKKS> input_pt;
                    for(int i = 0; i < block_num; i++){
                        input_pt = linear_.pack_matrix_by_row(
                            std::vector<std::vector<double>>(
                                input_mat.begin() + i * per_block_seq,
                                input_mat.begin() + i * per_block_seq + per_block_seq),
                                per_block_seq, embed_dim);
                        evaluator_.encrypt(input_pt, input_cts[i]);
                    }
                }
                std::cout << "=============== 密文推理 =================" << std::endl;
                // 3. 执行 Bert Attention 推理
                std::vector<heongpu::Ciphertext<heongpu::Scheme::CKKS>> output_cts(block_num);
                std::atomic<long long> total_time(0LL), q_time(0LL), k_time(0LL), 
                    score_tim(0LL), softmax_time(0LL), v_time(0LL), mulV_time(0LL);
                {
                    CppTimer timer(&total_time);
                    output_cts = model_.multiHeadAtten_ct(
                        input_cts, padded_seq_len, embed_dim, num_heads, atten_mask,
                        weight_query, bias_query, weight_key, bias_key, weight_value, bias_value,
                        std::sqrt(static_cast<double>(head_size)), 2, 1.0,
                        &q_time, &k_time, &score_tim, &softmax_time, &v_time, &mulV_time);
                }

                std::cout << "=============== 密文解密 =================" << std::endl;
                std::vector<std::vector<double>> predict_output_mat;
                {
                    for(int block_ind = 0; block_ind < block_num; block_ind++){
                        heongpu::Plaintext<heongpu::Scheme::CKKS> temp_pt;
                        decryptor.decrypt(temp_pt, output_cts[block_ind]);
                        std::vector<std::vector<double>> temp_mat = linear_.unpack_matrix_by_row(
                            temp_pt, per_block_seq, embed_dim, per_block_seq, embed_dim);
                        for(int seq_ind = 0; seq_ind < per_block_seq; seq_ind++)
                            predict_output_mat.emplace_back(temp_mat[seq_ind]);
                    }
                    predict_output_mat.resize(seq_len);
                }
                evaluator_.print_matrix(predict_output_mat, 
                    "The predict result: ", predict_output_mat.size(), predict_output_mat[0].size(), 6, 8, 4);
                std::cout << "=============== 明文推理 =================" << std::endl;
                // 5. 计算正确的结果
                std::vector<std::vector<double>> true_output_mat = model_.multiHeadAtten_pt(
                    input_mat, seq_len, embed_dim, num_heads, atten_mask,
                    weight_query, bias_query, weight_key, bias_key, weight_value, bias_value);
                evaluator_.print_matrix(true_output_mat, 
                    "The true result: ", true_output_mat.size(), true_output_mat[0].size(), 6, 8, 4);

                std::cout << "=============== 评测指标 =================" << std::endl;
                std::cout << "[M1] bert_ct_norm() MAE: " << std::fixed << std::setprecision(4) << 
                    evaluator_.calculate_MAE(predict_output_mat, true_output_mat) << std::endl;
                std::cout << "[M1] bert_ct_norm() Q time: " << std::fixed << std::setprecision(3)
                    << q_time.load() / 1000.0 << " s\n";
                std::cout << "[M1] bert_ct_norm() K time: " << std::fixed << std::setprecision(3)
                    << k_time.load() / 1000.0 << " s\n";
                std::cout << "[M1] bert_ct_norm() Score time: " << std::fixed << std::setprecision(3)
                    << score_tim.load() / 1000.0 << " s\n";
                std::cout << "[M1] bert_ct_norm() Softmax time: " << std::fixed << std::setprecision(3)
                    << softmax_time.load() / 1000.0 << " s\n";
                std::cout << "[M1] bert_ct_norm() V time: " << std::fixed << std::setprecision(3)
                    << v_time.load() / 1000.0 << " s\n";
                std::cout << "[M1] bert_ct_norm() MulV time: " << std::fixed << std::setprecision(3)
                    << mulV_time.load() / 1000.0 << " s\n";
                std::cout << "[M1] bert_ct_norm() Total time: " << std::fixed << std::setprecision(3)
                    << total_time.load() / 1000.0 << " s\n";
            }catch (const std::exception &e) {
                std::cerr << "[ERROR] exception during bert_ct_norm(): " << e.what() << std::endl;
            }
        }
        
        // ------------------- [M2] Bert Cipher Opti Version  -------------------
        if( test_option[1] ){
            std::cout << "[M2] [Model] run bert_ct_opti()\n";
            try{
                const int seq_len = 128, embed_dim = 768, num_heads = 12;
                // const int per_block_seq = slot_count / embed_dim;
                const int per_block_seq = 8;
                const int padded_seq_len = ((seq_len + per_block_seq - 1) / per_block_seq) * per_block_seq;
                const int block_num = padded_seq_len / per_block_seq;
                const int head_size = embed_dim / num_heads;
                std::cout << "=============== 配置说明 =================" << std::endl;
                std::cout << " >> 输入长度 : " << seq_len << std::endl;
                std::cout << " >> 特征维度 : " << embed_dim << std::endl;
                std::cout << " >> 多头个数 : " << num_heads << std::endl;
                std::cout << " >> 密文空间 : " << slot_count << std::endl;
                std::cout << "=============== 准备数据 =================" << std::endl;
                // 1. 准备输入数据
                std::vector<std::vector<double>> input_mat(seq_len, std::vector<double>(embed_dim, 0));
                std::vector<std::vector<double>> weight_query, weight_key, weight_value;
                std::vector<double> bias_query, bias_key, bias_value;
                std::vector<std::vector<bool>> atten_mask(seq_len, std::vector<bool>(seq_len, false));
                {
                    input_mat = file_.readMat("TextBackbone/input/input_mat_128");
                    atten_mask = file_.readMask("TextBackbone/input/input_mask_128");

                    weight_query = file_.readMat("TextBackbone/weight/query_weight");
                    weight_key = file_.readMat("TextBackbone/weight/key_weight");
                    weight_value = file_.readMat("TextBackbone/weight/value_weight");

                    bias_query = file_.readVec("TextBackbone/weight/query_bias");
                    bias_key = file_.readVec("TextBackbone/weight/key_bias");
                    bias_value = file_.readVec("TextBackbone/weight/value_bias");
                }
                std::cout << " >> 密文容纳长度 : " << per_block_seq << std::endl;
                std::cout << " >> 分块后密文数 : " << block_num << std::endl;
                if( padded_seq_len > seq_len ){
                    std::cout << " >> 输入填充后为 : " << padded_seq_len << std::endl;
                    input_mat.resize(padded_seq_len, std::vector<double>(embed_dim, 0.0));
                    atten_mask.resize(padded_seq_len, std::vector<bool>(padded_seq_len, false));
                    for (auto& row : atten_mask)
                        row.resize(padded_seq_len, false);
                }
                // 展示输入
                {
                    evaluator_.print_matrix(input_mat, 
                        " Input Matrix [ " + std::to_string(input_mat.size()) + 
                        " x " + std::to_string(input_mat[0].size()) + " ]:", 
                        input_mat.size(), input_mat[0].size(), 4, 8, 3);
                    evaluator_.print_matrix(atten_mask, 
                        " Input Mask [ " + std::to_string(atten_mask.size()) + 
                        " x " + std::to_string(atten_mask[0].size()) + " ]:", 
                        atten_mask.size(), atten_mask[0].size(), 4, 8);
                    
                }
                // 2. 分块，并打包加密输入
                std::vector<heongpu::Ciphertext<heongpu::Scheme::CKKS>> input_cts(block_num);
                {
                    heongpu::Plaintext<heongpu::Scheme::CKKS> input_pt;
                    for(int i = 0; i < block_num; i++){
                        input_pt = linear_.pack_matrix_by_row(
                            std::vector<std::vector<double>>(
                                input_mat.begin() + i * per_block_seq,
                                input_mat.begin() + i * per_block_seq + per_block_seq),
                            per_block_seq, embed_dim);
                        evaluator_.encrypt(input_pt, input_cts[i]);
                    }
                }
                std::cout << "=============== 密文推理 =================" << std::endl;
                // 3. 执行 Bert Attention 推理
                std::vector<heongpu::Ciphertext<heongpu::Scheme::CKKS>> output_cts(block_num);
                std::atomic<long long> total_time(0LL), q_time(0LL), k_time(0LL), 
                    score_tim(0LL), softmax_time(0LL), v_time(0LL), mulV_time(0LL);

                {
                    CppTimer timer(&total_time);
                    output_cts = model_.bert_ct_opti(
                        input_cts, padded_seq_len, embed_dim, num_heads, atten_mask,
                        weight_query, bias_query, weight_key, bias_key, weight_value, bias_value,
                        std::sqrt(static_cast<double>(head_size)), 2, 1.0,
                        &q_time, &k_time, &score_tim, &softmax_time, &v_time, &mulV_time);
                }
                std::cout << "=============== 密文解密 =================" << std::endl;
                // 4. 解包预测结果
                std::vector<std::vector<double>> predict_output_mat;
                {
                    for(int block_ind = 0; block_ind < block_num; block_ind++){
                        heongpu::Plaintext<heongpu::Scheme::CKKS> temp_pt;
                        decryptor.decrypt(temp_pt, output_cts[block_ind]);
                        std::vector<std::vector<double>> temp_mat = linear_.unpack_matrix_by_row(
                            temp_pt, per_block_seq, embed_dim, per_block_seq, embed_dim);
                        for(int seq_ind = 0; seq_ind < per_block_seq; seq_ind++)
                            predict_output_mat.emplace_back(temp_mat[seq_ind]);
                    }
                    predict_output_mat.resize(seq_len);
                }
                evaluator_.print_matrix(predict_output_mat, 
                    "The predict result: ", predict_output_mat.size(), predict_output_mat[0].size(), 6, 8, 4);
                std::cout << "=============== 明文推理 =================" << std::endl;
                // 5. 计算正确的结果
                std::vector<std::vector<double>> true_output_mat = model_.multiHeadAtten_pt(
                    input_mat, seq_len, embed_dim, num_heads, atten_mask,
                    weight_query, bias_query, weight_key, bias_key, weight_value, bias_value);
                evaluator_.print_matrix(true_output_mat, 
                    "The true result: ", true_output_mat.size(), true_output_mat[0].size(), 6, 8, 4);

                std::cout << "=============== 评测指标 =================" << std::endl;
                std::cout << "[M2] bert_ct_opti() MAE: " << std::fixed << std::setprecision(4) << 
                    evaluator_.calculate_MAE(predict_output_mat, true_output_mat) << std::endl;
                std::cout << "[M2] bert_ct_opti() Q time: " << std::fixed << std::setprecision(3)
                    << q_time.load() / 1000.0 << " s\n";
                std::cout << "[M2] bert_ct_opti() K time: " << std::fixed << std::setprecision(3)
                    << k_time.load() / 1000.0 << " s\n";
                std::cout << "[M2] bert_ct_opti() Score time: " << std::fixed << std::setprecision(3)
                    << score_tim.load() / 1000.0 << " s\n";
                std::cout << "[M2] bert_ct_opti() Softmax time: " << std::fixed << std::setprecision(3)
                    << softmax_time.load() / 1000.0 << " s\n";
                std::cout << "[M2] bert_ct_opti() V time: " << std::fixed << std::setprecision(3)
                    << v_time.load() / 1000.0 << " s\n";
                std::cout << "[M2] bert_ct_opti() MulV time: " << std::fixed << std::setprecision(3)
                    << mulV_time.load() / 1000.0 << " s\n";
                std::cout << "[M2] bert_ct_opti() Total time: " << std::fixed << std::setprecision(3)
                    << total_time.load() / 1000.0 << " s\n";
            }catch (const std::exception &e) {
                std::cerr << "[ERROR] exception during bert_ct_opti(): " << e.what() << std::endl;
            }
        }
        
        // ------------------- [M3] Swin Cipher Norm Version  -------------------
        if( test_option[2] ){
            std::cout << "[M3] [Model] run swin_ct_norm()\n";
            try{
                const int input_row = 60, input_col = 60, embed_dim = 512, head_size = 32;
                const int input_num = input_row * input_col;
                const int per_block_seq = slot_count / embed_dim;
                // const int per_block_seq = 6;
                const int window_size = 12;
                const int block_num = input_num / per_block_seq;
                const int window_num = input_num / window_size / window_size;
                const int num_heads = embed_dim / head_size;
                std::cout << "=============== 配置说明 =================" << std::endl;
                std::cout << " >> 密文存储 : [ " << per_block_seq << " , " << embed_dim << " ]" << std::endl;
                std::cout << " >> 密文空间 : " << slot_count << std::endl;
                std::cout << "=============== 输入说明 =================" << std::endl;
                std::cout << " >> 输入维度 : [ " << input_num << " , " << embed_dim << " ]" << std::endl;
                std::cout << " >> 图尺寸 : [ " << input_row << " , " << input_col << " ]"<< std::endl;
                std::cout << " >> 分块后密文数 : " << block_num << std::endl;
                std::cout << " >> 窗口尺寸 : [ " << window_size << " , " << window_size << " ]" << std::endl; 
                std::cout << " >> 多头个数 : " << num_heads << std::endl;
                std::cout << " >> 窗口数 : " << window_num << std::endl;
                std::cout << "=============== 读取文件 =================" << std::endl;
                // 1. 准备输入数据
                std::vector<std::vector<double>> input_mat;
                std::vector<std::vector<double>> weight_query, weight_key, weight_value, position_bias_table;
                std::vector<double> bias_query, bias_key, bias_value;
                {
                    input_mat = file_.readMat("ImageBackbone/input/input_mat_" + 
                        std::to_string(input_row) + "_" + std::to_string(input_col) + "_" + std::to_string(embed_dim));
                    
                    weight_query = file_.readMat("ImageBackbone/weight/query_weight_" + 
                        std::to_string(embed_dim) + "_" + std::to_string(embed_dim));
                    weight_key = file_.readMat("ImageBackbone/weight/key_weight_" + 
                        std::to_string(embed_dim) + "_" + std::to_string(embed_dim));
                    weight_value = file_.readMat("ImageBackbone/weight/value_weight_" + 
                        std::to_string(embed_dim) + "_" + std::to_string(embed_dim));
                    position_bias_table = file_.readMat("ImageBackbone/weight/position_bias_table_529_" + 
                        std::to_string(num_heads));

                    bias_query = file_.readVec("ImageBackbone/weight/query_bias_" + std::to_string(embed_dim));
                    bias_key = file_.readVec("ImageBackbone/weight/key_bias_" + std::to_string(embed_dim));
                    bias_value = file_.readVec("ImageBackbone/weight/value_bias_" + std::to_string(embed_dim));
                }
                // 展示输入
                {
                    evaluator_.print_matrix(input_mat, 
                        " Input Matrix [ " + std::to_string(input_mat.size()) + 
                        " x " + std::to_string(input_mat[0].size()) + " ]:", 
                        input_mat.size(), input_mat[0].size(), 6, 8, 3);
                }

                // 2. 分块，并打包加密输入
                std::vector<heongpu::Ciphertext<heongpu::Scheme::CKKS>> input_cts(block_num);
                {
                    heongpu::Plaintext<heongpu::Scheme::CKKS> input_pt;
                    for(int i = 0; i < block_num; i++){
                        input_pt = linear_.pack_matrix_by_row(
                            std::vector<std::vector<double>>(
                                input_mat.begin() + i * per_block_seq,
                                input_mat.begin() + i * per_block_seq + per_block_seq),
                            per_block_seq, embed_dim);
                        evaluator_.encrypt(input_pt, input_cts[i]);
                    }
                }
                std::cout << "=============== 密文推理 =================" << std::endl;
                // 3. 执行推理
                std::vector<heongpu::Ciphertext<heongpu::Scheme::CKKS>> output_cts(block_num);
                std::atomic<long long> total_time(0LL), 
                    cyclic_shift_time(0LL), shift_reverse_time(0LL), window_partition_time(0LL), window_reverse_time(0LL),
                    q_time(0LL), k_time(0LL), score_tim(0LL), softmax_time(0LL), v_time(0LL), mulV_time(0LL);
                {
                    CppTimer timer(&total_time);
                    output_cts = model_.swin_ct_norm(input_cts, 
                        per_block_seq, embed_dim, num_heads, input_row, input_col, window_size, window_size / 2,
                        weight_query, bias_query, weight_key, bias_key, weight_value, bias_value, position_bias_table,
                        10.0, 3.0, 64.0,
                        &cyclic_shift_time, &shift_reverse_time, &window_partition_time, &window_reverse_time,
                        &q_time, &k_time, &score_tim, &softmax_time, &v_time, &mulV_time);
                }
                // file_.saveCtsMat(output_cts, "swin_ct_norm_" + std::to_string(embed_dim), per_block_seq, embed_dim);

                std::cout << "=============== 密文解密 =================" << std::endl;
                // 4. 解包预测结果
                std::vector<std::vector<double>> predict_output_mat;
                {
                    for(int block_ind = 0; block_ind < block_num; block_ind++){
                        heongpu::Plaintext<heongpu::Scheme::CKKS> temp_pt;
                        decryptor.decrypt(temp_pt, output_cts[block_ind]);
                        std::vector<std::vector<double>> temp_mat = linear_.unpack_matrix_by_row(
                            temp_pt, per_block_seq, embed_dim, per_block_seq, embed_dim);
                        for(int seq_ind = 0; seq_ind < per_block_seq; seq_ind++)
                            predict_output_mat.emplace_back(temp_mat[seq_ind]);
                    }
                    predict_output_mat.resize(input_num);
                }
                evaluator_.print_matrix(predict_output_mat, 
                    "The predict result: ", predict_output_mat.size(), predict_output_mat[0].size(), 6, 8, 4);
                std::cout << "=============== 明文推理 =================" << std::endl;
                // 5. 计算正确的结果
                std::vector<std::vector<double>> true_output_mat = model_.swin_pt(
                    input_mat, input_row, input_col, num_heads, window_size, window_size / 2,
                    weight_query, bias_query, weight_key, bias_key, weight_value, bias_value, position_bias_table);
                evaluator_.print_matrix(true_output_mat, 
                    "The true result: ", true_output_mat.size(), true_output_mat[0].size(), 6, 8, 4);

                // file_.writeMat(true_output_mat, "swin_pt_norm_" + std::to_string(embed_dim));

                std::cout << "=============== 评测指标 =================" << std::endl;
                std::cout << "[M3] swin_ct_norm() MAE: " << std::fixed << std::setprecision(4) << 
                    evaluator_.calculate_MAE(predict_output_mat, true_output_mat) << std::endl;
                std::cout << "[M3] swin_ct_norm() Cyclic Shift time: " << std::fixed << std::setprecision(3)
                    << cyclic_shift_time.load() / 1000.0 << " s\n";
                std::cout << "[M3] swin_ct_norm() Window Partition time: " << std::fixed << std::setprecision(3)
                    << window_partition_time.load() / 1000.0 << " s\n";   
                std::cout << "[M3] swin_ct_norm() Q time: " << std::fixed << std::setprecision(3)
                    << q_time.load() / 1000.0 << " s\n";
                std::cout << "[M3] swin_ct_norm() K time: " << std::fixed << std::setprecision(3)
                    << k_time.load() / 1000.0 << " s\n";
                std::cout << "[M3] swin_ct_norm() Score time: " << std::fixed << std::setprecision(3)
                    << score_tim.load() / 1000.0 << " s\n";
                std::cout << "[M3] swin_ct_norm() Softmax time: " << std::fixed << std::setprecision(3)
                    << softmax_time.load() / 1000.0 << " s\n";
                std::cout << "[M3] swin_ct_norm() V time: " << std::fixed << std::setprecision(3)
                    << v_time.load() / 1000.0 << " s\n";
                std::cout << "[M3] swin_ct_norm() MulV time: " << std::fixed << std::setprecision(3)
                    << mulV_time.load() / 1000.0 << " s\n";
                std::cout << "[M3] swin_ct_norm() Window Reverse time: " << std::fixed << std::setprecision(3)
                    << window_reverse_time.load() / 1000.0 << " s\n";
                std::cout << "[M3] swin_ct_norm() Reverse Shift time: " << std::fixed << std::setprecision(3)
                    << shift_reverse_time.load() / 1000.0 << " s\n";
                std::cout << "[M3] swin_ct_norm() Total time: " << std::fixed << std::setprecision(3)
                    << total_time.load() / 1000.0 << " s\n";
            }catch (const std::exception &e) {
                std::cerr << "[ERROR] exception during swin_ct_norm(): " << e.what() << std::endl;
            }
        }

        // ------------------- [M4] Swin Cipher Opti Version  -------------------
        if( test_option[3] ){
            std::cout << "[M4] [Model] run swin_ct_opti()\n";
            try{
                const int input_row = 60, input_col = 60, embed_dim = 512, head_size = 32;
                const int input_num = input_row * input_col;
                const int per_block_seq = slot_count / embed_dim;
                // const int per_block_seq = 6;
                const int window_size = 12;
                const int block_num = input_num / per_block_seq;
                const int window_num = input_num / window_size / window_size;
                const int num_heads = embed_dim / head_size;
                std::cout << "=============== 配置说明 =================" << std::endl;
                std::cout << " >> 密文存储 : [ " << per_block_seq << " , " << embed_dim << " ]" << std::endl;
                std::cout << " >> 密文空间 : " << slot_count << std::endl;
                std::cout << "=============== 输入说明 =================" << std::endl;
                std::cout << " >> 输入维度 : [ " << input_num << " , " << embed_dim << " ]" << std::endl;
                std::cout << " >> 图尺寸 : [ " << input_row << " , " << input_col << " ]"<< std::endl;
                std::cout << " >> 分块后密文数 : " << block_num << std::endl;
                std::cout << " >> 窗口尺寸 : [ " << window_size << " , " << window_size << " ]" << std::endl; 
                std::cout << " >> 多头个数 : " << num_heads << std::endl;
                std::cout << " >> 窗口数 : " << window_num << std::endl;
                std::cout << "=============== 读取文件 =================" << std::endl;
                // 1. 准备输入数据
                std::vector<std::vector<double>> input_mat;
                std::vector<std::vector<double>> weight_query, weight_key, weight_value, position_bias_table;
                std::vector<double> bias_query, bias_key, bias_value;
                {
                    input_mat = file_.readMat("ImageBackbone/input/input_mat_" + 
                        std::to_string(input_row) + "_" + std::to_string(input_col) + "_" + std::to_string(embed_dim));

                    weight_query = file_.readMat("ImageBackbone/weight/query_weight_" + 
                        std::to_string(embed_dim) + "_" + std::to_string(embed_dim));
                    weight_key = file_.readMat("ImageBackbone/weight/key_weight_" + 
                        std::to_string(embed_dim) + "_" + std::to_string(embed_dim));
                    weight_value = file_.readMat("ImageBackbone/weight/value_weight_" + 
                        std::to_string(embed_dim) + "_" + std::to_string(embed_dim));
                    position_bias_table = file_.readMat("ImageBackbone/weight/position_bias_table_529_" + 
                        std::to_string(num_heads));

                    bias_query = file_.readVec("ImageBackbone/weight/query_bias_" + std::to_string(embed_dim));
                    bias_key = file_.readVec("ImageBackbone/weight/key_bias_" + std::to_string(embed_dim));
                    bias_value = file_.readVec("ImageBackbone/weight/value_bias_" + std::to_string(embed_dim));
                }
                // 展示输入
                {
                    evaluator_.print_matrix(input_mat, 
                        " Input Matrix [ " + std::to_string(input_mat.size()) + 
                        " x " + std::to_string(input_mat[0].size()) + " ]:", 
                        input_mat.size(), input_mat[0].size(), 6, 8, 3);
                }

                // 2. 分块，并打包加密输入
                std::vector<heongpu::Ciphertext<heongpu::Scheme::CKKS>> input_cts(block_num);
                {
                    heongpu::Plaintext<heongpu::Scheme::CKKS> input_pt;
                    for(int i = 0; i < block_num; i++){
                        input_pt = linear_.pack_matrix_by_row(
                            std::vector<std::vector<double>>(
                                input_mat.begin() + i * per_block_seq,
                                input_mat.begin() + i * per_block_seq + per_block_seq),
                            per_block_seq, embed_dim);
                        evaluator_.encrypt(input_pt, input_cts[i]);
                    }
                }
                std::cout << "=============== 密文推理 =================" << std::endl;
                // 3. 执行推理
                std::vector<heongpu::Ciphertext<heongpu::Scheme::CKKS>> output_cts(block_num);
                std::atomic<long long> total_time(0LL), window_partition_time(0LL), window_reverse_time(0LL), 
                    q_time(0LL), k_time(0LL), score_tim(0LL), softmax_time(0LL), v_time(0LL), mulV_time(0LL);
                {
                    CppTimer timer(&total_time);
                    output_cts = model_.swin_ct_opti(input_cts, 
                        per_block_seq, embed_dim, num_heads, input_row, input_col, window_size, window_size / 2,
                        weight_query, bias_query, weight_key, bias_key, weight_value, bias_value, position_bias_table,
                        10.0, 3.0, 64.0,
                        &window_partition_time, &window_reverse_time, &q_time, &k_time, &score_tim, &softmax_time, &v_time, &mulV_time);
                }

                // file_.saveCtsMat(output_cts, "swin_ct_opti_" + std::to_string(embed_dim), per_block_seq, embed_dim);

                std::cout << "=============== 密文解密 =================" << std::endl;
                // 4. 解包预测结果
                std::vector<std::vector<double>> predict_output_mat;
                {
                    for(int block_ind = 0; block_ind < block_num; block_ind++){
                        heongpu::Plaintext<heongpu::Scheme::CKKS> temp_pt;
                        decryptor.decrypt(temp_pt, output_cts[block_ind]);
                        std::vector<std::vector<double>> temp_mat = linear_.unpack_matrix_by_row(
                            temp_pt, per_block_seq, embed_dim, per_block_seq, embed_dim);
                        for(int seq_ind = 0; seq_ind < per_block_seq; seq_ind++)
                            predict_output_mat.emplace_back(temp_mat[seq_ind]);
                    }
                    predict_output_mat.resize(input_num);
                }
                evaluator_.print_matrix(predict_output_mat, 
                    "The predict result: ", predict_output_mat.size(), predict_output_mat[0].size(), 6, 8, 4);
                std::cout << "=============== 明文推理 =================" << std::endl;
                // 5. 计算正确的结果
                std::vector<std::vector<double>> true_output_mat = model_.swin_pt(
                    input_mat, input_row, input_col, num_heads, window_size, window_size / 2,
                    weight_query, bias_query, weight_key, bias_key, weight_value, bias_value, position_bias_table);
                evaluator_.print_matrix(true_output_mat, 
                    "The true result: ", true_output_mat.size(), true_output_mat[0].size(), 6, 8, 4);
                // file_.writeMat(true_output_mat, "swin_pt_opti_" + std::to_string(embed_dim));

                std::cout << "=============== 评测指标 =================" << std::endl;
                std::cout << "[M4] swin_ct_opti() MAE: " << std::fixed << std::setprecision(4) << 
                    evaluator_.calculate_MAE(predict_output_mat, true_output_mat) << std::endl;
                std::cout << "[M4] swin_ct_opti() Window Partition time: " << std::fixed << std::setprecision(3)
                    << window_partition_time.load() / 1000.0 << " s\n";  
                std::cout << "[M4] swin_ct_opti() Q time: " << std::fixed << std::setprecision(3)
                    << q_time.load() / 1000.0 << " s\n";
                std::cout << "[M4] swin_ct_opti() K time: " << std::fixed << std::setprecision(3)
                    << k_time.load() / 1000.0 << " s\n";
                std::cout << "[M4] swin_ct_opti() Score time: " << std::fixed << std::setprecision(3)
                    << score_tim.load() / 1000.0 << " s\n";
                std::cout << "[M4] swin_ct_opti() Softmax time: " << std::fixed << std::setprecision(3)
                    << softmax_time.load() / 1000.0 << " s\n";
                std::cout << "[M4] swin_ct_opti() V time: " << std::fixed << std::setprecision(3)
                    << v_time.load() / 1000.0 << " s\n";
                std::cout << "[M4] swin_ct_opti() MulV time: " << std::fixed << std::setprecision(3)
                    << mulV_time.load() / 1000.0 << " s\n";
                std::cout << "[M4] swin_ct_opti() Window Reverse time: " << std::fixed << std::setprecision(3)
                    << window_reverse_time.load() / 1000.0 << " s\n";
                std::cout << "[M4] swin_ct_opti() Total time: " << std::fixed << std::setprecision(3)
                    << total_time.load() / 1000.0 << " s\n";
            }catch (const std::exception &e) {
                std::cerr << "[ERROR] exception during swin_ct_opti(): " << e.what() << std::endl;
            }
        }

        std::cout << "[main] all tests attempted, exiting normally\n";
    } catch (const std::exception &e) {
        std::cerr << "[FATAL] uncaught exception in main: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "[FATAL] unknown uncaught exception in main\n";
        return 2;
    }
    return 0;
}
