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

        std::cout << "[main] create encoder/encryptor/decryptor/operators\n";
        heongpu::HEEncoder<heongpu::Scheme::CKKS> encoder(ctx);
        heongpu::HEEncryptor<heongpu::Scheme::CKKS> encryptor(ctx, public_key);
        heongpu::HEDecryptor<heongpu::Scheme::CKKS> decryptor(ctx, secret_key);
        heongpu::HEArithmeticOperator<heongpu::Scheme::CKKS> operators(ctx, encoder);

        heongpu::BootstrappingConfig boot_config(3, 3, 7, false);
        operators.generate_bootstrapping_params(
            scale, boot_config,
            heongpu::arithmetic_bootstrapping_type::SLIM_BOOTSTRAPPING);
        std::vector<int> key_index = operators.bootstrapping_key_indexs();
        heongpu::Galoiskey<heongpu::Scheme::CKKS> galois_key(ctx, key_index);
        keygen.generate_galois_key(galois_key,secret_key);

        // ------------------- CKKSEvaluator -------------------
        CKKSEvaluator evaluator_(ctx, operators, encoder, encryptor, decryptor, relin_key, galois_key, scale, boot_config);
        fileHandler file_(ctx, encoder, evaluator_, scale);
        NonLinear nonLinear_(ctx, encoder, evaluator_, file_, scale);
        Linear linear_(ctx, encoder, evaluator_, scale);
        Model model_(ctx, encoder, evaluator_, linear_, nonLinear_, file_, scale);
 
        std::cout << "[1] Swin - One Window Test \n";

        std::cout << "[1-1] run [128] swin_ct_norm()\n";
        try{
            const int input_row = 12, input_col = 12, embed_dim = 128, head_size = 32;
            const int input_num = input_row * input_col;
            const int per_block_seq = 18;
            const int window_size = 12;
            const int block_num = input_num / per_block_seq;
            const int window_num = input_num / window_size / window_size;
            const int num_heads = embed_dim / head_size;
            std::cout << "=============== 配置说明 =================" << std::endl;
            std::cout << " >> 密文存储 : [ " << per_block_seq << " , " << embed_dim << " ]" << std::endl;
            std::cout << " >> 输入维度 : [ " << input_num << " , " << embed_dim << " ]" << std::endl;
            // 1. 准备输入数据
            std::vector<std::vector<double>> input_mat;
            std::vector<std::vector<double>> weight_query, weight_key, weight_value, position_bias_table;
            std::vector<double> bias_query, bias_key, bias_value;
            {
                input_mat = file_.readMat("ImageBackbone/input/input_mat_204_204_128");
                input_mat.resize(input_num);
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
            std::cout << "=============== swin_ct_norm =================" << std::endl;
            {
                std::vector<heongpu::Ciphertext<heongpu::Scheme::CKKS>> output_cts(block_num);
                std::atomic<long long> total_time(0LL), 
                    cyclic_shift_time(0LL), shift_reverse_time(0LL), window_partition_time(0LL), window_reverse_time(0LL),
                    q_time(0LL), k_time(0LL), score_tim(0LL), softmax_time(0LL), v_time(0LL), mulV_time(0LL);
                {
                    CppTimer timer(&total_time);
                    output_cts = model_.swin_ct_norm(input_cts, 
                        per_block_seq, embed_dim, num_heads, input_row, input_col, window_size, window_size / 2,
                        weight_query, bias_query, weight_key, bias_key, weight_value, bias_value, position_bias_table,
                        static_cast<double>(window_size), 3.0, 1.0,
                        &cyclic_shift_time, &shift_reverse_time, &window_partition_time, &window_reverse_time,
                        &q_time, &k_time, &score_tim, &softmax_time, &v_time, &mulV_time);
                }
                std::cout << "=============== swin_ct_norm 结果 =================" << std::endl;
                std::cout << "[1-1] swin_ct_norm() Cyclic Shift time: " << std::fixed << std::setprecision(3)
                    << cyclic_shift_time.load() / 1000.0 << " s\n";
                std::cout << "[1-1] swin_ct_norm() Window Partition time: " << std::fixed << std::setprecision(3)
                    << window_partition_time.load() / 1000.0 << " s\n";   
                std::cout << "[1-1] swin_ct_norm() Q time: " << std::fixed << std::setprecision(3)
                    << q_time.load() / 1000.0 << " s\n";
                std::cout << "[1-1] swin_ct_norm() K time: " << std::fixed << std::setprecision(3)
                    << k_time.load() / 1000.0 << " s\n";
                std::cout << "[1-1] swin_ct_norm() Score time: " << std::fixed << std::setprecision(3)
                    << score_tim.load() / 1000.0 << " s\n";
                std::cout << "[1-1] swin_ct_norm() Softmax time: " << std::fixed << std::setprecision(3)
                    << softmax_time.load() / 1000.0 << " s\n";
                std::cout << "[1-1] swin_ct_norm() V time: " << std::fixed << std::setprecision(3)
                    << v_time.load() / 1000.0 << " s\n";
                std::cout << "[1-1] swin_ct_norm() MulV time: " << std::fixed << std::setprecision(3)
                    << mulV_time.load() / 1000.0 << " s\n";
                std::cout << "[1-1] swin_ct_norm() Window Reverse time: " << std::fixed << std::setprecision(3)
                    << window_reverse_time.load() / 1000.0 << " s\n";
                std::cout << "[1-1] swin_ct_norm() Reverse Shift time: " << std::fixed << std::setprecision(3)
                    << shift_reverse_time.load() / 1000.0 << " s\n";
                std::cout << "[1-1] swin_ct_norm() Total time: " << std::fixed << std::setprecision(3)
                    << total_time.load() / 1000.0 << " s\n";
            }
            std::cout << "=============== swin_ct_opti =================" << std::endl;
            {
                std::vector<heongpu::Ciphertext<heongpu::Scheme::CKKS>> output_cts(block_num);
                std::atomic<long long> total_time(0LL), 
                    cyclic_shift_time(0LL), shift_reverse_time(0LL), window_partition_time(0LL), window_reverse_time(0LL),
                    q_time(0LL), k_time(0LL), score_tim(0LL), softmax_time(0LL), v_time(0LL), mulV_time(0LL);
                {
                    CppTimer timer(&total_time);
                    output_cts = model_.swin_ct_opti(input_cts, 
                        per_block_seq, embed_dim, num_heads, input_row, input_col, window_size, window_size / 2,
                        weight_query, bias_query, weight_key, bias_key, weight_value, bias_value, position_bias_table,
                        static_cast<double>(window_size), 3.0, 1.0,
                        &window_partition_time, &window_reverse_time, &q_time, &k_time, &score_tim, &softmax_time, &v_time, &mulV_time);
                }
                std::cout << "=============== swin_ct_opti 结果 =================" << std::endl;
                std::cout << "[1-1] swin_ct_opti() Cyclic Shift time: " << std::fixed << std::setprecision(3)
                    << cyclic_shift_time.load() / 1000.0 << " s\n";
                std::cout << "[1-1] swin_ct_opti() Window Partition time: " << std::fixed << std::setprecision(3)
                    << window_partition_time.load() / 1000.0 << " s\n";   
                std::cout << "[1-1] swin_ct_opti() Q time: " << std::fixed << std::setprecision(3)
                    << q_time.load() / 1000.0 << " s\n";
                std::cout << "[1-1] swin_ct_opti() K time: " << std::fixed << std::setprecision(3)
                    << k_time.load() / 1000.0 << " s\n";
                std::cout << "[1-1] swin_ct_opti() Score time: " << std::fixed << std::setprecision(3)
                    << score_tim.load() / 1000.0 << " s\n";
                std::cout << "[1-1] swin_ct_opti() Softmax time: " << std::fixed << std::setprecision(3)
                    << softmax_time.load() / 1000.0 << " s\n";
                std::cout << "[1-1] swin_ct_opti() V time: " << std::fixed << std::setprecision(3)
                    << v_time.load() / 1000.0 << " s\n";
                std::cout << "[1-1] swin_ct_opti() MulV time: " << std::fixed << std::setprecision(3)
                    << mulV_time.load() / 1000.0 << " s\n";
                std::cout << "[1-1] swin_ct_opti() Window Reverse time: " << std::fixed << std::setprecision(3)
                    << window_reverse_time.load() / 1000.0 << " s\n";
                std::cout << "[1-1] swin_ct_opti() Reverse Shift time: " << std::fixed << std::setprecision(3)
                    << shift_reverse_time.load() / 1000.0 << " s\n";
                std::cout << "[1-1] swin_ct_opti() Total time: " << std::fixed << std::setprecision(3)
                    << total_time.load() / 1000.0 << " s\n";
            }
        }catch (const std::exception &e) {
            std::cerr << "[ERROR] exception : " << e.what() << std::endl;
        }
        std::cout << "[1-2] run [256] swin_ct_norm()\n";
        try{
            const int input_row = 12, input_col = 12, embed_dim = 256, head_size = 32;
            const int input_num = input_row * input_col;
            const int per_block_seq = 18;
            const int window_size = 12;
            const int block_num = input_num / per_block_seq;
            const int window_num = input_num / window_size / window_size;
            const int num_heads = embed_dim / head_size;
            std::cout << "=============== 配置说明 =================" << std::endl;
            std::cout << " >> 密文存储 : [ " << per_block_seq << " , " << embed_dim << " ]" << std::endl;
            std::cout << " >> 输入维度 : [ " << input_num << " , " << embed_dim << " ]" << std::endl;
            // 1. 准备输入数据
            std::vector<std::vector<double>> input_mat;
            std::vector<std::vector<double>> weight_query, weight_key, weight_value, position_bias_table;
            std::vector<double> bias_query, bias_key, bias_value;
            {
                input_mat = file_.readMat("ImageBackbone/input/input_mat_108_108_256");
                input_mat.resize(input_num);
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
            std::cout << "=============== swin_ct_norm =================" << std::endl;
            {
                std::vector<heongpu::Ciphertext<heongpu::Scheme::CKKS>> output_cts(block_num);
                std::atomic<long long> total_time(0LL), 
                    cyclic_shift_time(0LL), shift_reverse_time(0LL), window_partition_time(0LL), window_reverse_time(0LL),
                    q_time(0LL), k_time(0LL), score_tim(0LL), softmax_time(0LL), v_time(0LL), mulV_time(0LL);
                {
                    CppTimer timer(&total_time);
                    output_cts = model_.swin_ct_norm(input_cts, 
                        per_block_seq, embed_dim, num_heads, input_row, input_col, window_size, window_size / 2,
                        weight_query, bias_query, weight_key, bias_key, weight_value, bias_value, position_bias_table,
                        static_cast<double>(window_size), 3.0, 1.0,
                        &cyclic_shift_time, &shift_reverse_time, &window_partition_time, &window_reverse_time,
                        &q_time, &k_time, &score_tim, &softmax_time, &v_time, &mulV_time);
                }
                std::cout << "=============== swin_ct_norm 结果 =================" << std::endl;
                std::cout << "[1-2] swin_ct_norm() Cyclic Shift time: " << std::fixed << std::setprecision(3)
                    << cyclic_shift_time.load() / 1000.0 << " s\n";
                std::cout << "[1-2] swin_ct_norm() Window Partition time: " << std::fixed << std::setprecision(3)
                    << window_partition_time.load() / 1000.0 << " s\n";   
                std::cout << "[1-2] swin_ct_norm() Q time: " << std::fixed << std::setprecision(3)
                    << q_time.load() / 1000.0 << " s\n";
                std::cout << "[1-2] swin_ct_norm() K time: " << std::fixed << std::setprecision(3)
                    << k_time.load() / 1000.0 << " s\n";
                std::cout << "[1-2] swin_ct_norm() Score time: " << std::fixed << std::setprecision(3)
                    << score_tim.load() / 1000.0 << " s\n";
                std::cout << "[1-2] swin_ct_norm() Softmax time: " << std::fixed << std::setprecision(3)
                    << softmax_time.load() / 1000.0 << " s\n";
                std::cout << "[1-2] swin_ct_norm() V time: " << std::fixed << std::setprecision(3)
                    << v_time.load() / 1000.0 << " s\n";
                std::cout << "[1-2] swin_ct_norm() MulV time: " << std::fixed << std::setprecision(3)
                    << mulV_time.load() / 1000.0 << " s\n";
                std::cout << "[1-2] swin_ct_norm() Window Reverse time: " << std::fixed << std::setprecision(3)
                    << window_reverse_time.load() / 1000.0 << " s\n";
                std::cout << "[1-2] swin_ct_norm() Reverse Shift time: " << std::fixed << std::setprecision(3)
                    << shift_reverse_time.load() / 1000.0 << " s\n";
                std::cout << "[1-2] swin_ct_norm() Total time: " << std::fixed << std::setprecision(3)
                    << total_time.load() / 1000.0 << " s\n";
            }
            std::cout << "=============== swin_ct_opti =================" << std::endl;
            {
                std::vector<heongpu::Ciphertext<heongpu::Scheme::CKKS>> output_cts(block_num);
                std::atomic<long long> total_time(0LL), 
                    cyclic_shift_time(0LL), shift_reverse_time(0LL), window_partition_time(0LL), window_reverse_time(0LL),
                    q_time(0LL), k_time(0LL), score_tim(0LL), softmax_time(0LL), v_time(0LL), mulV_time(0LL);
                {
                    CppTimer timer(&total_time);
                    output_cts = model_.swin_ct_opti(input_cts, 
                        per_block_seq, embed_dim, num_heads, input_row, input_col, window_size, window_size / 2,
                        weight_query, bias_query, weight_key, bias_key, weight_value, bias_value, position_bias_table,
                        static_cast<double>(window_size), 3.0, 1.0,
                        &window_partition_time, &window_reverse_time, &q_time, &k_time, &score_tim, &softmax_time, &v_time, &mulV_time);
                }
                std::cout << "=============== swin_ct_opti 结果 =================" << std::endl;
                std::cout << "[1-2] swin_ct_opti() Cyclic Shift time: " << std::fixed << std::setprecision(3)
                    << cyclic_shift_time.load() / 1000.0 << " s\n";
                std::cout << "[1-2] swin_ct_opti() Window Partition time: " << std::fixed << std::setprecision(3)
                    << window_partition_time.load() / 1000.0 << " s\n";   
                std::cout << "[1-2] swin_ct_opti() Q time: " << std::fixed << std::setprecision(3)
                    << q_time.load() / 1000.0 << " s\n";
                std::cout << "[1-2] swin_ct_opti() K time: " << std::fixed << std::setprecision(3)
                    << k_time.load() / 1000.0 << " s\n";
                std::cout << "[1-2] swin_ct_opti() Score time: " << std::fixed << std::setprecision(3)
                    << score_tim.load() / 1000.0 << " s\n";
                std::cout << "[1-2] swin_ct_opti() Softmax time: " << std::fixed << std::setprecision(3)
                    << softmax_time.load() / 1000.0 << " s\n";
                std::cout << "[1-2] swin_ct_opti() V time: " << std::fixed << std::setprecision(3)
                    << v_time.load() / 1000.0 << " s\n";
                std::cout << "[1-2] swin_ct_opti() MulV time: " << std::fixed << std::setprecision(3)
                    << mulV_time.load() / 1000.0 << " s\n";
                std::cout << "[1-2] swin_ct_opti() Window Reverse time: " << std::fixed << std::setprecision(3)
                    << window_reverse_time.load() / 1000.0 << " s\n";
                std::cout << "[1-2] swin_ct_opti() Reverse Shift time: " << std::fixed << std::setprecision(3)
                    << shift_reverse_time.load() / 1000.0 << " s\n";
                std::cout << "[1-2] swin_ct_opti() Total time: " << std::fixed << std::setprecision(3)
                    << total_time.load() / 1000.0 << " s\n";
            }
        }catch (const std::exception &e) {
            std::cerr << "[ERROR] exception : " << e.what() << std::endl;
        }
        std::cout << "[1-3] run [512] swin_ct_norm()\n";
        try{
            const int input_row = 12, input_col = 12, embed_dim = 512, head_size = 32;
            const int input_num = input_row * input_col;
            const int per_block_seq = 12;
            const int window_size = 12;
            const int block_num = input_num / per_block_seq;
            const int window_num = input_num / window_size / window_size;
            const int num_heads = embed_dim / head_size;
            std::cout << "=============== 配置说明 =================" << std::endl;
            std::cout << " >> 密文存储 : [ " << per_block_seq << " , " << embed_dim << " ]" << std::endl;
            std::cout << " >> 输入维度 : [ " << input_num << " , " << embed_dim << " ]" << std::endl;
            // 1. 准备输入数据
            std::vector<std::vector<double>> input_mat;
            std::vector<std::vector<double>> weight_query, weight_key, weight_value, position_bias_table;
            std::vector<double> bias_query, bias_key, bias_value;
            {
                input_mat = file_.readMat("ImageBackbone/input/input_mat_60_60_512");
                input_mat.resize(input_num);
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
            std::cout << "=============== swin_ct_norm =================" << std::endl;
            {
                std::vector<heongpu::Ciphertext<heongpu::Scheme::CKKS>> output_cts(block_num);
                std::atomic<long long> total_time(0LL), 
                    cyclic_shift_time(0LL), shift_reverse_time(0LL), window_partition_time(0LL), window_reverse_time(0LL),
                    q_time(0LL), k_time(0LL), score_tim(0LL), softmax_time(0LL), v_time(0LL), mulV_time(0LL);
                {
                    CppTimer timer(&total_time);
                    output_cts = model_.swin_ct_norm(input_cts, 
                        per_block_seq, embed_dim, num_heads, input_row, input_col, window_size, window_size / 2,
                        weight_query, bias_query, weight_key, bias_key, weight_value, bias_value, position_bias_table,
                        static_cast<double>(window_size), 3.0, 1.0,
                        &cyclic_shift_time, &shift_reverse_time, &window_partition_time, &window_reverse_time,
                        &q_time, &k_time, &score_tim, &softmax_time, &v_time, &mulV_time);
                }
                std::cout << "=============== swin_ct_norm 结果 =================" << std::endl;
                std::cout << "[1-3] swin_ct_norm() Cyclic Shift time: " << std::fixed << std::setprecision(3)
                    << cyclic_shift_time.load() / 1000.0 << " s\n";
                std::cout << "[1-3] swin_ct_norm() Window Partition time: " << std::fixed << std::setprecision(3)
                    << window_partition_time.load() / 1000.0 << " s\n";   
                std::cout << "[1-3] swin_ct_norm() Q time: " << std::fixed << std::setprecision(3)
                    << q_time.load() / 1000.0 << " s\n";
                std::cout << "[1-3] swin_ct_norm() K time: " << std::fixed << std::setprecision(3)
                    << k_time.load() / 1000.0 << " s\n";
                std::cout << "[1-3] swin_ct_norm() Score time: " << std::fixed << std::setprecision(3)
                    << score_tim.load() / 1000.0 << " s\n";
                std::cout << "[1-3] swin_ct_norm() Softmax time: " << std::fixed << std::setprecision(3)
                    << softmax_time.load() / 1000.0 << " s\n";
                std::cout << "[1-3] swin_ct_norm() V time: " << std::fixed << std::setprecision(3)
                    << v_time.load() / 1000.0 << " s\n";
                std::cout << "[1-3] swin_ct_norm() MulV time: " << std::fixed << std::setprecision(3)
                    << mulV_time.load() / 1000.0 << " s\n";
                std::cout << "[1-3] swin_ct_norm() Window Reverse time: " << std::fixed << std::setprecision(3)
                    << window_reverse_time.load() / 1000.0 << " s\n";
                std::cout << "[1-3] swin_ct_norm() Reverse Shift time: " << std::fixed << std::setprecision(3)
                    << shift_reverse_time.load() / 1000.0 << " s\n";
                std::cout << "[1-3] swin_ct_norm() Total time: " << std::fixed << std::setprecision(3)
                    << total_time.load() / 1000.0 << " s\n";
            }
            std::cout << "=============== swin_ct_opti =================" << std::endl;
            {
                std::vector<heongpu::Ciphertext<heongpu::Scheme::CKKS>> output_cts(block_num);
                std::atomic<long long> total_time(0LL), 
                    cyclic_shift_time(0LL), shift_reverse_time(0LL), window_partition_time(0LL), window_reverse_time(0LL),
                    q_time(0LL), k_time(0LL), score_tim(0LL), softmax_time(0LL), v_time(0LL), mulV_time(0LL);
                {
                    CppTimer timer(&total_time);
                    output_cts = model_.swin_ct_opti(input_cts, 
                        per_block_seq, embed_dim, num_heads, input_row, input_col, window_size, window_size / 2,
                        weight_query, bias_query, weight_key, bias_key, weight_value, bias_value, position_bias_table,
                        static_cast<double>(window_size), 3.0, 1.0,
                        &window_partition_time, &window_reverse_time, &q_time, &k_time, &score_tim, &softmax_time, &v_time, &mulV_time);
                }
                std::cout << "=============== swin_ct_opti 结果 =================" << std::endl;
                std::cout << "[1-3] swin_ct_opti() Cyclic Shift time: " << std::fixed << std::setprecision(3)
                    << cyclic_shift_time.load() / 1000.0 << " s\n";
                std::cout << "[1-3] swin_ct_opti() Window Partition time: " << std::fixed << std::setprecision(3)
                    << window_partition_time.load() / 1000.0 << " s\n";   
                std::cout << "[1-3] swin_ct_opti() Q time: " << std::fixed << std::setprecision(3)
                    << q_time.load() / 1000.0 << " s\n";
                std::cout << "[1-3] swin_ct_opti() K time: " << std::fixed << std::setprecision(3)
                    << k_time.load() / 1000.0 << " s\n";
                std::cout << "[1-3] swin_ct_opti() Score time: " << std::fixed << std::setprecision(3)
                    << score_tim.load() / 1000.0 << " s\n";
                std::cout << "[1-3] swin_ct_opti() Softmax time: " << std::fixed << std::setprecision(3)
                    << softmax_time.load() / 1000.0 << " s\n";
                std::cout << "[1-3] swin_ct_opti() V time: " << std::fixed << std::setprecision(3)
                    << v_time.load() / 1000.0 << " s\n";
                std::cout << "[1-3] swin_ct_opti() MulV time: " << std::fixed << std::setprecision(3)
                    << mulV_time.load() / 1000.0 << " s\n";
                std::cout << "[1-3] swin_ct_opti() Window Reverse time: " << std::fixed << std::setprecision(3)
                    << window_reverse_time.load() / 1000.0 << " s\n";
                std::cout << "[1-3] swin_ct_opti() Reverse Shift time: " << std::fixed << std::setprecision(3)
                    << shift_reverse_time.load() / 1000.0 << " s\n";
                std::cout << "[1-3] swin_ct_opti() Total time: " << std::fixed << std::setprecision(3)
                    << total_time.load() / 1000.0 << " s\n";
            }
        }catch (const std::exception &e) {
            std::cerr << "[ERROR] exception : " << e.what() << std::endl;
        }
        std::cout << "[1-4] run [1024] swin_ct_norm()\n";
        try{
            const int input_row = 12, input_col = 12, embed_dim = 1024, head_size = 32;
            const int input_num = input_row * input_col;
            const int per_block_seq = 6;
            const int window_size = 12;
            const int block_num = input_num / per_block_seq;
            const int window_num = input_num / window_size / window_size;
            const int num_heads = embed_dim / head_size;
            std::cout << "=============== 配置说明 =================" << std::endl;
            std::cout << " >> 密文存储 : [ " << per_block_seq << " , " << embed_dim << " ]" << std::endl;
            std::cout << " >> 输入维度 : [ " << input_num << " , " << embed_dim << " ]" << std::endl;
            // 1. 准备输入数据
            std::vector<std::vector<double>> input_mat;
            std::vector<std::vector<double>> weight_query, weight_key, weight_value, position_bias_table;
            std::vector<double> bias_query, bias_key, bias_value;
            {
                input_mat = file_.readMat("ImageBackbone/input/input_mat_36_36_1024");
                input_mat.resize(input_num);
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
            std::cout << "=============== swin_ct_norm =================" << std::endl;
            {
                std::vector<heongpu::Ciphertext<heongpu::Scheme::CKKS>> output_cts(block_num);
                std::atomic<long long> total_time(0LL), 
                    cyclic_shift_time(0LL), shift_reverse_time(0LL), window_partition_time(0LL), window_reverse_time(0LL),
                    q_time(0LL), k_time(0LL), score_tim(0LL), softmax_time(0LL), v_time(0LL), mulV_time(0LL);
                {
                    CppTimer timer(&total_time);
                    output_cts = model_.swin_ct_norm(input_cts, 
                        per_block_seq, embed_dim, num_heads, input_row, input_col, window_size, window_size / 2,
                        weight_query, bias_query, weight_key, bias_key, weight_value, bias_value, position_bias_table,
                        static_cast<double>(window_size), 3.0, 1.0,
                        &cyclic_shift_time, &shift_reverse_time, &window_partition_time, &window_reverse_time,
                        &q_time, &k_time, &score_tim, &softmax_time, &v_time, &mulV_time);
                }
                std::cout << "=============== swin_ct_norm 结果 =================" << std::endl;
                std::cout << "[1-4] swin_ct_norm() Cyclic Shift time: " << std::fixed << std::setprecision(3)
                    << cyclic_shift_time.load() / 1000.0 << " s\n";
                std::cout << "[1-4] swin_ct_norm() Window Partition time: " << std::fixed << std::setprecision(3)
                    << window_partition_time.load() / 1000.0 << " s\n";   
                std::cout << "[1-4] swin_ct_norm() Q time: " << std::fixed << std::setprecision(3)
                    << q_time.load() / 1000.0 << " s\n";
                std::cout << "[1-4] swin_ct_norm() K time: " << std::fixed << std::setprecision(3)
                    << k_time.load() / 1000.0 << " s\n";
                std::cout << "[1-4] swin_ct_norm() Score time: " << std::fixed << std::setprecision(3)
                    << score_tim.load() / 1000.0 << " s\n";
                std::cout << "[1-4] swin_ct_norm() Softmax time: " << std::fixed << std::setprecision(3)
                    << softmax_time.load() / 1000.0 << " s\n";
                std::cout << "[1-4] swin_ct_norm() V time: " << std::fixed << std::setprecision(3)
                    << v_time.load() / 1000.0 << " s\n";
                std::cout << "[1-4] swin_ct_norm() MulV time: " << std::fixed << std::setprecision(3)
                    << mulV_time.load() / 1000.0 << " s\n";
                std::cout << "[1-4] swin_ct_norm() Window Reverse time: " << std::fixed << std::setprecision(3)
                    << window_reverse_time.load() / 1000.0 << " s\n";
                std::cout << "[1-4] swin_ct_norm() Reverse Shift time: " << std::fixed << std::setprecision(3)
                    << shift_reverse_time.load() / 1000.0 << " s\n";
                std::cout << "[1-4] swin_ct_norm() Total time: " << std::fixed << std::setprecision(3)
                    << total_time.load() / 1000.0 << " s\n";
            }
            std::cout << "=============== swin_ct_opti =================" << std::endl;
            {
                std::vector<heongpu::Ciphertext<heongpu::Scheme::CKKS>> output_cts(block_num);
                std::atomic<long long> total_time(0LL), 
                    cyclic_shift_time(0LL), shift_reverse_time(0LL), window_partition_time(0LL), window_reverse_time(0LL),
                    q_time(0LL), k_time(0LL), score_tim(0LL), softmax_time(0LL), v_time(0LL), mulV_time(0LL);
                {
                    CppTimer timer(&total_time);
                    output_cts = model_.swin_ct_opti(input_cts, 
                        per_block_seq, embed_dim, num_heads, input_row, input_col, window_size, window_size / 2,
                        weight_query, bias_query, weight_key, bias_key, weight_value, bias_value, position_bias_table,
                        static_cast<double>(window_size), 3.0, 1.0,
                        &window_partition_time, &window_reverse_time, &q_time, &k_time, &score_tim, &softmax_time, &v_time, &mulV_time);
                }
                std::cout << "=============== swin_ct_opti 结果 =================" << std::endl;
                std::cout << "[1-4] swin_ct_opti() Cyclic Shift time: " << std::fixed << std::setprecision(3)
                    << cyclic_shift_time.load() / 1000.0 << " s\n";
                std::cout << "[1-4] swin_ct_opti() Window Partition time: " << std::fixed << std::setprecision(3)
                    << window_partition_time.load() / 1000.0 << " s\n";   
                std::cout << "[1-4] swin_ct_opti() Q time: " << std::fixed << std::setprecision(3)
                    << q_time.load() / 1000.0 << " s\n";
                std::cout << "[1-4] swin_ct_opti() K time: " << std::fixed << std::setprecision(3)
                    << k_time.load() / 1000.0 << " s\n";
                std::cout << "[1-4] swin_ct_opti() Score time: " << std::fixed << std::setprecision(3)
                    << score_tim.load() / 1000.0 << " s\n";
                std::cout << "[1-4] swin_ct_opti() Softmax time: " << std::fixed << std::setprecision(3)
                    << softmax_time.load() / 1000.0 << " s\n";
                std::cout << "[1-4] swin_ct_opti() V time: " << std::fixed << std::setprecision(3)
                    << v_time.load() / 1000.0 << " s\n";
                std::cout << "[1-4] swin_ct_opti() MulV time: " << std::fixed << std::setprecision(3)
                    << mulV_time.load() / 1000.0 << " s\n";
                std::cout << "[1-4] swin_ct_opti() Window Reverse time: " << std::fixed << std::setprecision(3)
                    << window_reverse_time.load() / 1000.0 << " s\n";
                std::cout << "[1-4] swin_ct_opti() Reverse Shift time: " << std::fixed << std::setprecision(3)
                    << shift_reverse_time.load() / 1000.0 << " s\n";
                std::cout << "[1-4] swin_ct_opti() Total time: " << std::fixed << std::setprecision(3)
                    << total_time.load() / 1000.0 << " s\n";
            }
        }catch (const std::exception &e) {
            std::cerr << "[ERROR] exception : " << e.what() << std::endl;
        }

        std::cout << "[2] Swin - Four Windows Test \n";
        
        std::cout << "[2-1] run [128] swin_ct_norm()\n";
        try{
            const int input_row = 24, input_col = 24, embed_dim = 128, head_size = 32;
            const int input_num = input_row * input_col;
            const int per_block_seq = 18;
            const int window_size = 12;
            const int block_num = input_num / per_block_seq;
            const int window_num = input_num / window_size / window_size;
            const int num_heads = embed_dim / head_size;
            std::cout << "=============== 配置说明 =================" << std::endl;
            std::cout << " >> 密文存储 : [ " << per_block_seq << " , " << embed_dim << " ]" << std::endl;
            std::cout << " >> 输入维度 : [ " << input_num << " , " << embed_dim << " ]" << std::endl;
            // 1. 准备输入数据
            std::vector<std::vector<double>> input_mat;
            std::vector<std::vector<double>> weight_query, weight_key, weight_value, position_bias_table;
            std::vector<double> bias_query, bias_key, bias_value;
            {
                input_mat = file_.readMat("ImageBackbone/input/input_mat_204_204_128");
                input_mat.resize(input_num);
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
            std::cout << "=============== swin_ct_norm =================" << std::endl;
            {
                std::vector<heongpu::Ciphertext<heongpu::Scheme::CKKS>> output_cts(block_num);
                std::atomic<long long> total_time(0LL), 
                    cyclic_shift_time(0LL), shift_reverse_time(0LL), window_partition_time(0LL), window_reverse_time(0LL),
                    q_time(0LL), k_time(0LL), score_tim(0LL), softmax_time(0LL), v_time(0LL), mulV_time(0LL);
                {
                    CppTimer timer(&total_time);
                    output_cts = model_.swin_ct_norm(input_cts, 
                        per_block_seq, embed_dim, num_heads, input_row, input_col, window_size, window_size / 2,
                        weight_query, bias_query, weight_key, bias_key, weight_value, bias_value, position_bias_table,
                        static_cast<double>(window_size), 3.0, 1.0,
                        &cyclic_shift_time, &shift_reverse_time, &window_partition_time, &window_reverse_time,
                        &q_time, &k_time, &score_tim, &softmax_time, &v_time, &mulV_time);
                }
                std::cout << "=============== swin_ct_norm 结果 =================" << std::endl;
                std::cout << "[2-1] swin_ct_norm() Cyclic Shift time: " << std::fixed << std::setprecision(3)
                    << cyclic_shift_time.load() / 1000.0 << " s\n";
                std::cout << "[2-1] swin_ct_norm() Window Partition time: " << std::fixed << std::setprecision(3)
                    << window_partition_time.load() / 1000.0 << " s\n";   
                std::cout << "[2-1] swin_ct_norm() Q time: " << std::fixed << std::setprecision(3)
                    << q_time.load() / 1000.0 << " s\n";
                std::cout << "[2-1] swin_ct_norm() K time: " << std::fixed << std::setprecision(3)
                    << k_time.load() / 1000.0 << " s\n";
                std::cout << "[2-1] swin_ct_norm() Score time: " << std::fixed << std::setprecision(3)
                    << score_tim.load() / 1000.0 << " s\n";
                std::cout << "[2-1] swin_ct_norm() Softmax time: " << std::fixed << std::setprecision(3)
                    << softmax_time.load() / 1000.0 << " s\n";
                std::cout << "[2-1] swin_ct_norm() V time: " << std::fixed << std::setprecision(3)
                    << v_time.load() / 1000.0 << " s\n";
                std::cout << "[2-1] swin_ct_norm() MulV time: " << std::fixed << std::setprecision(3)
                    << mulV_time.load() / 1000.0 << " s\n";
                std::cout << "[2-1] swin_ct_norm() Window Reverse time: " << std::fixed << std::setprecision(3)
                    << window_reverse_time.load() / 1000.0 << " s\n";
                std::cout << "[2-1] swin_ct_norm() Reverse Shift time: " << std::fixed << std::setprecision(3)
                    << shift_reverse_time.load() / 1000.0 << " s\n";
                std::cout << "[2-1] swin_ct_norm() Total time: " << std::fixed << std::setprecision(3)
                    << total_time.load() / 1000.0 << " s\n";
            }
            std::cout << "=============== swin_ct_opti =================" << std::endl;
            {
                std::vector<heongpu::Ciphertext<heongpu::Scheme::CKKS>> output_cts(block_num);
                std::atomic<long long> total_time(0LL), 
                    cyclic_shift_time(0LL), shift_reverse_time(0LL), window_partition_time(0LL), window_reverse_time(0LL),
                    q_time(0LL), k_time(0LL), score_tim(0LL), softmax_time(0LL), v_time(0LL), mulV_time(0LL);
                {
                    CppTimer timer(&total_time);
                    output_cts = model_.swin_ct_opti(input_cts, 
                        per_block_seq, embed_dim, num_heads, input_row, input_col, window_size, window_size / 2,
                        weight_query, bias_query, weight_key, bias_key, weight_value, bias_value, position_bias_table,
                        static_cast<double>(window_size), 3.0, 1.0,
                        &window_partition_time, &window_reverse_time, &q_time, &k_time, &score_tim, &softmax_time, &v_time, &mulV_time);
                }
                std::cout << "=============== swin_ct_opti 结果 =================" << std::endl;
                std::cout << "[2-1] swin_ct_opti() Cyclic Shift time: " << std::fixed << std::setprecision(3)
                    << cyclic_shift_time.load() / 1000.0 << " s\n";
                std::cout << "[2-1] swin_ct_opti() Window Partition time: " << std::fixed << std::setprecision(3)
                    << window_partition_time.load() / 1000.0 << " s\n";   
                std::cout << "[2-1] swin_ct_opti() Q time: " << std::fixed << std::setprecision(3)
                    << q_time.load() / 1000.0 << " s\n";
                std::cout << "[2-1] swin_ct_opti() K time: " << std::fixed << std::setprecision(3)
                    << k_time.load() / 1000.0 << " s\n";
                std::cout << "[2-1] swin_ct_opti() Score time: " << std::fixed << std::setprecision(3)
                    << score_tim.load() / 1000.0 << " s\n";
                std::cout << "[2-1] swin_ct_opti() Softmax time: " << std::fixed << std::setprecision(3)
                    << softmax_time.load() / 1000.0 << " s\n";
                std::cout << "[2-1] swin_ct_opti() V time: " << std::fixed << std::setprecision(3)
                    << v_time.load() / 1000.0 << " s\n";
                std::cout << "[2-1] swin_ct_opti() MulV time: " << std::fixed << std::setprecision(3)
                    << mulV_time.load() / 1000.0 << " s\n";
                std::cout << "[2-1] swin_ct_opti() Window Reverse time: " << std::fixed << std::setprecision(3)
                    << window_reverse_time.load() / 1000.0 << " s\n";
                std::cout << "[2-1] swin_ct_opti() Reverse Shift time: " << std::fixed << std::setprecision(3)
                    << shift_reverse_time.load() / 1000.0 << " s\n";
                std::cout << "[2-1] swin_ct_opti() Total time: " << std::fixed << std::setprecision(3)
                    << total_time.load() / 1000.0 << " s\n";
            }
        }catch (const std::exception &e) {
            std::cerr << "[ERROR] exception : " << e.what() << std::endl;
        }
        std::cout << "[2-2] run [256] swin_ct_norm()\n";
        try{
            const int input_row = 24, input_col = 24, embed_dim = 256, head_size = 32;
            const int input_num = input_row * input_col;
            const int per_block_seq = 18;
            const int window_size = 12;
            const int block_num = input_num / per_block_seq;
            const int window_num = input_num / window_size / window_size;
            const int num_heads = embed_dim / head_size;
            std::cout << "=============== 配置说明 =================" << std::endl;
            std::cout << " >> 密文存储 : [ " << per_block_seq << " , " << embed_dim << " ]" << std::endl;
            std::cout << " >> 输入维度 : [ " << input_num << " , " << embed_dim << " ]" << std::endl;
            // 1. 准备输入数据
            std::vector<std::vector<double>> input_mat;
            std::vector<std::vector<double>> weight_query, weight_key, weight_value, position_bias_table;
            std::vector<double> bias_query, bias_key, bias_value;
            {
                input_mat = file_.readMat("ImageBackbone/input/input_mat_108_108_256");
                input_mat.resize(input_num);
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
            std::cout << "=============== swin_ct_norm =================" << std::endl;
            {
                std::vector<heongpu::Ciphertext<heongpu::Scheme::CKKS>> output_cts(block_num);
                std::atomic<long long> total_time(0LL), 
                    cyclic_shift_time(0LL), shift_reverse_time(0LL), window_partition_time(0LL), window_reverse_time(0LL),
                    q_time(0LL), k_time(0LL), score_tim(0LL), softmax_time(0LL), v_time(0LL), mulV_time(0LL);
                {
                    CppTimer timer(&total_time);
                    output_cts = model_.swin_ct_norm(input_cts, 
                        per_block_seq, embed_dim, num_heads, input_row, input_col, window_size, window_size / 2,
                        weight_query, bias_query, weight_key, bias_key, weight_value, bias_value, position_bias_table,
                        static_cast<double>(window_size), 3.0, 1.0,
                        &cyclic_shift_time, &shift_reverse_time, &window_partition_time, &window_reverse_time,
                        &q_time, &k_time, &score_tim, &softmax_time, &v_time, &mulV_time);
                }
                std::cout << "=============== swin_ct_norm 结果 =================" << std::endl;
                std::cout << "[2-2] swin_ct_norm() Cyclic Shift time: " << std::fixed << std::setprecision(3)
                    << cyclic_shift_time.load() / 1000.0 << " s\n";
                std::cout << "[2-2] swin_ct_norm() Window Partition time: " << std::fixed << std::setprecision(3)
                    << window_partition_time.load() / 1000.0 << " s\n";   
                std::cout << "[2-2] swin_ct_norm() Q time: " << std::fixed << std::setprecision(3)
                    << q_time.load() / 1000.0 << " s\n";
                std::cout << "[2-2] swin_ct_norm() K time: " << std::fixed << std::setprecision(3)
                    << k_time.load() / 1000.0 << " s\n";
                std::cout << "[2-2] swin_ct_norm() Score time: " << std::fixed << std::setprecision(3)
                    << score_tim.load() / 1000.0 << " s\n";
                std::cout << "[2-2] swin_ct_norm() Softmax time: " << std::fixed << std::setprecision(3)
                    << softmax_time.load() / 1000.0 << " s\n";
                std::cout << "[2-2] swin_ct_norm() V time: " << std::fixed << std::setprecision(3)
                    << v_time.load() / 1000.0 << " s\n";
                std::cout << "[2-2] swin_ct_norm() MulV time: " << std::fixed << std::setprecision(3)
                    << mulV_time.load() / 1000.0 << " s\n";
                std::cout << "[2-2] swin_ct_norm() Window Reverse time: " << std::fixed << std::setprecision(3)
                    << window_reverse_time.load() / 1000.0 << " s\n";
                std::cout << "[2-2] swin_ct_norm() Reverse Shift time: " << std::fixed << std::setprecision(3)
                    << shift_reverse_time.load() / 1000.0 << " s\n";
                std::cout << "[2-2] swin_ct_norm() Total time: " << std::fixed << std::setprecision(3)
                    << total_time.load() / 1000.0 << " s\n";
            }
            std::cout << "=============== swin_ct_opti =================" << std::endl;
            {
                std::vector<heongpu::Ciphertext<heongpu::Scheme::CKKS>> output_cts(block_num);
                std::atomic<long long> total_time(0LL), 
                    cyclic_shift_time(0LL), shift_reverse_time(0LL), window_partition_time(0LL), window_reverse_time(0LL),
                    q_time(0LL), k_time(0LL), score_tim(0LL), softmax_time(0LL), v_time(0LL), mulV_time(0LL);
                {
                    CppTimer timer(&total_time);
                    output_cts = model_.swin_ct_opti(input_cts, 
                        per_block_seq, embed_dim, num_heads, input_row, input_col, window_size, window_size / 2,
                        weight_query, bias_query, weight_key, bias_key, weight_value, bias_value, position_bias_table,
                        static_cast<double>(window_size), 3.0, 1.0,
                        &window_partition_time, &window_reverse_time, &q_time, &k_time, &score_tim, &softmax_time, &v_time, &mulV_time);
                }
                std::cout << "=============== swin_ct_opti 结果 =================" << std::endl;
                std::cout << "[2-2] swin_ct_opti() Cyclic Shift time: " << std::fixed << std::setprecision(3)
                    << cyclic_shift_time.load() / 1000.0 << " s\n";
                std::cout << "[2-2] swin_ct_opti() Window Partition time: " << std::fixed << std::setprecision(3)
                    << window_partition_time.load() / 1000.0 << " s\n";   
                std::cout << "[2-2] swin_ct_opti() Q time: " << std::fixed << std::setprecision(3)
                    << q_time.load() / 1000.0 << " s\n";
                std::cout << "[2-2] swin_ct_opti() K time: " << std::fixed << std::setprecision(3)
                    << k_time.load() / 1000.0 << " s\n";
                std::cout << "[2-2] swin_ct_opti() Score time: " << std::fixed << std::setprecision(3)
                    << score_tim.load() / 1000.0 << " s\n";
                std::cout << "[2-2] swin_ct_opti() Softmax time: " << std::fixed << std::setprecision(3)
                    << softmax_time.load() / 1000.0 << " s\n";
                std::cout << "[2-2] swin_ct_opti() V time: " << std::fixed << std::setprecision(3)
                    << v_time.load() / 1000.0 << " s\n";
                std::cout << "[2-2] swin_ct_opti() MulV time: " << std::fixed << std::setprecision(3)
                    << mulV_time.load() / 1000.0 << " s\n";
                std::cout << "[2-2] swin_ct_opti() Window Reverse time: " << std::fixed << std::setprecision(3)
                    << window_reverse_time.load() / 1000.0 << " s\n";
                std::cout << "[2-2] swin_ct_opti() Reverse Shift time: " << std::fixed << std::setprecision(3)
                    << shift_reverse_time.load() / 1000.0 << " s\n";
                std::cout << "[2-2] swin_ct_opti() Total time: " << std::fixed << std::setprecision(3)
                    << total_time.load() / 1000.0 << " s\n";
            }
        }catch (const std::exception &e) {
            std::cerr << "[ERROR] exception : " << e.what() << std::endl;
        }
        std::cout << "[2-3] run [512] swin_ct_norm()\n";
        try{
            const int input_row = 24, input_col = 24, embed_dim = 512, head_size = 32;
            const int input_num = input_row * input_col;
            const int per_block_seq = 12;
            const int window_size = 12;
            const int block_num = input_num / per_block_seq;
            const int window_num = input_num / window_size / window_size;
            const int num_heads = embed_dim / head_size;
            std::cout << "=============== 配置说明 =================" << std::endl;
            std::cout << " >> 密文存储 : [ " << per_block_seq << " , " << embed_dim << " ]" << std::endl;
            std::cout << " >> 输入维度 : [ " << input_num << " , " << embed_dim << " ]" << std::endl;
            // 1. 准备输入数据
            std::vector<std::vector<double>> input_mat;
            std::vector<std::vector<double>> weight_query, weight_key, weight_value, position_bias_table;
            std::vector<double> bias_query, bias_key, bias_value;
            {
                input_mat = file_.readMat("ImageBackbone/input/input_mat_60_60_512");
                input_mat.resize(input_num);
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
            std::cout << "=============== swin_ct_norm =================" << std::endl;
            {
                std::vector<heongpu::Ciphertext<heongpu::Scheme::CKKS>> output_cts(block_num);
                std::atomic<long long> total_time(0LL), 
                    cyclic_shift_time(0LL), shift_reverse_time(0LL), window_partition_time(0LL), window_reverse_time(0LL),
                    q_time(0LL), k_time(0LL), score_tim(0LL), softmax_time(0LL), v_time(0LL), mulV_time(0LL);
                {
                    CppTimer timer(&total_time);
                    output_cts = model_.swin_ct_norm(input_cts, 
                        per_block_seq, embed_dim, num_heads, input_row, input_col, window_size, window_size / 2,
                        weight_query, bias_query, weight_key, bias_key, weight_value, bias_value, position_bias_table,
                        static_cast<double>(window_size), 3.0, 1.0,
                        &cyclic_shift_time, &shift_reverse_time, &window_partition_time, &window_reverse_time,
                        &q_time, &k_time, &score_tim, &softmax_time, &v_time, &mulV_time);
                }
                std::cout << "=============== swin_ct_norm 结果 =================" << std::endl;
                std::cout << "[2-3] swin_ct_norm() Cyclic Shift time: " << std::fixed << std::setprecision(3)
                    << cyclic_shift_time.load() / 1000.0 << " s\n";
                std::cout << "[2-3] swin_ct_norm() Window Partition time: " << std::fixed << std::setprecision(3)
                    << window_partition_time.load() / 1000.0 << " s\n";   
                std::cout << "[2-3] swin_ct_norm() Q time: " << std::fixed << std::setprecision(3)
                    << q_time.load() / 1000.0 << " s\n";
                std::cout << "[2-3] swin_ct_norm() K time: " << std::fixed << std::setprecision(3)
                    << k_time.load() / 1000.0 << " s\n";
                std::cout << "[2-3] swin_ct_norm() Score time: " << std::fixed << std::setprecision(3)
                    << score_tim.load() / 1000.0 << " s\n";
                std::cout << "[2-3] swin_ct_norm() Softmax time: " << std::fixed << std::setprecision(3)
                    << softmax_time.load() / 1000.0 << " s\n";
                std::cout << "[2-3] swin_ct_norm() V time: " << std::fixed << std::setprecision(3)
                    << v_time.load() / 1000.0 << " s\n";
                std::cout << "[2-3] swin_ct_norm() MulV time: " << std::fixed << std::setprecision(3)
                    << mulV_time.load() / 1000.0 << " s\n";
                std::cout << "[2-3] swin_ct_norm() Window Reverse time: " << std::fixed << std::setprecision(3)
                    << window_reverse_time.load() / 1000.0 << " s\n";
                std::cout << "[2-3] swin_ct_norm() Reverse Shift time: " << std::fixed << std::setprecision(3)
                    << shift_reverse_time.load() / 1000.0 << " s\n";
                std::cout << "[2-3] swin_ct_norm() Total time: " << std::fixed << std::setprecision(3)
                    << total_time.load() / 1000.0 << " s\n";
            }
            std::cout << "=============== swin_ct_opti =================" << std::endl;
            {
                std::vector<heongpu::Ciphertext<heongpu::Scheme::CKKS>> output_cts(block_num);
                std::atomic<long long> total_time(0LL), 
                    cyclic_shift_time(0LL), shift_reverse_time(0LL), window_partition_time(0LL), window_reverse_time(0LL),
                    q_time(0LL), k_time(0LL), score_tim(0LL), softmax_time(0LL), v_time(0LL), mulV_time(0LL);
                {
                    CppTimer timer(&total_time);
                    output_cts = model_.swin_ct_opti(input_cts, 
                        per_block_seq, embed_dim, num_heads, input_row, input_col, window_size, window_size / 2,
                        weight_query, bias_query, weight_key, bias_key, weight_value, bias_value, position_bias_table,
                        static_cast<double>(window_size), 3.0, 1.0,
                        &window_partition_time, &window_reverse_time, &q_time, &k_time, &score_tim, &softmax_time, &v_time, &mulV_time);
                }
                std::cout << "=============== swin_ct_opti 结果 =================" << std::endl;
                std::cout << "[2-3] swin_ct_opti() Cyclic Shift time: " << std::fixed << std::setprecision(3)
                    << cyclic_shift_time.load() / 1000.0 << " s\n";
                std::cout << "[2-3] swin_ct_opti() Window Partition time: " << std::fixed << std::setprecision(3)
                    << window_partition_time.load() / 1000.0 << " s\n";   
                std::cout << "[2-3] swin_ct_opti() Q time: " << std::fixed << std::setprecision(3)
                    << q_time.load() / 1000.0 << " s\n";
                std::cout << "[2-3] swin_ct_opti() K time: " << std::fixed << std::setprecision(3)
                    << k_time.load() / 1000.0 << " s\n";
                std::cout << "[2-3] swin_ct_opti() Score time: " << std::fixed << std::setprecision(3)
                    << score_tim.load() / 1000.0 << " s\n";
                std::cout << "[2-3] swin_ct_opti() Softmax time: " << std::fixed << std::setprecision(3)
                    << softmax_time.load() / 1000.0 << " s\n";
                std::cout << "[2-3] swin_ct_opti() V time: " << std::fixed << std::setprecision(3)
                    << v_time.load() / 1000.0 << " s\n";
                std::cout << "[2-3] swin_ct_opti() MulV time: " << std::fixed << std::setprecision(3)
                    << mulV_time.load() / 1000.0 << " s\n";
                std::cout << "[2-3] swin_ct_opti() Window Reverse time: " << std::fixed << std::setprecision(3)
                    << window_reverse_time.load() / 1000.0 << " s\n";
                std::cout << "[2-3] swin_ct_opti() Reverse Shift time: " << std::fixed << std::setprecision(3)
                    << shift_reverse_time.load() / 1000.0 << " s\n";
                std::cout << "[2-3] swin_ct_opti() Total time: " << std::fixed << std::setprecision(3)
                    << total_time.load() / 1000.0 << " s\n";
            }
        }catch (const std::exception &e) {
            std::cerr << "[ERROR] exception : " << e.what() << std::endl;
        }
        std::cout << "[2-4] run [1024] swin_ct_norm()\n";
        try{
            const int input_row = 24, input_col = 24, embed_dim = 1024, head_size = 32;
            const int input_num = input_row * input_col;
            const int per_block_seq = 6;
            const int window_size = 12;
            const int block_num = input_num / per_block_seq;
            const int window_num = input_num / window_size / window_size;
            const int num_heads = embed_dim / head_size;
            std::cout << "=============== 配置说明 =================" << std::endl;
            std::cout << " >> 密文存储 : [ " << per_block_seq << " , " << embed_dim << " ]" << std::endl;
            std::cout << " >> 输入维度 : [ " << input_num << " , " << embed_dim << " ]" << std::endl;
            // 1. 准备输入数据
            std::vector<std::vector<double>> input_mat;
            std::vector<std::vector<double>> weight_query, weight_key, weight_value, position_bias_table;
            std::vector<double> bias_query, bias_key, bias_value;
            {
                input_mat = file_.readMat("ImageBackbone/input/input_mat_36_36_1024");
                input_mat.resize(input_num);
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
            std::cout << "=============== swin_ct_norm =================" << std::endl;
            {
                std::vector<heongpu::Ciphertext<heongpu::Scheme::CKKS>> output_cts(block_num);
                std::atomic<long long> total_time(0LL), 
                    cyclic_shift_time(0LL), shift_reverse_time(0LL), window_partition_time(0LL), window_reverse_time(0LL),
                    q_time(0LL), k_time(0LL), score_tim(0LL), softmax_time(0LL), v_time(0LL), mulV_time(0LL);
                {
                    CppTimer timer(&total_time);
                    output_cts = model_.swin_ct_norm(input_cts, 
                        per_block_seq, embed_dim, num_heads, input_row, input_col, window_size, window_size / 2,
                        weight_query, bias_query, weight_key, bias_key, weight_value, bias_value, position_bias_table,
                        static_cast<double>(window_size), 3.0, 1.0,
                        &cyclic_shift_time, &shift_reverse_time, &window_partition_time, &window_reverse_time,
                        &q_time, &k_time, &score_tim, &softmax_time, &v_time, &mulV_time);
                }
                std::cout << "=============== swin_ct_norm 结果 =================" << std::endl;
                std::cout << "[2-4] swin_ct_norm() Cyclic Shift time: " << std::fixed << std::setprecision(3)
                    << cyclic_shift_time.load() / 1000.0 << " s\n";
                std::cout << "[2-4] swin_ct_norm() Window Partition time: " << std::fixed << std::setprecision(3)
                    << window_partition_time.load() / 1000.0 << " s\n";   
                std::cout << "[2-4] swin_ct_norm() Q time: " << std::fixed << std::setprecision(3)
                    << q_time.load() / 1000.0 << " s\n";
                std::cout << "[2-4] swin_ct_norm() K time: " << std::fixed << std::setprecision(3)
                    << k_time.load() / 1000.0 << " s\n";
                std::cout << "[2-4] swin_ct_norm() Score time: " << std::fixed << std::setprecision(3)
                    << score_tim.load() / 1000.0 << " s\n";
                std::cout << "[2-4] swin_ct_norm() Softmax time: " << std::fixed << std::setprecision(3)
                    << softmax_time.load() / 1000.0 << " s\n";
                std::cout << "[2-4] swin_ct_norm() V time: " << std::fixed << std::setprecision(3)
                    << v_time.load() / 1000.0 << " s\n";
                std::cout << "[2-4] swin_ct_norm() MulV time: " << std::fixed << std::setprecision(3)
                    << mulV_time.load() / 1000.0 << " s\n";
                std::cout << "[2-4] swin_ct_norm() Window Reverse time: " << std::fixed << std::setprecision(3)
                    << window_reverse_time.load() / 1000.0 << " s\n";
                std::cout << "[2-4] swin_ct_norm() Reverse Shift time: " << std::fixed << std::setprecision(3)
                    << shift_reverse_time.load() / 1000.0 << " s\n";
                std::cout << "[2-4] swin_ct_norm() Total time: " << std::fixed << std::setprecision(3)
                    << total_time.load() / 1000.0 << " s\n";
            }
            std::cout << "=============== swin_ct_opti =================" << std::endl;
            {
                std::vector<heongpu::Ciphertext<heongpu::Scheme::CKKS>> output_cts(block_num);
                std::atomic<long long> total_time(0LL), 
                    cyclic_shift_time(0LL), shift_reverse_time(0LL), window_partition_time(0LL), window_reverse_time(0LL),
                    q_time(0LL), k_time(0LL), score_tim(0LL), softmax_time(0LL), v_time(0LL), mulV_time(0LL);
                {
                    CppTimer timer(&total_time);
                    output_cts = model_.swin_ct_opti(input_cts, 
                        per_block_seq, embed_dim, num_heads, input_row, input_col, window_size, window_size / 2,
                        weight_query, bias_query, weight_key, bias_key, weight_value, bias_value, position_bias_table,
                        static_cast<double>(window_size), 3.0, 1.0,
                        &window_partition_time, &window_reverse_time, &q_time, &k_time, &score_tim, &softmax_time, &v_time, &mulV_time);
                }
                std::cout << "=============== swin_ct_opti 结果 =================" << std::endl;
                std::cout << "[2-4] swin_ct_opti() Cyclic Shift time: " << std::fixed << std::setprecision(3)
                    << cyclic_shift_time.load() / 1000.0 << " s\n";
                std::cout << "[2-4] swin_ct_opti() Window Partition time: " << std::fixed << std::setprecision(3)
                    << window_partition_time.load() / 1000.0 << " s\n";   
                std::cout << "[2-4] swin_ct_opti() Q time: " << std::fixed << std::setprecision(3)
                    << q_time.load() / 1000.0 << " s\n";
                std::cout << "[2-4] swin_ct_opti() K time: " << std::fixed << std::setprecision(3)
                    << k_time.load() / 1000.0 << " s\n";
                std::cout << "[2-4] swin_ct_opti() Score time: " << std::fixed << std::setprecision(3)
                    << score_tim.load() / 1000.0 << " s\n";
                std::cout << "[2-4] swin_ct_opti() Softmax time: " << std::fixed << std::setprecision(3)
                    << softmax_time.load() / 1000.0 << " s\n";
                std::cout << "[2-4] swin_ct_opti() V time: " << std::fixed << std::setprecision(3)
                    << v_time.load() / 1000.0 << " s\n";
                std::cout << "[2-4] swin_ct_opti() MulV time: " << std::fixed << std::setprecision(3)
                    << mulV_time.load() / 1000.0 << " s\n";
                std::cout << "[2-4] swin_ct_opti() Window Reverse time: " << std::fixed << std::setprecision(3)
                    << window_reverse_time.load() / 1000.0 << " s\n";
                std::cout << "[2-4] swin_ct_opti() Reverse Shift time: " << std::fixed << std::setprecision(3)
                    << shift_reverse_time.load() / 1000.0 << " s\n";
                std::cout << "[2-4] swin_ct_opti() Total time: " << std::fixed << std::setprecision(3)
                    << total_time.load() / 1000.0 << " s\n";
            }
        }catch (const std::exception &e) {
            std::cerr << "[ERROR] exception : " << e.what() << std::endl;
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
