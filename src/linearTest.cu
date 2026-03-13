#include "ckks_evaluator.cuh"
#include "linear_experi.cuh"

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
                50, 50, 50, 50, 50, 50, 50, 50, 50},
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
        LinearExperi linear_(ctx, encoder, evaluator_, scale);

        float duration_time = 0;
        cudaEvent_t start_time, stop_time;
        cudaEventCreate(&start_time);
        cudaEventCreate(&stop_time);

        // ------------------- [L1] Pt Ct Matmul -------------------
        std::cout << "[L1] [Linear] run Pt_Ct_Matmul()\n";
        try{
            size_t input_dim = 8, hidden_dim = 768, output_dim = 768;
            std::vector<std::vector<double>> input_mat(input_dim, std::vector<double>(hidden_dim, 0));
            for(int i = 0; i < input_dim; ++i)
                for(int j = 0; j < hidden_dim; ++j)
                    input_mat[i][j] = ((double)i * hidden_dim + j) / (double)(input_dim * hidden_dim);
            evaluator_.print_matrix(input_mat, 
                " Input Matrix [ " + std::to_string(input_dim) + " x " + std::to_string(hidden_dim) + " ]:", 
                input_dim, hidden_dim, 4, 4, 3);
            std::vector<std::vector<double>> weight_mat(hidden_dim, std::vector<double>(output_dim, 0));
            for(int i = 0; i < hidden_dim; ++i)
                for(int j = 0; j < output_dim; ++j)
                    weight_mat[i][j] = ((double)i * output_dim + j) / (double)(hidden_dim * output_dim);
            evaluator_.print_matrix(weight_mat, 
                " Weight Matrix [ " + std::to_string(hidden_dim) + " x " + std::to_string(output_dim) + " ]:",
                hidden_dim, output_dim, 4, 4, 3);
            std::vector<double> bias_vec(output_dim, 0.0);
            for(int i = 0; i < output_dim; ++i)
                bias_vec[i] = (double)i / (double)output_dim;
            evaluator_.print_matrix(bias_vec, 
                " Bias Vector [ " + std::to_string(output_dim) + " ]:", 
                1, output_dim, 1, 4, 3);

            heongpu::Plaintext<heongpu::Scheme::CKKS> input_pt;
            input_pt = linear_.pack_matrix_by_row(input_mat, input_dim, hidden_dim);
            heongpu::Ciphertext<heongpu::Scheme::CKKS> input_ct(ctx);
            encryptor.encrypt(input_ct, input_pt);
            cudaEventRecord(start_time);
            heongpu::Ciphertext<heongpu::Scheme::CKKS> output_ct = 
                linear_.pt_ct_matmul(input_ct, weight_mat, bias_vec, input_dim, hidden_dim, output_dim);
            cudaEventRecord(stop_time);
            cudaEventSynchronize(stop_time);
            cudaEventElapsedTime(&duration_time, start_time, stop_time);

            heongpu::Plaintext<heongpu::Scheme::CKKS> output_pt;
            decryptor.decrypt(output_pt, output_ct);
            std::vector<std::vector<double>> output_mat = 
                linear_.unpack_matrix_by_row(output_pt, input_dim, output_dim, input_dim, output_dim);
            evaluator_.print_matrix(output_mat,
                " Predict Matrix [ " + std::to_string(input_dim) + " x " + std::to_string(output_dim) + " ]:", 
                input_dim, output_dim, 4, 4, 3);

            // Calculate true result
            std::vector<std::vector<double>> true_mat(input_dim, std::vector<double>(output_dim, 0.0));
            for(int i = 0; i < input_dim; ++i){
                for(int j = 0; j < output_dim; ++j){
                    for(int k = 0; k < hidden_dim; ++k){
                        true_mat[i][j] += input_mat[i][k] * weight_mat[k][j];
                    }
                    true_mat[i][j] += bias_vec[j];
                }
            }
            evaluator_.print_matrix(true_mat,
                " True Matrix [ " +  std::to_string(input_dim) + " x " + std::to_string(output_dim) + " ]:",
                input_dim, output_dim, 4, 4, 3);
            std::cout << " Pt_Ct_Matmul MAE: " << std::fixed << std::setprecision(5) << evaluator_.calculate_MAE(output_mat, true_mat) << std::endl;
            std::cout << " Pt_Ct_Matmul Time elapsed: " << duration_time << " ms" << std::endl;
        }catch (const std::exception &e) {
            std::cerr << "[ERROR] exception during Pt_Ct_Matmul(): " << e.what() << std::endl;
        }
        

        // ------------------- [L2] Ct Ct Atten Score -------------------
        std::cout << "[L2] [Linear] run Ct_Ct_AttenScore()\n";
        try{
            int seq_dim = 8, hidden_dim = 64;

            std::vector<std::vector<double>> query_mat(seq_dim, std::vector<double>(hidden_dim, 0));
            for(int i = 0; i < seq_dim; ++i)
                for(int j = 0; j < hidden_dim; ++j)
                    query_mat[i][j] = ((double)i * hidden_dim + j) / (double)(seq_dim * hidden_dim);
            evaluator_.print_matrix(query_mat, 
                " Query Matrix [ " + std::to_string(seq_dim) + " x " + std::to_string(hidden_dim) + " ]:", 
                seq_dim, hidden_dim, 4, 4, 3);

            std::vector<std::vector<double>> key_mat(seq_dim, std::vector<double>(hidden_dim, 0));
            for(int i = 0; i < seq_dim; ++i)
                for(int j = 0; j < hidden_dim; ++j)
                    key_mat[i][j] = ((double)(seq_dim - i -1) * hidden_dim + j) / (double)(seq_dim * hidden_dim);
            evaluator_.print_matrix(key_mat,
                " Key Matrix [ " + std::to_string(seq_dim) + " x " + std::to_string(hidden_dim) + " ]:",
                seq_dim, hidden_dim, 4, 4, 3);

            heongpu::Plaintext<heongpu::Scheme::CKKS> query_pt;
            query_pt = linear_.pack_matrix_by_row(query_mat, seq_dim, hidden_dim);
            heongpu::Ciphertext<heongpu::Scheme::CKKS> query_ct(ctx);
            encryptor.encrypt(query_ct, query_pt);

            heongpu::Plaintext<heongpu::Scheme::CKKS> key_pt;
            key_pt = linear_.pack_matrix_by_row(key_mat, seq_dim, hidden_dim);
            heongpu::Ciphertext<heongpu::Scheme::CKKS> key_ct(ctx);
            encryptor.encrypt(key_ct, key_pt);

            cudaEventRecord(start_time);
            heongpu::Ciphertext<heongpu::Scheme::CKKS> atten_score_ct = 
                linear_.ct_ct_attenscore(query_ct, key_ct, seq_dim, hidden_dim);
            cudaEventRecord(stop_time);
            cudaEventSynchronize(stop_time);
            cudaEventElapsedTime(&duration_time, start_time, stop_time);

            heongpu::Plaintext<heongpu::Scheme::CKKS> atten_score_pt;
            decryptor.decrypt(atten_score_pt, atten_score_ct);
            std::vector<std::vector<double>> atten_score_mat = 
                linear_.unpack_matrix_by_row(atten_score_pt, seq_dim, seq_dim, seq_dim, seq_dim);
            evaluator_.print_matrix(atten_score_mat,
                " Predict atten_score mat [ " + std::to_string(seq_dim) + " x " + std::to_string(seq_dim) + " ]:", 
                seq_dim, seq_dim, 4, 4, 3);

            // Calculate true result
            std::vector<std::vector<double>> true_mat(seq_dim, std::vector<double>(seq_dim, 0.0));
            for(int i = 0; i < seq_dim; ++i){
                for(int j = 0; j < seq_dim; ++j){
                    for(int k = 0; k < hidden_dim; ++k){
                        true_mat[i][j] += query_mat[i][k] * key_mat[j][k];
                    }
                }
            }
            evaluator_.print_matrix(true_mat,
                " True atten_score mat [ " +  std::to_string(seq_dim) + " x " + std::to_string(seq_dim) + " ]:",
                seq_dim, seq_dim, 4, 4, 3);

            std::cout << " Ct_Ct_AttenScore MAE: " << std::fixed << std::setprecision(5) << evaluator_.calculate_MAE(atten_score_mat, true_mat) << std::endl;
            std::cout << " Ct_Ct_AttenScore Time elapsed: " << duration_time << " ms" << std::endl;
        }catch (const std::exception &e) {
            std::cerr << "[ERROR] exception during Ct_Ct_AttenScore(): " << e.what() << std::endl;
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
