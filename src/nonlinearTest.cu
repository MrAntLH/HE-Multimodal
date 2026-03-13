#include "ckks_evaluator.cuh"
#include "nonlinear_experi.cuh"
#include "linear.cuh"
#include "model.cuh"

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
        ctx.set_poly_modulus_degree(8192);
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
        NonLinearExperi nonLinearExperi_(ctx, encoder, evaluator_, scale);
        NonLinear nonLinear_(ctx, encoder, evaluator_, scale);
        Linear linear_(ctx, encoder, evaluator_, scale);
        Model model_(ctx, encoder, evaluator_, linear_, nonLinear_, scale);

        float duration_time = 0;
        cudaEvent_t start_time, stop_time;
        cudaEventCreate(&start_time);
        cudaEventCreate(&stop_time);

        // ------------------- 1-1) gelu_nexus -------------------
        std::cout << "[1-1] [main] run gelu_nexus()\n";
        try {
            // Test [-5,5] for gelu
            std::vector<double> x(slot_count);
            for (int i = 0; i < slot_count; ++i) 
                x[i] = -5 + 10.0 * i / slot_count;
            heongpu::Plaintext<heongpu::Scheme::CKKS> pt(ctx);
            encoder.encode(pt, x, scale);
            heongpu::Ciphertext<heongpu::Scheme::CKKS> ct(ctx);
            encryptor.encrypt(ct, pt);

            cudaEventRecord(start_time);
            heongpu::Ciphertext<heongpu::Scheme::CKKS> ct_gelu = nonLinearExperi_.gelu_nexus(ct);
            cudaEventRecord(stop_time);
            cudaEventSynchronize(stop_time);
            cudaEventElapsedTime(&duration_time, start_time, stop_time);

            std::vector<double> y_gelu(slot_count);
            const double sqrt_2_over_pi = std::sqrt(2.0 / M_PI);
            for (int i = 0; i < slot_count; ++i){
                y_gelu[i]  = 0.5 * x[i] * (1.0 + std::tanh(sqrt_2_over_pi * (x[i] + 0.044715 * x[i] * x[i] * x[i])));
            }

            evaluator_.print_vector(x, "gelu_nexus input");
            evaluator_.print_vector(y_gelu, "gelu_nexus true");
            evaluator_.print_cipher(ct_gelu, "gelu_nexus result");
            std::cout << "gelu_nexus MAE: " << std::fixed << std::setprecision(5) << evaluator_.calculate_MAE(y_gelu, ct_gelu) << std::endl;
            std::cout << "gelu_nexus Time elapsed: " << duration_time << " ms" << std::endl;
        } catch (const std::exception &e) {
            std::cerr << "[ERROR] exception during gelu_nexus(): " << e.what() << std::endl;
        }

        // ------------------- 1-2) gelu_thor -------------------
        std::cout << "[1-2] [main] run gelu_thor()\n";
        try {
            // Test [-5,5] for gelu
            std::vector<double> x(slot_count);
            for (int i = 0; i < slot_count; ++i) 
                x[i] = -5 + 10.0 * i / slot_count;
            heongpu::Plaintext<heongpu::Scheme::CKKS> pt(ctx);
            encoder.encode(pt, x, scale);
            heongpu::Ciphertext<heongpu::Scheme::CKKS> ct(ctx);
            encryptor.encrypt(ct, pt);

            cudaEventRecord(start_time);
            heongpu::Ciphertext<heongpu::Scheme::CKKS> ct_gelu = nonLinearExperi_.gelu_thor(ct);
            cudaEventRecord(stop_time);
            cudaEventSynchronize(stop_time);
            cudaEventElapsedTime(&duration_time, start_time, stop_time);

            std::vector<double> y_gelu(slot_count);
            const double sqrt_2_over_pi = std::sqrt(2.0 / M_PI);
            for (int i = 0; i < slot_count; ++i){
                y_gelu[i]  = 0.5 * x[i] * (1.0 + std::tanh(sqrt_2_over_pi * (x[i] + 0.044715 * x[i] * x[i] * x[i])));
            }

            evaluator_.print_vector(x, "gelu_thor input");
            evaluator_.print_vector(y_gelu, "gelu_thor true");
            evaluator_.print_cipher(ct_gelu, "gelu_thor result");
            std::cout << "gelu_thor MAE: " << std::fixed << std::setprecision(5) << evaluator_.calculate_MAE(y_gelu, ct_gelu) << std::endl;
            std::cout << "gelu_thor Time elapsed: " << duration_time << " ms" << std::endl;
        } catch (const std::exception &e) {
            std::cerr << "[ERROR] exception during gelu_thor(): " << e.what() << std::endl;
        }

        // ------------------- 2-1) layer_norm_nexus -------------------
        std::cout << "[2-1] [main] run layer_norm_nexus()\n";
        try {
            // Test [-10,10] for layer_norm
            // Because of the repeate, the input need to be smaller than slot_count/2
            int Num = slot_count/2;
            std::vector<double> x(Num);
            for (int i = 0; i < x.size(); ++i) 
                x[i] = -10 + 20.0 * i / x.size();
            heongpu::Plaintext<heongpu::Scheme::CKKS> pt(ctx);
            encoder.encode(pt, x, scale);
            heongpu::Ciphertext<heongpu::Scheme::CKKS> ct(ctx);
            encryptor.encrypt(ct, pt);

            cudaEventRecord(start_time);
            heongpu::Ciphertext<heongpu::Scheme::CKKS> ct_layer_norm = nonLinearExperi_.layer_norm_nexus(ct, Num);
            cudaEventRecord(stop_time);
            cudaEventSynchronize(stop_time);
            cudaEventElapsedTime(&duration_time, start_time, stop_time);

            decryptor.decrypt(pt, ct_layer_norm);
            std::vector<double> y_predict;
            encoder.decode(y_predict, pt);
            y_predict.resize(Num);

            std::vector<double> y_layer_norm(Num);
            // 计算均值
            double mean = std::accumulate(x.begin(), x.end(), 0.0) / y_layer_norm.size();
            // 计算方差
            double variance = 0.0;
            for (double val : x)
                variance += std::pow(val - mean, 2);
            variance /= y_layer_norm.size();
            // 标准化
            for (int i = 0; i < y_layer_norm.size(); ++i)
                y_layer_norm[i]  = (x[i] - mean) / std::sqrt( variance + 1e-5);

            evaluator_.print_vector(x, "layer_norm_nexus input");
            evaluator_.print_vector(y_layer_norm, "layer_norm_nexus true");
            evaluator_.print_vector(y_predict, "layer_norm_nexus result");
            std::cout << "layer_norm_nexus MAE: " << std::fixed << std::setprecision(5) << evaluator_.calculate_MAE(y_layer_norm, ct_layer_norm) << std::endl;
            std::cout << "layer_norm_nexus Time elapsed: " << duration_time << " ms" << std::endl;
        } catch (const std::exception &e) {
            std::cerr << "[ERROR] exception during layer_norm_nexus(): " << e.what() << std::endl;
        }

        // ------------------- 2-2) layer_norm_thor -------------------
        std::cout << "[2-2] [main] run layer_norm_thor()\n";
        try {
            // Test [-10,10] for layer_norm
            // Because of the repeate, the input need to be smaller than slot_count/2
            int Num = slot_count/2;
            std::vector<double> x(Num);
            for (int i = 0; i < x.size(); ++i) 
                x[i] = -10 + 20.0 * i / x.size();
            heongpu::Plaintext<heongpu::Scheme::CKKS> pt(ctx);
            encoder.encode(pt, x, scale);
            heongpu::Ciphertext<heongpu::Scheme::CKKS> ct(ctx);
            encryptor.encrypt(ct, pt);

            cudaEventRecord(start_time);
            heongpu::Ciphertext<heongpu::Scheme::CKKS> ct_layer_norm = nonLinearExperi_.layer_norm_thor(ct, Num, 10);
            cudaEventRecord(stop_time);
            cudaEventSynchronize(stop_time);
            cudaEventElapsedTime(&duration_time, start_time, stop_time);

            decryptor.decrypt(pt, ct_layer_norm);
            std::vector<double> y_predict;
            encoder.decode(y_predict, pt);
            y_predict.resize(Num);

            std::vector<double> y_layer_norm(Num);
            // 计算均值
            double mean = std::accumulate(x.begin(), x.end(), 0.0) / y_layer_norm.size();
            // 计算方差
            double variance = 0.0;
            for (double val : x)
                variance += std::pow(val - mean, 2);
            variance /= y_layer_norm.size();
            // 标准化
            for (int i = 0; i < y_layer_norm.size(); ++i)
                y_layer_norm[i]  = (x[i] - mean) / std::sqrt( variance + 1e-5);

            evaluator_.print_vector(x, "layer_norm_nexus input");
            evaluator_.print_vector(y_layer_norm, "layer_norm_nexus true");
            evaluator_.print_vector(y_predict, "layer_norm_nexus result");
            std::cout << "layer_norm_nexus MAE: " << std::fixed << std::setprecision(5) << evaluator_.calculate_MAE(y_layer_norm, ct_layer_norm) << std::endl;
            std::cout << "layer_norm_nexus Time elapsed: " << duration_time << " ms" << std::endl;
        } catch (const std::exception &e) {
            std::cerr << "[ERROR] exception during layer_norm_nexus(): " << e.what() << std::endl;
        }

        // ------------------- 3-1) softmax_nexus -------------------
        std::cout << "[3-1] [main] run softmax_nexus()\n";
        try {
            // Test [-10,0] for softmax_nexus
            // Because of the repeate, the input need to be smaller than slot_count/2
            int Num = slot_count/2;
            std::vector<double> x(Num);
            for (int i = 0; i < x.size(); ++i) 
                x[i] = -30 + 30.0 * i / x.size();
            heongpu::Plaintext<heongpu::Scheme::CKKS> pt(ctx);
            encoder.encode(pt, x, scale);
            heongpu::Ciphertext<heongpu::Scheme::CKKS> ct(ctx);
            encryptor.encrypt(ct, pt);

            std::vector<double> y_softmax(Num);
            double max_val = *std::max_element(x.begin(), x.end());
            double sum_exp = 0.0;
            for (int i = 0; i < Num; ++i) {
                y_softmax[i] = std::exp(x[i] - max_val);
                sum_exp += y_softmax[i];
            }
            for (int i = 0; i < Num; ++i)
                y_softmax[i] /= sum_exp;
            cudaEventRecord(start_time);
            heongpu::Ciphertext<heongpu::Scheme::CKKS> ct_softmax = nonLinearExperi_.softmax_nexus(ct, Num, max_val);
            cudaEventRecord(stop_time);
            cudaEventSynchronize(stop_time);
            cudaEventElapsedTime(&duration_time, start_time, stop_time);

            decryptor.decrypt(pt, ct_softmax);
            std::vector<double> y_predict;
            encoder.decode(y_predict, pt);
            y_predict.resize(Num);

            evaluator_.print_vector(x, "softmax_nexus input");
            evaluator_.print_vector(y_softmax, "softmax_nexus true");
            evaluator_.print_vector(y_predict, "softmax_nexus result");
            std::cout << "softmax_nexus MAE: " << std::fixed << std::setprecision(5) << evaluator_.calculate_MAE(y_softmax, ct_softmax) << std::endl;
            std::cout << "softmax_nexus Time elapsed: " << duration_time << " ms" << std::endl;
        } catch (const std::exception &e) {
            std::cerr << "[ERROR] exception during softmax_nexus(): " << e.what() << std::endl;
        }

        // ------------------- 3-2) softmax_thor -------------------
        std::cout << "[3-2] [main] run softmax_thor()\n";
        try {
            // Test [-10,10] for softmax_thor
            int Num = slot_count/2;
            std::vector<double> x(Num);
            for (int i = 0; i < x.size(); ++i) 
                x[i] = -16 + 32.0 * i / x.size();
            heongpu::Plaintext<heongpu::Scheme::CKKS> pt(ctx);
            encoder.encode(pt, x, scale);
            heongpu::Ciphertext<heongpu::Scheme::CKKS> ct(ctx);
            encryptor.encrypt(ct, pt);

            cudaEventRecord(start_time);
            heongpu::Ciphertext<heongpu::Scheme::CKKS> ct_softmax = nonLinearExperi_.softmax_thor(ct, Num);
            cudaEventRecord(stop_time);
            cudaEventSynchronize(stop_time);
            cudaEventElapsedTime(&duration_time, start_time, stop_time);

            decryptor.decrypt(pt, ct_softmax);
            std::vector<double> y_predict;
            encoder.decode(y_predict, pt);
            y_predict.resize(Num);

            std::vector<double> y_softmax(Num);
            double sum_exp = 0.0;
            for (int i = 0; i < Num; ++i){
                y_softmax[i] = std::exp(x[i]);
                sum_exp += y_softmax[i];
            }
            for (int i = 0; i < Num; ++i)
                y_softmax[i] /= sum_exp;
            evaluator_.print_vector(x, "softmax_thor input");
            evaluator_.print_vector(y_softmax, "softmax_thor true");
            evaluator_.print_vector(y_predict, "softmax_thor result");
            std::cout << "softmax_thor MAE: " << std::fixed << std::setprecision(5) << evaluator_.calculate_MAE(y_softmax, ct_softmax) << std::endl;
            std::cout << "softmax_thor Time elapsed: " << duration_time << " ms" << std::endl;
        } catch (const std::exception &e) {
            std::cerr << "[ERROR] exception during softmax_thor(): " << e.what() << std::endl;
        }

        // ------------------- 3-3) softmax_thor -------------------
        std::cout << "[3-3] [main] run softmax_our()\n";
        try {
            // Test [-16,16] for softmax_thor
            int Num = slot_count/2;
            std::vector<double> x(Num);
            for (int i = 0; i < x.size(); ++i) 
                x[i] = -32 + 32.0 * i / x.size();
            heongpu::Plaintext<heongpu::Scheme::CKKS> pt(ctx);
            encoder.encode(pt, x, scale);
            heongpu::Ciphertext<heongpu::Scheme::CKKS> ct(ctx);
            encryptor.encrypt(ct, pt);

            cudaEventRecord(start_time);
            heongpu::Ciphertext<heongpu::Scheme::CKKS> ct_softmax = nonLinearExperi_.softmax_our(ct, Num);
            cudaEventRecord(stop_time);
            cudaEventSynchronize(stop_time);
            cudaEventElapsedTime(&duration_time, start_time, stop_time);

            decryptor.decrypt(pt, ct_softmax);
            std::vector<double> y_predict;
            encoder.decode(y_predict, pt);
            y_predict.resize(Num);

            std::vector<double> y_softmax(Num);
            double sum_exp = 0.0;
            for (int i = 0; i < Num; ++i){
                y_softmax[i] = std::exp(x[i]);
                sum_exp += y_softmax[i];
            }
            for (int i = 0; i < Num; ++i)
                y_softmax[i] /= sum_exp;
            evaluator_.print_vector(x, "softmax_our input");
            evaluator_.print_vector(y_softmax, "softmax_our true");
            evaluator_.print_vector(y_predict, "softmax_our result");
            std::cout << "softmax_our MAE: " << std::fixed << std::setprecision(5) << evaluator_.calculate_MAE(y_softmax, ct_softmax) << std::endl;
            std::cout << "softmax_our Time elapsed: " << duration_time << " ms" << std::endl;
        } catch (const std::exception &e) {
            std::cerr << "[ERROR] exception during softmax_our(): " << e.what() << std::endl;
        }

        // ------------------- 4-1) argmax_nexus -------------------
        std::cout << "[4-1] [main] run argmax_nexus() - 64\n";
        try {
            // Test [-0.5,0.5] for argmax
            // Because of the repeate, the input need to be smaller than slot_count/2
            int Num = 64; // 2^5 = 32
            std::vector<double> x(Num, 0);
            for (int i = 0; i < Num; ++i) 
                x[i] = -0.5 + 1.0 * i / Num;
            heongpu::Plaintext<heongpu::Scheme::CKKS> pt(ctx);
            encoder.encode(pt, x, scale);
            heongpu::Ciphertext<heongpu::Scheme::CKKS> ct(ctx);
            encryptor.encrypt(ct, pt);

            cudaEventRecord(start_time);
            heongpu::Ciphertext<heongpu::Scheme::CKKS> ct_argmax = nonLinearExperi_.ArgmaxNexus(ct, Num);
            cudaEventRecord(stop_time);
            cudaEventSynchronize(stop_time);
            cudaEventElapsedTime(&duration_time, start_time, stop_time);

            decryptor.decrypt(pt, ct_argmax);
            std::vector<double> y_predict;
            encoder.decode(y_predict, pt);
            y_predict.resize(Num);

            int max_idx = std::max_element(x.begin(), x.end()) - x.begin();
            std::vector<double> y_argmax(Num, 0);
            y_argmax[max_idx] = 1;
            
            evaluator_.print_vector(x, "argmax_nexus input");
            evaluator_.print_vector(y_argmax, "argmax_nexus true");
            evaluator_.print_vector(y_predict, "argmax_nexus predict");
            std::cout << "argmax_nexus MAE: " << std::fixed << std::setprecision(5) << evaluator_.calculate_MAE(y_argmax, ct_argmax) << std::endl;
            std::cout << "argmax_nexus Time elapsed: " << duration_time << " ms" << std::endl;
        } catch (const std::exception &e) {
            std::cerr << "[ERROR] exception during argmax_nexus(): " << e.what() << std::endl;
        }

        // ------------------- 4-2) argmax_efficient -------------------
        std::cout << "[4-2] [main] run argmax_efficient -64 \n";
        try {
            // Test [-0.5,0.5] for argmax_efficient
            // Because of the repeate, the input need to be smaller than sqrt(slot_count)
            int Num = 64;
            std::vector<double> x(Num, 0);
            for (int i = 0; i < Num; ++i) 
                x[i] = -0.5 + 1.0 * i / Num;
            heongpu::Plaintext<heongpu::Scheme::CKKS> pt(ctx);
            encoder.encode(pt, x, scale);
            heongpu::Ciphertext<heongpu::Scheme::CKKS> ct(ctx);
            encryptor.encrypt(ct, pt);

            cudaEventRecord(start_time);
            heongpu::Ciphertext<heongpu::Scheme::CKKS> ct_argmax = nonLinearExperi_.ArgmaxEffi(ct, Num);
            cudaEventRecord(stop_time);
            cudaEventSynchronize(stop_time);
            cudaEventElapsedTime(&duration_time, start_time, stop_time);

            decryptor.decrypt(pt, ct_argmax);
            std::vector<double> y_predict;
            encoder.decode(y_predict, pt);
            y_predict.resize(Num);

            int max_idx = std::max_element(x.begin(), x.end()) - x.begin();
            std::vector<double> y_argmax(Num, 0);
            y_argmax[max_idx] = 1;
            
            evaluator_.print_vector(x, "argmax_efficient input");
            evaluator_.print_vector(y_argmax, "argmax true");
            evaluator_.print_vector(y_predict, "argmax_efficient predicted");
            std::cout << "argmax_efficient MAE: " << std::fixed << std::setprecision(5) << evaluator_.calculate_MAE(y_argmax, ct_argmax) << std::endl;
            std::cout << "argmax_efficient Time elapsed: " << duration_time << " ms" << std::endl;
        } catch (const std::exception &e) {
            std::cerr << "[ERROR] exception during argmax_efficient(): " << e.what() << std::endl;
        }

        // ------------------- 4-3) argmax_our -------------------
        std::cout << "[4-3] [main] run argmax_our -64 \n";
        try {
            // Test [-0.5,0.5] for argmax_our
            // Because of the repeate, the input need to be smaller than sqrt(slot_count)
            int Num = 64;
            std::vector<double> x(Num, 0);
            for (int i = 0; i < Num; ++i) 
                x[i] = -0.5 + 1.0 * i / Num;
            heongpu::Plaintext<heongpu::Scheme::CKKS> pt(ctx);
            encoder.encode(pt, x, scale);
            heongpu::Ciphertext<heongpu::Scheme::CKKS> ct(ctx);
            encryptor.encrypt(ct, pt);

            cudaEventRecord(start_time);
            heongpu::Ciphertext<heongpu::Scheme::CKKS> ct_argmax = nonLinearExperi_.ArgmaxOur(ct, Num);
            cudaEventRecord(stop_time);
            cudaEventSynchronize(stop_time);
            cudaEventElapsedTime(&duration_time, start_time, stop_time);

            decryptor.decrypt(pt, ct_argmax);
            std::vector<double> y_predict;
            encoder.decode(y_predict, pt);
            y_predict.resize(Num);

            int max_idx = std::max_element(x.begin(), x.end()) - x.begin();
            std::vector<double> y_argmax(Num, 0);
            y_argmax[max_idx] = 1;
            
            evaluator_.print_vector(x, "argmax_our input");
            evaluator_.print_vector(y_argmax, "argmax true");
            evaluator_.print_vector(y_predict, "argmax_our predicted");
            std::cout << "argmax_our MAE: " << std::fixed << std::setprecision(5) << evaluator_.calculate_MAE(y_argmax, ct_argmax) << std::endl;
            std::cout << "argmax_our Time elapsed: " << duration_time << " ms" << std::endl;
        } catch (const std::exception &e) {
            std::cerr << "[ERROR] exception during argmax_our(): " << e.what() << std::endl;
        }

        // ------------------- 5) tanh_thor -------------------
        std::cout << "[5] [main] run tanh_thor() - 64\n";
        try {
            // Test [-5,5] for tanh_thor
            std::vector<double> x(slot_count);
            for (int i = 0; i < slot_count; ++i) 
                x[i] = -5 + 10.0 * i / slot_count;
            heongpu::Plaintext<heongpu::Scheme::CKKS> pt(ctx);
            encoder.encode(pt, x, scale);
            heongpu::Ciphertext<heongpu::Scheme::CKKS> ct(ctx);
            encryptor.encrypt(ct, pt);

            cudaEventRecord(start_time);
            heongpu::Ciphertext<heongpu::Scheme::CKKS> ct_thor = nonLinearExperi_.tanh_thor(ct);
            cudaEventRecord(stop_time);
            cudaEventSynchronize(stop_time);
            cudaEventElapsedTime(&duration_time, start_time, stop_time);

            std::vector<double> y_thor(slot_count);
            for (int i = 0; i < slot_count; ++i){
                if(x[i] >= 0)
                    y_thor[i] = (std::exp(2 * x[i]) - 1) / (std::exp(2 * x[i]) + 1);
                else
                    y_thor[i] = (1 - std::exp(-2 * x[i])) / (1 + std::exp(-2 * x[i]));
            }

            evaluator_.print_vector(x, "tanh_thor input");
            evaluator_.print_vector(y_thor, "tanh_thor true");
            evaluator_.print_cipher(ct_thor, "tanh_thor result");
            std::cout << "tanh_thor MAE: " << std::fixed << std::setprecision(5) << evaluator_.calculate_MAE(y_thor, ct_thor) << std::endl;
            std::cout << "tanh_thor Time elapsed: " << duration_time << " ms" << std::endl;
        } catch (const std::exception &e) {
            std::cerr << "[ERROR] exception during tanh_thor(): " << e.what() << std::endl;
        }
        

        // ------------------- 6) Grid Sample -------------------
        std::cout << "[6] [main] run GridSampleOurs() - [grid=16] \n";
        try {
            int N = 4;
            double x_scale = 0.5, y_scale = 0.5;
            std::vector<double> vec_grid(slot_count, 0);
            heongpu::Ciphertext<heongpu::Scheme::CKKS> ct_grid(ctx), ct_x_scale(ctx), ct_y_scale(ctx);
            {
                heongpu::Plaintext<heongpu::Scheme::CKKS> pt(ctx);

                for(int i = 0; i < N; i++){
                    for(int j = 0; j < N; j++){
                        vec_grid[i * N + j] = i + 1 + 0.1 * (j + 1);
                    }
                }
                encoder.encode(pt, vec_grid, scale);
                encryptor.encrypt(ct_grid, pt);

                encoder.encode(pt, std::vector<double>(slot_count, x_scale), scale);
                encryptor.encrypt(ct_x_scale, pt);

                encoder.encode(pt, std::vector<double>(slot_count, y_scale), scale);
                encryptor.encrypt(ct_y_scale, pt);
            }

            cudaEventRecord(start_time);
            heongpu::Ciphertext<heongpu::Scheme::CKKS> ct_GridSample = nonLinearExperi_.GridSampleOurs(ct_grid, 
                ct_x_scale, ct_y_scale, N, N);
            cudaEventRecord(stop_time);
            cudaEventSynchronize(stop_time);
            cudaEventElapsedTime(&duration_time, start_time, stop_time);

            std::vector<double> vec_predict;
            {
                heongpu::Plaintext<heongpu::Scheme::CKKS> pt(ctx);
                decryptor.decrypt(pt, ct_GridSample);
                encoder.decode(vec_predict, pt);
                vec_predict.resize(1);
            }

            std::vector<double> truth_GridSample(1);
            {
                double x = (N - 1) * (x_scale + 1) / 2;
                double y = (N - 1) * (y_scale + 1) / 2;
                int x_round = std::round(x);
                int y_round = std::round(y);
                truth_GridSample[0] = vec_grid[x_round * N + y_round];
            }

            evaluator_.print_matrix(vec_grid, 
                        " GridSampleOurs input, x_scale=" + std::to_string(x_scale) + 
                        ", y_scale=" + std::to_string(y_scale), 
                        N, N, N, N, 2);
            evaluator_.print_vector(truth_GridSample, " GridSampleOurs true", 1);
            evaluator_.print_vector(vec_predict, " GridSampleOurs predict", 1);
            std::cout << " GridSampleOurs MAE: " << std::fixed << std::setprecision(5) << evaluator_.calculate_MAE(truth_GridSample, vec_predict) << std::endl;
            std::cout << " GridSampleOurs Time elapsed: " << duration_time << " ms" << std::endl;
        } catch (const std::exception &e) {
            std::cerr << "[ERROR] exception during GridSampleOurs(): " << e.what() << std::endl;
        }

        // ------------------- 7) TopK Gather -------------------
        std::cout << "[7] [main] run TopkGatherOurs() - [N=8, K=4] - coords = [x, y, w, h] \n";
        try {
            int N = 8, K = 4;;
            std::vector<double> vec_query_scores(slot_count, 0), vec_query_coords(slot_count, 0);
            heongpu::Ciphertext<heongpu::Scheme::CKKS> ct_query_scores(ctx);
            std::vector<heongpu::Ciphertext<heongpu::Scheme::CKKS>> ct_query_coords(4);
            {
                srand(static_cast<unsigned int>(time(nullptr))); 
                heongpu::Plaintext<heongpu::Scheme::CKKS> pt(ctx);
                for(int i = 0; i < N; i++){
                    vec_query_scores[i] = 1.0 + 9.0 * rand() / static_cast<double>(RAND_MAX);
                }
                encoder.encode(pt, vec_query_scores, scale);
                encryptor.encrypt(ct_query_scores, pt);

                for(int coord_idx = 0; coord_idx < 4; coord_idx++){
                    std::vector<double> vec_temp_coord(slot_count, 0);
                    for(int i = 0; i < N; i++){
                        vec_query_coords[i * 4 + coord_idx] = i + 1 + 0.11 * (coord_idx + 1);
                        vec_temp_coord[i] = vec_query_coords[i * 4 + coord_idx];
                    }
                    encoder.encode(pt, vec_temp_coord, scale);
                    heongpu::Ciphertext<heongpu::Scheme::CKKS> ct_temp_coord(ctx);
                    encryptor.encrypt(ct_temp_coord, pt);
                    ct_query_coords[coord_idx] = ct_temp_coord;
                }
            }
            std::vector<heongpu::Ciphertext<heongpu::Scheme::CKKS>> ct_results(4);
            cudaEventRecord(start_time);

            nonLinearExperi_.TopkGatherOurs(ct_query_scores, ct_query_coords, N, K, ct_results);

            cudaEventRecord(stop_time);
            cudaEventSynchronize(stop_time);
            cudaEventElapsedTime(&duration_time, start_time, stop_time);

            std::vector<double> vec_predict(K * 4, 0);
            {
                for(int coord_idx = 0; coord_idx < 4; coord_idx++){
                    heongpu::Plaintext<heongpu::Scheme::CKKS> pt(ctx);
                    decryptor.decrypt(pt, ct_results[coord_idx]);
                    std::vector<double> vec_temp_predict;
                    encoder.decode(vec_temp_predict, pt);
                    for(int i = 0; i < K; i++){
                        vec_predict[i * 4 + coord_idx] = vec_temp_predict[i];
                    }
                }
            }
            vec_query_scores.resize(N);
            vec_query_coords.resize(N * 4);
            evaluator_.print_vector(vec_query_scores, " TopkGatherOurs input scores: ", N / 2);
            evaluator_.print_matrix(vec_query_coords, " TopkGatherOurs input coords: ", N, 4, 4, 4, 2);
            std::vector<double> vec_truth(K * 4, 0);
            {
                for(int i = 0; i < K; i++){
                    auto max_iter = std::max_element(vec_query_scores.begin(), vec_query_scores.begin() + N);
                    int max_idx = max_iter - vec_query_scores.begin();
                    vec_query_scores[max_idx] = -1e9; // mark as used
                    for(int coord_idx = 0; coord_idx < 4; coord_idx++){
                        vec_truth[i * 4 + coord_idx] = vec_query_coords[max_idx * 4 + coord_idx];
                    }
                }
            }
            evaluator_.print_matrix(vec_truth, " TopkGatherOurs true: ", K, 4, K, 4, 2);
            evaluator_.print_matrix(vec_predict, " TopkGatherOurs predict: ", K, 4, K, 4, 2);

            std::cout << " TopkGatherOurs MAE: " << std::fixed << std::setprecision(5) << evaluator_.calculate_MAE(vec_truth, vec_predict) << std::endl;
            std::cout << " TopkGatherOurs Time elapsed: " << duration_time << " ms" << std::endl;
        } catch (const std::exception &e) {
            std::cerr << "[ERROR] exception during TopkGatherOurs(): " << e.what() << std::endl;
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
