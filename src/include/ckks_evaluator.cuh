#pragma once
#include <heongpu.cuh>
#include <vector>
#include <utility>
#include <string>
#include <iostream>

using namespace heongpu;

class CKKSEvaluator {
public:
    CKKSEvaluator(
        HEContext<Scheme::CKKS>& ctx, HEArithmeticOperator<Scheme::CKKS>& operators, HEEncoder<Scheme::CKKS>& encoder, 
        HEEncryptor<heongpu::Scheme::CKKS> encryptor, HEDecryptor<heongpu::Scheme::CKKS> decryptor,
        Relinkey<Scheme::CKKS>& relin_key, Galoiskey<Scheme::CKKS>& galois_key, double scale, BootstrappingConfig boot_config)
        : ctx_(ctx),operators_(operators), encoder_(encoder), encryptor_(encryptor), decryptor_(decryptor),
         relin_key_(relin_key),galois_key_(galois_key),scale_(scale),boot_config_(boot_config) {
            slot_count = ctx_.get_poly_modulus_degree() / 2;
            max_depth = ctx.get_ciphertext_modulus_count() - boot_config_.StoC_piece_ - 1;
          }

    /*  Multiply */
    void multiply(Ciphertext<Scheme::CKKS>& ct1, Ciphertext<Scheme::CKKS>& ct2, 
        Ciphertext<Scheme::CKKS>& dest);
    void multiply(Ciphertext<Scheme::CKKS>& ct1, Ciphertext<Scheme::CKKS>& ct2);
    void multiply(Ciphertext<Scheme::CKKS>& ct, Plaintext<Scheme::CKKS>& pt, 
        Ciphertext<Scheme::CKKS>& dest);
    void multiply(Ciphertext<Scheme::CKKS>& ct, Plaintext<Scheme::CKKS>& pt);
    void square(Ciphertext<Scheme::CKKS>& ct, Ciphertext<Scheme::CKKS>& dest);
    void square(Ciphertext<Scheme::CKKS>& ct);

    /*  Add */
    void add(Ciphertext<Scheme::CKKS>& ct1, Ciphertext<Scheme::CKKS>& ct2, 
        Ciphertext<Scheme::CKKS>& dest);
    void add(Ciphertext<Scheme::CKKS>& ct1, Ciphertext<Scheme::CKKS>& ct2);
    void add(Ciphertext<Scheme::CKKS>& ct, Plaintext<Scheme::CKKS>& pt, 
        Ciphertext<Scheme::CKKS>& dest);
    void add(Ciphertext<Scheme::CKKS>& ct, Plaintext<Scheme::CKKS>& pt);

    /*  Sub */
    void sub(Ciphertext<Scheme::CKKS>& ct1, Ciphertext<Scheme::CKKS>& ct2, 
        Ciphertext<Scheme::CKKS>& dest);
    void sub(Ciphertext<Scheme::CKKS>& ct1, Ciphertext<Scheme::CKKS>& ct2);
    void sub(Ciphertext<Scheme::CKKS>& ct, Plaintext<Scheme::CKKS>& pt, 
        Ciphertext<Scheme::CKKS>& dest);
    void sub(Ciphertext<Scheme::CKKS>& ct, Plaintext<Scheme::CKKS>& pt);

    /*  Negate */
    void negate(Ciphertext<Scheme::CKKS>& ct, Ciphertext<Scheme::CKKS>& dest);
    void negate(Ciphertext<Scheme::CKKS>& ct);

    /*  Rotate */
    void rotate_rows(Ciphertext<Scheme::CKKS>& ct, int steps, Ciphertext<Scheme::CKKS>& dest);
    void rotate_rows(Ciphertext<Scheme::CKKS>& ct, int steps);

    /*  Encrypt */
    void encrypt(Plaintext<Scheme::CKKS>& pt,  Ciphertext<Scheme::CKKS>& ct);

    /*  Bootstrapping */
    void bootstrap(Ciphertext<Scheme::CKKS>& ct, double scale = pow(2.0, 6));

    /*  Decrypt */
    // 测试使用的解密接口，实际不需要
    std::vector<double> decrypt(Ciphertext<Scheme::CKKS>& ct);

    /*  Util */
    void print_cipher(Ciphertext<Scheme::CKKS> ct, std::string message, int print_size = 3);
    void print_vector(std::vector<double> vec, std::string message, int print_size = 3, int precision = 2);

    void print_matrix(std::vector<std::vector<double>>& matrix, std::string message, 
        int rows, int cols, int print_row_nums, int print_col_nums, int precision = 1);
    void print_matrix(std::vector<double>& matrix, std::string message, int rows, int cols, 
        int print_row_nums, int print_col_nums, int precision = 1);
    void print_matrix(std::vector<std::vector<bool>>& matrix, 
        std::string message, int rows, int cols, int print_row_nums, int print_col_nums);
    
    void print_cipher_matrix(Ciphertext<Scheme::CKKS> ct, std::string message, 
        int rows, int cols, int print_row_nums, int print_col_nums, int precision = 1);

    double calculate_MAE(std::vector<double> true_vec, Ciphertext<Scheme::CKKS> ct);
    double calculate_MAE(std::vector<double> true_vec, std::vector<double> predict_vec);
    double calculate_MAE(std::vector<std::vector<double>> true_mat, std::vector<std::vector<double>> predict_mat);
    int get_max_depth() { return max_depth; }

private:
    HEContext<Scheme::CKKS>& ctx_;
    HEArithmeticOperator<Scheme::CKKS>& operators_;
    HEEncoder<Scheme::CKKS>& encoder_;
    HEEncryptor<heongpu::Scheme::CKKS> encryptor_;
    HEDecryptor<heongpu::Scheme::CKKS> decryptor_;
    Relinkey<Scheme::CKKS>& relin_key_;
    Galoiskey<Scheme::CKKS>& galois_key_;
    BootstrappingConfig& boot_config_;
    double scale_;
    int slot_count;
    int max_depth;
};
