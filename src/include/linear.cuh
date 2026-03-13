#pragma once
#include <heongpu.cuh>
#include <ckks_evaluator.cuh>
#include <cpp_timer.cuh>
#include <vector>
#include <utility>
#include <string>
#include <functional>
#include <iostream>
#include <cmath>
#include <string>
#include <algorithm>
#include <numeric>

using namespace heongpu;

class Linear {
public:
    Linear(
        HEContext<Scheme::CKKS>& ctx, HEEncoder<Scheme::CKKS>& encoder,
        CKKSEvaluator& evaluator, double scale)
        : ctx_(ctx), encoder_(encoder), evaluator_(evaluator), scale_(scale), 
          slot_count(ctx_.get_poly_modulus_degree() / 2)
          {}
    /**
     * @brief  实现 密文-明文 矩阵乘法 加偏置计算，数学表达式：y = x @ W.T + b
     * @details 输入为行紧凑打包，权重采用列重复打包，结果为 行 紧凑打包
     * @details 使用场景为 Q 和 K 矩阵的计算，维度为 [seq_len, hidden_dim] @ [hidden_dim, hidden_dim] + [hidden_dim]
     * @param  ct          输入密文，对应矩阵X，其维度隐含关联 input_dim 与 hidden_dim
     * @param  weights     明文权重矩阵 [hidden_dim, hidden_dim]
     * @param  bias        明文偏置矩阵 [hidden_dim]
     * @return Ciphertext  输出密文，对应结果矩阵Y
     */
    Ciphertext<Scheme::CKKS> ctpt_matmul_to_row(Ciphertext<Scheme::CKKS>& ct, 
        const std::vector<std::vector<double>>& weights,
        const std::vector<double>& bias, int input_dim, int hidden_dim);
    
    /**
     * @brief  实现 密文-明文 矩阵乘法 加偏置计算，数学表达式：y = x @ W.T + b
     * @details 乘法方式为：行重复打包，列紧凑打包，结果为 列打包（为适配atten_score，每 head_size 中存 seq_len 个，即 1 列）
     * @details 同时逻辑上压缩多头，每 hidden_dim 中存储 head_num 个头, 每个密文共存 per_ct_cols = gcd(seq_len, head_size) 个
     * @return std::vector<Ciphertext>  为便于适配后续操作，此处输出多头的每列至一个密文，共计 head_size / per_ct_cols 个密文
     */
    std::vector<Ciphertext<Scheme::CKKS>> ctpt_matmul_to_col(Ciphertext<Scheme::CKKS>& input_ct, 
        const std::vector<std::vector<double>>& weights, const std::vector<double>& bias, 
        const int seq_len, const int head_num, const int head_size);
    
    /**
     * @brief  实现 注意力得分的 密文-密文矩阵乘法计算，数学表达式：Atten_Score = Q @ K.T
     * @details 输入密文ct_q对应紧凑按行打包的多头矩阵Q，密文ct_kt 可视为矩阵K.T 对应紧凑按列多头打包
     * @details 同时考虑多头，即 逻辑上按多头计算，维度为[seq_dim, hidden_dim/head_num] @ [hidden_dim/head_num, seq_dim]
     * @details 结合输出形式，要求 seq_dim <= hidden_dim/head_num, 即保障每个头的注意力得分矩阵能存进原先的块内
     * @param  ct_q        输入密文，对应明文矩阵Q，其维度隐含关联 seq_dim 与 hidden_dim
     * @param  ct_kt      输入密文，对应明文矩阵K的转置矩阵K.T，其维度隐含关联 seq_dim 与 hidden_dim
     * @return Ciphertext<Scheme::CKKS>  输出密文，每 seq_dim * hidden_dim/head_num 块内 前 seq_dim * seq_dim 元素为行打包得分矩阵
     */
    Ciphertext<Scheme::CKKS> ctct_attenscore(Ciphertext<Scheme::CKKS>& ct_q, 
        Ciphertext<Scheme::CKKS>& ct_kt, int seq_dim, int hidden_dim, int head_num);

    /**
     * @brief  实现 多头注意力最后的 atten_score @ v 输出
     * @details 为便于此处快捷乘法，V是直接按照列打包放在不同密文内，共计 head_size 个密文 
     * @details 此处输入的 cts_atten_score 共 input_ct_nums 个，即受一开始分块策略影响
     * @return Ciphertext<Scheme::CKKS>  输出密文，内存储维度为 [per_seq_len, head_num * head_size] 行打包的结果
     */
    Ciphertext<Scheme::CKKS> ctct_attenOutLinear(std::vector<Ciphertext<Scheme::CKKS>>& cts_atten_score,
        std::vector<std::vector<Ciphertext<Scheme::CKKS>>>& v_cts, 
        const int input_ct_nums, const int per_seq_len, const int head_num, const int head_size);

    /**
     * @brief  实现 多头注意力最后的 atten_score @ v 输出, 单密文版本
     */
    Ciphertext<Scheme::CKKS> ctct_attenOutLinear(Ciphertext<Scheme::CKKS>& cts_atten_score, 
        std::vector<Ciphertext<Scheme::CKKS>>& v_cts, 
        const int per_seq_len, const int head_num, const int head_size);
    
    /**
     * @brief  按行打包二维 double 矩阵为 CKKS 明文（Plaintext）
     * @details 遵循行优先（row-major）打包规则，若输入矩阵行列数小于目标打包行列数，自动填充（补0）;
     * @param  mat           输入二维double矩阵（待打包数据）
     * @param  target_rows   目标打包行数（输入矩阵行数 ≤ 该值）
     * @param  target_cols   目标打包列数（输入矩阵列数 ≤ 该值）
     * @return Plaintext<Scheme::CKKS>  打包后的CKKS明文
     */
    Plaintext<Scheme::CKKS> pack_matrix_by_row(const std::vector<std::vector<double>>& mat,
        int target_rows, int target_cols);

    /**
     * @brief  按行解包CKKS明文（Plaintext）为二维 double 矩阵
     * @param  pack          输入CKKS明文（待解包数据）
     * @param  pack_rows     打包时的行数
     * @param  pack_cols     打包时的列数
     * @param  target_rows   目标解包行数（≤ pack_rows）
     * @param  target_cols   目标解包列数（≤ pack_cols）
     * @return std::vector<std::vector<double>>  解包后的二维double矩阵
     */
    std::vector<std::vector<double>> unpack_matrix_by_row(Plaintext<Scheme::CKKS> pack,
        int pack_rows, int pack_cols, int target_rows, int target_cols);

    // 明文矩阵乘法, 用于验证
    std::vector<std::vector<double>> matMul(const std::vector<std::vector<double>>& A, 
                                        const std::vector<std::vector<double>>& B);
    // 明文矩阵加偏置, 用于验证
    std::vector<std::vector<double>> matAddBias(const std::vector<std::vector<double>>& A, 
                                            const std::vector<double>& bias);                            

private:
    HEContext<Scheme::CKKS>& ctx_;
    HEEncoder<Scheme::CKKS>& encoder_;
    CKKSEvaluator& evaluator_;

    const double scale_;
    const int slot_count;

    int floor_power_of_two_loop(int input_num);
};
