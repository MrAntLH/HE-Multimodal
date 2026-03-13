#pragma once
#include <heongpu.cuh>
#include <ckks_evaluator.cuh>
#include <vector>
#include <utility>
#include <string>
#include <functional>
#include <cmath>

using namespace heongpu;

class LinearExperi {
public:
    LinearExperi(
        HEContext<Scheme::CKKS>& ctx, HEEncoder<Scheme::CKKS>& encoder,
        CKKSEvaluator& evaluator, double scale)
        : ctx_(ctx), encoder_(encoder), evaluator_(evaluator), scale_(scale), 
          slot_count(ctx_.get_poly_modulus_degree() / 2)
          {}
    /**
     * @brief  实现明文-密文矩阵乘法加偏置计算，数学表达式：y = x @ W.T + b
     * @details 输入密文ct对应明文矩阵X，权重矩阵weight、偏置矩阵bias支持维度冗余（仅取前部分有效维度参与计算）
     *          注：weights 维度最小为 hidden_dim x output_dim，bias 维度最小为 1 x output_dim
     * @param  ct          输入密文，对应明文矩阵X，其维度隐含关联input_dim与hidden_dim
     * @param  weights     明文权重矩阵（支持维度冗余，仅前hidden_dim行、前output_dim列有效）
     * @param  bias        明文偏置矩阵（支持维度冗余，仅前hidden_dim行、前output_dim列有效）
     * @param  input_dim   输入特征维度（对应明文矩阵X的列数/权重矩阵的输入维度）
     * @param  output_dim  输出特征维度（对应结果矩阵Y的列数/权重矩阵的输出维度）
     * @param  hidden_dim  中间隐藏维度（连接输入与输出的桥梁维度，对应通用矩阵乘法的k维度）
     * @return Ciphertext<Scheme::CKKS>  输出密文，对应结果矩阵Y
     */
    Ciphertext<Scheme::CKKS> pt_ct_matmul(Ciphertext<Scheme::CKKS>& ct, const std::vector<std::vector<double>>& weights,
        const std::vector<double>& bias, int input_dim, int hidden_dim, int output_dim);
    
    
    /**
     * @brief  实现密文-密文矩阵乘法计算，数学表达式：Atten_Score = Q @ K.T
     * @details 输入密文ct_q对应明文矩阵Q，密文ct_k_t对应明文矩阵K的转置矩阵K.T
     * @param  ct_q        输入密文，对应明文矩阵Q，其维度隐含关联 seq_dim 与 hidden_dim
     * @param  ct_k_t      输入密文，对应明文矩阵K的转置矩阵K.T，其维度隐含关联 seq_dim 与 hidden_dim
     * @param  seq_dim     序列长度维度（对应明文矩阵Q与K的行数）
     * @param  hidden_dim  隐藏特征维度（对应明文矩阵Q与K的列数）
     * @return Ciphertext<Scheme::CKKS>  输出密文，对应注意力得分矩阵attenscore
     */
    Ciphertext<Scheme::CKKS> ct_ct_attenscore(Ciphertext<Scheme::CKKS>& ct_q, 
        Ciphertext<Scheme::CKKS>& ct_k_t, int seq_dim, int hidden_dim);
    
    /**
     * @brief  按行打包二维double矩阵为CKKS明文（Plaintext）
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

private:
    HEContext<Scheme::CKKS>& ctx_;
    HEEncoder<Scheme::CKKS>& encoder_;
    CKKSEvaluator& evaluator_;

    const double scale_;
    const int slot_count;

    int floor_power_of_two_loop(int input_num);
};
