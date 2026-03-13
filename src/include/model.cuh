#pragma once
#include <heongpu.cuh>
#include <ckks_evaluator.cuh>
#include <linear.cuh>
#include <nonlinear.cuh>
#include <file_handler.cuh>
#include <cpp_timer.cuh>

#include <vector>
#include <utility>
#include <string>
#include <functional>
#include <cmath>
#include <algorithm>

using namespace heongpu;

class Model {
public:
    Model(
        HEContext<Scheme::CKKS>& ctx, HEEncoder<Scheme::CKKS>& encoder,
        CKKSEvaluator& evaluator, Linear& linear, NonLinear& nonlinear, 
        fileHandler& file, double scale)
        : ctx_(ctx), encoder_(encoder), evaluator_(evaluator), 
          linear_(linear), nonlinear_(nonlinear), file_(file), scale_(scale),
          slot_count(ctx_.get_poly_modulus_degree() / 2)
          {}
    
    /**
     * @brief  计算 Bert 的优化版密文推理，普通版见multiHeadAtten_ct，针对 "片段级掩码"进行了优化，减少了计算量
     * @details 具体来说:根据 atten_mask 对输入进行切割，以小份输入的形式分别执行推理，减少计算量
     */
    std::vector<Ciphertext<Scheme::CKKS>> bert_ct_opti(
                        std::vector<Ciphertext<Scheme::CKKS>>& input_cts,
                        const int seq_len, const int hidden_dim, const int num_heads,
                        const std::vector<std::vector<bool>>& atten_mask,
                        const std::vector<std::vector<double>>& Wq, const std::vector<double>& Bq,
                        const std::vector<std::vector<double>>& Wk, const std::vector<double>& Bk,
                        const std::vector<std::vector<double>>& Wv,const std::vector<double>& Bv,
                        const double soft_bias, const int soft_delta_exp, const double soft_inv_scale,
                        std::atomic<long long>* q_time = nullptr, std::atomic<long long>* k_time = nullptr, 
                        std::atomic<long long>* score_time = nullptr, std::atomic<long long>* softmax_time = nullptr, 
                        std::atomic<long long>* v_time = nullptr, std::atomic<long long>* mulV_time = nullptr);
    /**
     * @brief  计算 swin 的密文推理，不同于 Bert，视觉模型 swin 需要考虑窗口内循环移位操作
     * @details 具体来说: 先根据窗口值的一半进行循环移位，再进行多头自注意力计算，最后再反向循环移位回去
     */
    std::vector<Ciphertext<Scheme::CKKS>> swin_ct_norm(
                        std::vector<Ciphertext<Scheme::CKKS>>& input_cts,
                        const int per_seq_len, const int hidden_dim, const int num_heads,
                        const int input_row, const int input_col, const int window_size, const int shift_size,
                        const std::vector<std::vector<double>>& Wq, const std::vector<double>& Bq,
                        const std::vector<std::vector<double>>& Wk, const std::vector<double>& Bk,
                        const std::vector<std::vector<double>>& Wv, const std::vector<double>& Bv,
                        const std::vector<std::vector<double>>& position_bias_table,
                        const double soft_bias, const int soft_delta_exp, const double soft_inv_scale,
                        std::atomic<long long>* cyclic_shift_time = nullptr, std::atomic<long long>* shift_reverse_time = nullptr,
                        std::atomic<long long>* window_partition_time = nullptr, std::atomic<long long>* window_reverse_time = nullptr,
                        std::atomic<long long>* q_time = nullptr, std::atomic<long long>* k_time = nullptr, 
                        std::atomic<long long>* score_time = nullptr, std::atomic<long long>* softmax_time = nullptr, 
                        std::atomic<long long>* v_time = nullptr, std::atomic<long long>* mulV_time = nullptr);
    /**
     * @brief  计算 swin 的优化密文推理，无需进行循环移位，适配不同的位置编码来节省移位操作
     * @details 具体来说: 根据mask的尺寸，采取不同的窗口切割方案，同时适配底层不同的位置编码，减少计算量
     * @details 注：此处实现是针对 模型 shift_size = window_size / 2 进行的设计，其他尺寸补充实现即可
     */
    std::vector<Ciphertext<Scheme::CKKS>> swin_ct_opti(
                        std::vector<Ciphertext<Scheme::CKKS>>& input_cts,
                        const int per_seq_len, const int hidden_dim, const int num_heads,
                        const int input_row, const int input_col, const int window_size, const int shift_size,
                        const std::vector<std::vector<double>>& Wq, const std::vector<double>& Bq,
                        const std::vector<std::vector<double>>& Wk, const std::vector<double>& Bk,
                        const std::vector<std::vector<double>>& Wv, const std::vector<double>& Bv,
                        const std::vector<std::vector<double>>& position_bias_table,
                        const double soft_bias, const int soft_delta_exp, const double soft_inv_scale,
                        std::atomic<long long>* window_partition_time = nullptr, std::atomic<long long>* window_reverse_time = nullptr,
                        std::atomic<long long>* q_time = nullptr, std::atomic<long long>* k_time = nullptr, 
                        std::atomic<long long>* score_time = nullptr, std::atomic<long long>* softmax_time = nullptr, 
                        std::atomic<long long>* v_time = nullptr, std::atomic<long long>* mulV_time = nullptr);

    /**
     * @brief  计算 多头自注意力机制
     * @details 输入为按行分块后的多个密文，每个密文包含 per_input_seq_len = seq_len / input_cts.size() 行
     * @details 目前不支持 per_input_seq_len * per_input_seq_len > head_size 尺寸，中间结果存不下
     * @param  input_cts           输入密文序列（支持多个密文，按序列顺序存储）
     * @return vector<Ciphertext<Scheme::CKKS>>  对应计算结果
     */
    std::vector<Ciphertext<Scheme::CKKS>> multiHeadAtten_ct(
                        std::vector<Ciphertext<Scheme::CKKS>>& input_ct,
                        const int seq_len, const int hidden_dim, const int num_heads,
                        const std::vector<std::vector<bool>>& atten_mask,
                        const std::vector<std::vector<double>>& Wq, const std::vector<double>& Bq,
                        const std::vector<std::vector<double>>& Wk, const std::vector<double>& Bk,
                        const std::vector<std::vector<double>>& Wv,const std::vector<double>& Bv,
                        const double soft_bias, const int soft_delta_exp, const double soft_inv_scale,
                        std::atomic<long long>* q_time = nullptr, std::atomic<long long>* k_time = nullptr, 
                        std::atomic<long long>* score_time = nullptr, std::atomic<long long>* softmax_time = nullptr, 
                        std::atomic<long long>* v_time = nullptr, std::atomic<long long>* mulV_time = nullptr);
    /**
     * @brief  计算 多头自注意力机制
     * @details 上一个的单密文版本，适用于 不分块 的场景
     */
    Ciphertext<Scheme::CKKS> multiHeadAtten_ct(
                        Ciphertext<Scheme::CKKS>& input_cts,
                        const int per_input_seq_len, const int hidden_dim, const int num_heads, 
                        const std::vector<std::vector<bool>>& atten_mask,
                        const std::vector<std::vector<double>>& Wq, const std::vector<double>& Bq,
                        const std::vector<std::vector<double>>& Wk, const std::vector<double>& Bk,
                        const std::vector<std::vector<double>>& Wv,const std::vector<double>& Bv,
                        const double soft_bias, const int soft_delta_exp, const double soft_inv_scale,
                        std::atomic<long long>* q_time = nullptr, std::atomic<long long>* k_time = nullptr, 
                        std::atomic<long long>* score_time = nullptr, std::atomic<long long>* softmax_time = nullptr, 
                        std::atomic<long long>* v_time = nullptr, std::atomic<long long>* mulV_time = nullptr);

    /**
     * @brief  对应 multiHeadAtten 的明文推理
     */
    std::vector<std::vector<double>> multiHeadAtten_pt(
                        const std::vector<std::vector<double>>& input,
                        int seq_len, int hidden_dim, int num_heads,
                        const std::vector<std::vector<bool>>& atten_mask,
                        const std::vector<std::vector<double>>& Wq, const std::vector<double>& Bq,
                        const std::vector<std::vector<double>>& Wk, const std::vector<double>& Bk,
                        const std::vector<std::vector<double>>& Wv,const std::vector<double>& Bv);
    /**
     * @brief  对应 swin 的明文推理
     */
    std::vector<std::vector<double>> swin_pt(const std::vector<std::vector<double>>& input_mat, 
                        const int input_row, const int input_col, const int num_heads, const int window_size, const int shift_size,
                        const std::vector<std::vector<double>>& Wq, const std::vector<double>& Bq,
                        const std::vector<std::vector<double>>& Wk, const std::vector<double>& Bk,
                        const std::vector<std::vector<double>>& Wv, const std::vector<double>& Bv,
                        const std::vector<std::vector<double>>& position_bias_table);

private:
    HEContext<Scheme::CKKS>& ctx_;
    HEEncoder<Scheme::CKKS>& encoder_;
    CKKSEvaluator& evaluator_;
    Linear& linear_;
    NonLinear& nonlinear_;
    fileHandler& file_;

    const double scale_;
    const int slot_count;

    /**
     * @brief  计算 滑动窗口版的多头注意力机制，
     * @details 区别在于多了个参数 head_pos_bias，并且 atten_mask 变成可选
     */
    std::vector<Ciphertext<Scheme::CKKS>> window_attn_ct(
                        std::vector<Ciphertext<Scheme::CKKS>>& input_cts,
                        const int per_input_seq_len, const int hidden_dim, const int num_heads, 
                        const std::vector<std::vector<bool>>& atten_mask,
                        const std::vector<std::vector<double>>& Wq, const std::vector<double>& Bq,
                        const std::vector<std::vector<double>>& Wk, const std::vector<double>& Bk,
                        const std::vector<std::vector<double>>& Wv,const std::vector<double>& Bv,
                        const std::vector<std::vector<std::vector<double>>>& head_pos_bias,
                        const double soft_bias, const int soft_delta_exp, const double soft_inv_scale,
                        std::atomic<long long>* q_time = nullptr, std::atomic<long long>* k_time = nullptr, 
                        std::atomic<long long>* score_time = nullptr, std::atomic<long long>* softmax_time = nullptr, 
                        std::atomic<long long>* v_time = nullptr, std::atomic<long long>* mulV_time = nullptr);
    
    std::vector<std::vector<std::vector<double>>> get_swin_pos_bias(const int windw_size, 
                        const std::vector<std::vector<double>>& pos_bias_table);

    /**
     * @brief  进行窗口分割的函数，即从 input_cts 中分离出要推理的窗口
     * @details 以 [begin_row_ind, begin_col_ind] 为起始坐标，截取 [window_row_len, window_col_len] 大小窗口
     * @details 输入输出的每个密文均以 [per_seq_len × hidden_dim] 紧凑打包
     */
    std::vector<Ciphertext<Scheme::CKKS>> window_partition_cts(
        const std::vector<Ciphertext<Scheme::CKKS>>& input_cts, 
        const int input_row_len, const int input_col_len, const int per_seq_len, const int hidden_dim,
        const int begin_row_ind, const int begin_col_ind, const int window_row_len, const int window_col_len,
        std::atomic<long long>* duration_time = nullptr);

    void window_reverse_cts(const std::vector<Ciphertext<Scheme::CKKS>>& window_cts, 
        const int input_row_len, const int input_col_len, const int per_seq_len, const int hidden_dim,
        std::vector<Ciphertext<Scheme::CKKS>>& output_cts, std::vector<bool>& is_output_visited,
        const int begin_row_ind, const int begin_col_ind, const int window_row_len, const int window_col_len, 
        std::atomic<long long>* duration_time = nullptr);

    std::vector<std::vector<std::vector<double>>> get_part_pos_bias(
        const std::vector<std::vector<std::vector<double>>>& heads_pos_bias, const int window_row, const int window_col,
        const int begin_row_ind, const int begin_col_ind, const int end_row_ind, const int end_col_ind, const int paded_size);

    std::vector<std::vector<std::vector<double>>> window_partition_pt(std::vector<std::vector<double>> input, 
        const int row_len, const int col_len, const int hidden_dim, const int window_size);

    std::vector<std::vector<double>> window_reverse_pt(
        std::vector<std::vector<std::vector<double>>> input_windows, 
        const int row_len, const int col_len, const int hidden_dim, const int window_size);

    std::vector<std::vector<std::vector<bool>>> get_attn_mask(const int row_len, const int col_len, 
        const int window_size, const int shift_size);
};
