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

using namespace heongpu;

class NonLinear {
public:
    NonLinear(
        HEContext<Scheme::CKKS>& ctx, HEEncoder<Scheme::CKKS>& encoder,
        CKKSEvaluator& evaluator, fileHandler& file, double scale)
        : ctx_(ctx), encoder_(encoder), evaluator_(evaluator), file_(file), scale_(scale),
          slot_count(ctx_.get_poly_modulus_degree() / 2)
          {}
    /**
     * @brief  进行 Softmax 近似，特征维度为多个密文间接拼接存储的形式
     * @details 输入密文中，共 per_seq_len * head_num * head_size 个连续元素
     *          由于是 Q@K.T 计算后的结果，故每个头 head_size 中仅有前 per_seq_len 个元素有效
     *          受分块策略影响，每个密文代表 per_seq_len 行，共 seq_len / per_seq_len 个密文
     * @param  delta_exp      为二次缩小参数，即 e^(x / (2^delta_exp)) 近似
     * @param  atten_mask     [per_seq_len, seq_len] 尺寸的一个掩码，用 0和1 表示
     * @return  近似后的密文序列，形式同输入
     */
    std::vector<Ciphertext<Scheme::CKKS>> Softmax(std::vector<Ciphertext<Scheme::CKKS>>& input_cts,
        const std::vector<std::vector<bool>>& atten_mask, 
        int seq_len, int per_seq_len, int head_num, int head_size, 
        const double& soft_bias, const int& soft_delta_exp, const double& soft_inv_scale);

    /**
     * @brief  单密文版本，即特征维度就是 per_seq_len
     */
    Ciphertext<Scheme::CKKS> Softmax(Ciphertext<Scheme::CKKS>& input_ct,
        const std::vector<std::vector<bool>>& atten_mask,
        const int per_seq_len, const int head_num, const int head_size, 
        const double& soft_bias, const int& soft_delta_exp, const double& soft_inv_scale);
    

private:
    HEContext<Scheme::CKKS>& ctx_;
    HEEncoder<Scheme::CKKS>& encoder_;
    CKKSEvaluator& evaluator_;
    fileHandler& file_;

    const double scale_;
    const int slot_count;

    /**
     * @brief  对单个密文进行指数函数近似，采用最小二乘法，此处实现为：y = e^(4x) in [-1,1] 的拟合
     * @details 此处不区分有效数字，同步对密文的所有槽进行近似
     */
    Ciphertext<Scheme::CKKS> exp_ct(Ciphertext<Scheme::CKKS>& x);

    /**
     * @brief  对单个密文进行多项式运算，多项式系数由外部传入
     * @details require degree in [16, 32], i.e. a.size() > 16 and (a.size()-1) <= 32
     */
    void eval_poly(const std::vector<double> &a, Ciphertext<Scheme::CKKS>& x, 
        Ciphertext<Scheme::CKKS>& dest);
    
    /**
     * @brief  对密文内各块同步进行求和，即对每个头 head_size 中前 per_ct_seq_len 个元素求和
     * @details 为适配后续对块内所有元素/sum(e^x)的操作，故此处要将求和结果重复到 per_ct_seq_len 个
     * @details 为提高效率，此处利用 块内后面 per_ct_seq_len 个闲置槽位进行旋转预存
     * @details Note:此处实现不适用 2 * per_ct_seq_len > head_size 情况，如有需求请补充降级策略
     * @example 举例：per_ct_seq_len = 2, head_num = 2, head_size = 4
     *          input : [1  2  0  0  3  4  0  0  5  6 0 0  7  8 0 0]
     *          output: [3  3  0  0  7  7  0  0 11 11 0 0 15 15 0 0]
     */
    Ciphertext<Scheme::CKKS> ct_block_sum(Ciphertext<Scheme::CKKS>& input_ct, 
        int per_ct_seq_len, int head_num, int head_size);

    /**
     * @brief  对单个密文进行逆函数近似，基于无穷乘积收敛，此处实现为：y = 1/x 的拟合
     * @details 受方案限制，此处需限制输入的范围为(0,2), 越靠近 1 效果越好
     * @details 此处不区分有效数字，同步对密文的所有槽进行近似
     * @param  iter      迭代次数，默认4次
     */
    Ciphertext<Scheme::CKKS> inverse_ct(Ciphertext<Scheme::CKKS> x, int iter = 4);

};
