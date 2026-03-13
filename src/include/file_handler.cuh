#pragma once

#include <heongpu.cuh>
#include <ckks_evaluator.cuh>
#include <vector>
#include <string>
#include <sstream>
#include <stdexcept>

using namespace heongpu;

class fileHandler {
public:
    fileHandler(
        HEContext<Scheme::CKKS>& ctx, HEEncoder<Scheme::CKKS>& encoder,
        CKKSEvaluator& evaluator, double scale, 
        std::string file_dir = "../data/")
        : ctx_(ctx), encoder_(encoder), evaluator_(evaluator), scale_(scale),
          file_dir(file_dir), slot_count(ctx_.get_poly_modulus_degree() / 2)
          {}
                            
    void writeMat(const std::vector<std::vector<double>>& mat, const std::string& filename);
    
    std::vector<std::vector<double>> readMat(const std::string& filename);

    std::vector<double> readVec(const std::string& filename);

    std::vector<std::vector<bool>> readMask(const std::string& filename);

    void writeMask(const std::vector<std::vector<bool>>& mat, const std::string& filename);

    /**
     * @brief  实现 密文矩阵解密并保存为 csv 文件
     * @details 解密后直接按照 [seq_len, hidden_num] 尺寸写入文件
     */
    void saveCtMat(Ciphertext<Scheme::CKKS>& ct, const std::string file_name,
        const int seq_len, const int hidden_num);

    /**
     * @brief  实现 密文矩阵解密并保存为 csv 文件，上个函数的多密文版本，按行拼接
     * @details 按照 [seq_len * ct_num, hidden_num] 尺寸写入文件
     */
    void saveCtsMat(std::vector<Ciphertext<Scheme::CKKS>>& cts, const std::string file_name,
        const int seq_len, const int hidden_num);
    
    /**
     * @brief  实现 注意力密文矩阵解密并保存为 csv 文件
     * @details 解密后直接按照 [seq_len * head_num, seq_len] 尺寸写入文件
     *          此处适配的是 linear 中乘法的存储方式，即 每 head_size 中前 seq_len 为有效数
     */
    void saveCtAttenScore(Ciphertext<Scheme::CKKS>& ct, const std::string file_name,
        const int seq_len, const int head_num, const int head_size);

    /**
     * @brief  实现 注意力密文矩阵解密并保存为 csv 文件
     * @details 与上述不同的是，Atten_Score 中一行将多个密文拼接，在Softmax中做验证
     */
    void saveCtAttenScore(std::vector<Ciphertext<Scheme::CKKS>>& cts, const std::string file_name,
        const int seq_len, const int head_num, const int head_size);
        

private:
    HEContext<Scheme::CKKS>& ctx_;
    HEEncoder<Scheme::CKKS>& encoder_;
    CKKSEvaluator& evaluator_;

    const double scale_;
    const int slot_count;

    const std::string file_dir;
    bool isFileExist(const std::string& filename) const {
        std::ifstream file_check(filename, std::ios::in);
        return file_check.is_open();
    }
};
