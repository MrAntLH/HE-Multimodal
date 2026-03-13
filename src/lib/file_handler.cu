#include "file_handler.cuh"

void fileHandler::saveCtMat(Ciphertext<Scheme::CKKS>& ct, const std::string file_name,
    const int seq_len, const int hidden_num){
    if (seq_len * hidden_num > slot_count) {
        throw std::invalid_argument("[saveCtMat] error: 输入尺寸 > slot_count ");
    }
    std::vector<double> vec = evaluator_.decrypt(ct);
    std::vector<std::vector<double>> matrix(seq_len, std::vector<double>(hidden_num));
    for (int i = 0; i < seq_len; ++i) {  // 遍历每一行
        std::vector<double>& current_row = matrix[i];
        for (int j = 0; j < hidden_num; ++j) {
            const int vec_idx = i * hidden_num + j;
            current_row[j] = vec[vec_idx];
        }
    }
    writeMat(matrix, file_name);
}

void fileHandler::saveCtsMat(std::vector<Ciphertext<Scheme::CKKS>>& cts, const std::string file_name,
    const int seq_len, const int hidden_num){
    if (seq_len * hidden_num > slot_count) {
        throw std::invalid_argument("[saveCtMat] error: 输入尺寸 > slot_count ");
    }
    const int ct_num = cts.size();
    std::vector<std::vector<double>> matrix;
    matrix.reserve(ct_num * seq_len);
    for(int ct_ind = 0; ct_ind < ct_num; ++ct_ind){
        std::vector<double> vec = evaluator_.decrypt(cts[ct_ind]);
        for(int seq_ind = 0; seq_ind < seq_len; ++seq_ind){
            int idx = seq_ind * hidden_num;
            matrix.emplace_back(vec.begin() + idx, vec.begin() + idx + hidden_num);
        }
    }
    writeMat(matrix, file_name);
}

void fileHandler::saveCtAttenScore(Ciphertext<Scheme::CKKS>& ct, const std::string file_name,
    const int per_ct_seq, const int head_num, const int head_size){
    const int hidden_num = head_num * head_size;
    if (per_ct_seq * hidden_num > slot_count) {
        throw std::invalid_argument("[saveCtAttenScore] error: 输入尺寸 > slot_count ");
    }
    std::vector<double> vec = evaluator_.decrypt(ct);
    std::vector<std::vector<double>> matrix;
    std::vector<double> zero_vec(per_ct_seq, 0.0);
    for(int head_ind = 0; head_ind < head_num; head_ind++){
        for(int seq_ind = 0; seq_ind < per_ct_seq; seq_ind++){
            std::vector<double> one_line(per_ct_seq);
            for(int temp = 0; temp < per_ct_seq; temp++){
                one_line[temp] = vec[head_ind * head_size + seq_ind * hidden_num + temp];
            }
            matrix.emplace_back(one_line);
        }
        matrix.emplace_back(zero_vec);
    }
    writeMat(matrix, file_name);
}

void fileHandler::saveCtAttenScore(std::vector<Ciphertext<Scheme::CKKS>>& cts, const std::string file_name,
    const int seq_len, const int head_num, const int head_size){
    const int hidden_num = head_num * head_size;
    const int ct_num = cts.size();
    const int per_ct_seq = seq_len / ct_num;

    if (per_ct_seq * hidden_num > slot_count) {
        throw std::invalid_argument("[saveCtAttenScore] error: 输入尺寸 > slot_count ");
    }
    std::vector<std::vector<double>> decrypt_mat(ct_num);
    for(int ct_ind = 0; ct_ind < ct_num; ct_ind++){
        decrypt_mat[ct_ind] = evaluator_.decrypt(cts[ct_ind]);
    }

    std::vector<std::vector<double>> output_mat;
    std::vector<double> zero_vec(seq_len, 0.0);
    for(int head_ind = 0; head_ind < head_num; head_ind++){
        for(int seq_ind = 0; seq_ind < per_ct_seq; seq_ind++){
            std::vector<double> one_line(seq_len);
            for(int ct_ind = 0; ct_ind < ct_num; ct_ind++){
                for(int temp = 0; temp < per_ct_seq; temp++){
                    int idx = head_ind * head_size + seq_ind * hidden_num + temp;
                    one_line[ct_ind * per_ct_seq + temp] = decrypt_mat[ct_ind][idx];
                }
            }
            output_mat.emplace_back(one_line);
        }
        output_mat.emplace_back(zero_vec);
    }
    writeMat(output_mat, file_name);
}

void fileHandler::writeMat(const std::vector<std::vector<double>>& matrix, 
    const std::string& filename){
    const std::string fullfilename = file_dir + filename + ".csv";
    std::ofstream csv_file(fullfilename, std::ios::out | std::ios::trunc);
    // 检查文件是否成功打开（即使创建文件，也可能因权限问题失败）
    if (!csv_file.is_open()) {
        std::cerr << "错误：无法打开/创建文件 " << fullfilename << " 进行写入！" << std::endl;
    }
    // 逐行逐列写入矩阵数据
    for (size_t i = 0; i < matrix.size(); ++i) {
        for (size_t j = 0; j < matrix[i].size(); ++j) {
            csv_file << matrix[i][j];
            if (j != matrix[i].size() - 1) {
                csv_file << ",";
            }
        }
        csv_file << std::endl;
    }
    csv_file.close();
}

void fileHandler::writeMask(const std::vector<std::vector<bool>>& matrix, const std::string& filename){
    std::vector<std::vector<double>> double_mat;
    double_mat.reserve(matrix.size());
    for (const auto& row : matrix) {
        std::vector<double> d_row;
        d_row.reserve(row.size());
        for (bool b : row) d_row.push_back(b ? 1.0 : 0.0);
        double_mat.push_back(std::move(d_row)); 
    }
    writeMat(double_mat, filename);
}
    
std::vector<std::vector<double>> fileHandler::readMat(const std::string& filename){
    const std::string fullfilename = file_dir + filename + ".csv";
    if (!isFileExist(fullfilename)) {
        std::cerr << "错误：文件 " << fullfilename << " 不存在，无法读取！" << std::endl;
    }
    std::ifstream csv_file(fullfilename, std::ios::in);
    if (!csv_file.is_open()) {
        std::cerr << "错误：无法打开文件 " << fullfilename << " 进行读取！" << std::endl;
    }
    std::vector<std::vector<double>> matrix_out;
    std::string line;  // 存储每行的字符串
    while (std::getline(csv_file, line)) {
        std::vector<double> row_data; 
        std::stringstream ss(line);
        std::string cell_str;

        while (std::getline(ss, cell_str, ',')) {
            try {

                std::stringstream cell_ss(cell_str);
                double cell_value;
                cell_ss >> cell_value;
                if (cell_ss.fail()) 
                    throw std::invalid_argument("单元格数据转换失败");
                row_data.push_back(cell_value);
            } catch (const std::exception& e) {
                std::cerr << "警告：单元格数据 \"" << cell_str << "\" 转换失败，跳过该单元格。错误信息：" << e.what() << std::endl;
                continue;
            }
        }
        if (!row_data.empty()) {
            matrix_out.push_back(row_data);
        }
    }
    csv_file.close();

    return matrix_out;
}

std::vector<double> fileHandler::readVec(const std::string& filename){
    const std::string fullfilename = file_dir + filename + ".csv";
    if (!isFileExist(fullfilename)) {
        std::cerr << "错误：文件 " << fullfilename << " 不存在，无法读取！" << std::endl;
        return {};
    }
    std::ifstream csv_file(fullfilename, std::ios::in);
    if (!csv_file.is_open()) {
        std::cerr << "错误：无法打开文件 " << fullfilename << " 进行读取！" << std::endl;
        return {}; 
    }
    std::vector<double> vec_out;
    std::string line;
    if (std::getline(csv_file, line)) {
        std::stringstream ss(line);
        std::string cell_str;
        while (std::getline(ss, cell_str, ',')) {
            try {
                std::stringstream cell_ss(cell_str);
                double cell_value;
                cell_ss >> cell_value;
                if (cell_ss.fail()) 
                    throw std::invalid_argument("单元格数据转换失败");
                vec_out.push_back(cell_value);
            } catch (const std::exception& e) {
                std::cerr << "警告：单元格数据 \"" << cell_str << "\" 转换失败，跳过该单元格。错误信息：" << e.what() << std::endl;
                continue;
            }
        }
    } else {
        std::cerr << "警告：文件 " << fullfilename << " 为空，无有效数据可读取！" << std::endl;
    }
    csv_file.close();
    return vec_out;
}

std::vector<std::vector<bool>> fileHandler::readMask(const std::string& filename){
    const std::string fullfilename = file_dir + filename + ".csv";
    if (!isFileExist(fullfilename)) {
        std::cerr << "错误：文件 " << fullfilename << " 不存在，无法读取！" << std::endl;
    }
    std::ifstream csv_file(fullfilename, std::ios::in);
    if (!csv_file.is_open()) {
        std::cerr << "错误：无法打开文件 " << fullfilename << " 进行读取！" << std::endl;
    }
    std::vector<std::vector<bool>> mask_out;
    const double THRESHOLD = -50;
    std::string line;
    while (std::getline(csv_file, line)) {
        std::vector<bool> row_data;
        std::stringstream ss(line);
        std::string cell_str;
        while (std::getline(ss, cell_str, ',')) {
            try {
                std::stringstream cell_ss(cell_str);
                double cell_value;
                cell_ss >> cell_value;
                if (cell_ss.fail()) 
                    throw std::invalid_argument("单元格数据转换失败");
                row_data.push_back(cell_value >= THRESHOLD);
            } catch (const std::exception& e) {
                std::cerr << "警告：单元格数据 \"" << cell_str << "\" 转换失败，跳过该单元格。错误信息：" << e.what() << std::endl;
                continue;
            }
        }
        if (!row_data.empty()) {
            mask_out.push_back(row_data);
        }
    }
    csv_file.close();
    return mask_out;
}
