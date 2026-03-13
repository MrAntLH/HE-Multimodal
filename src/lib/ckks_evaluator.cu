#include <ckks_evaluator.cuh>

void CKKSEvaluator::multiply(Ciphertext<Scheme::CKKS>& ct1, Ciphertext<Scheme::CKKS>& ct2, 
    Ciphertext<Scheme::CKKS>& dest){
    if( ct1.depth() >= max_depth){
        bootstrap(ct1);
    }
    if( ct2.depth() >= max_depth){
        bootstrap(ct2);
    }
    while( ct1.depth() != ct2.depth()){
        if( ct1.depth() > ct2.depth() ){
            operators_.mod_drop_inplace(ct2);
        } else {
            operators_.mod_drop_inplace(ct1);
        }
    }
    operators_.multiply(ct1, ct2, dest);
    if(dest.relinearization_required()){
        operators_.relinearize_inplace(dest, relin_key_);
    }
    if(dest.rescale_required()){
        operators_.rescale_inplace(dest);
    }
}

void CKKSEvaluator::multiply(Ciphertext<Scheme::CKKS>& ct1, Ciphertext<Scheme::CKKS>& ct2){
    if( ct1.depth() >= max_depth){
        bootstrap(ct1);
    }
    if( ct2.depth() >= max_depth){
        bootstrap(ct2);
    }
    while( ct1.depth() != ct2.depth()){
        if( ct1.depth() > ct2.depth() ){
            operators_.mod_drop_inplace(ct2);
        } else {
            operators_.mod_drop_inplace(ct1);
        }
    }
    operators_.multiply_inplace(ct1, ct2);
    if(ct1.relinearization_required()){
        operators_.relinearize_inplace(ct1, relin_key_);
    }
    if(ct1.rescale_required()){
        operators_.rescale_inplace(ct1);
    }
}

void CKKSEvaluator::multiply(Ciphertext<Scheme::CKKS>& ct, Plaintext<Scheme::CKKS>& pt, 
    Ciphertext<Scheme::CKKS>& dest){
    if( ct.depth() >= max_depth ){
        bootstrap(ct);
    }
    while(ct.depth() > pt.depth()){
        operators_.mod_drop_inplace(pt);
    }
    operators_.multiply_plain(ct, pt, dest);
    if(dest.rescale_required()){
        operators_.rescale_inplace(dest);
    }
}

void CKKSEvaluator::multiply(Ciphertext<Scheme::CKKS>& ct, Plaintext<Scheme::CKKS>& pt){  
    if( ct.depth() >= max_depth ){
        bootstrap(ct);
    }
    while(ct.depth() > pt.depth()){
        operators_.mod_drop_inplace(pt);
    }
    operators_.multiply_plain_inplace(ct, pt); 
    if(ct.rescale_required()){
        operators_.rescale_inplace(ct);
    }
}

void CKKSEvaluator::square(Ciphertext<Scheme::CKKS>& ct, Ciphertext<Scheme::CKKS>& dest){
    if( ct.depth() >= max_depth ){
        bootstrap(ct);
    }
    operators_.multiply(ct, ct, dest);
    if(dest.relinearization_required()){
        operators_.relinearize_inplace(dest, relin_key_);
    }
    if(dest.rescale_required()){
        operators_.rescale_inplace(dest);
    }
}

void CKKSEvaluator::square(Ciphertext<Scheme::CKKS>& ct){   
    if( ct.depth() >= max_depth ){
        bootstrap(ct);
    }
    operators_.multiply_inplace(ct, ct);
    if(ct.relinearization_required()){
        operators_.relinearize_inplace(ct, relin_key_);
    }
    if(ct.rescale_required()){
        operators_.rescale_inplace(ct);
    }
}

void CKKSEvaluator::add(Ciphertext<Scheme::CKKS>& ct1, Ciphertext<Scheme::CKKS>& ct2, 
    Ciphertext<Scheme::CKKS>& dest){
    while(ct1.depth() != ct2.depth()){
        if( ct1.depth() > ct2.depth() ){
            operators_.mod_drop_inplace(ct2);
        } else {
            operators_.mod_drop_inplace(ct1);
        }
    }
    operators_.add(ct1, ct2, dest);
}

void CKKSEvaluator::add(Ciphertext<Scheme::CKKS>& ct1, Ciphertext<Scheme::CKKS>& ct2){
    while(ct1.depth() != ct2.depth()){
        if( ct1.depth() > ct2.depth() ){
            operators_.mod_drop_inplace(ct2);
        } else {
            operators_.mod_drop_inplace(ct1);
        }
    }
    operators_.add_inplace(ct1, ct2);
}

void CKKSEvaluator::add(Ciphertext<Scheme::CKKS>& ct, Plaintext<Scheme::CKKS>& pt, 
    Ciphertext<Scheme::CKKS>& dest){ 
    while(pt.depth() < ct.depth()){
        operators_.mod_drop_inplace(pt);
    }
    operators_.add_plain(ct, pt, dest);
}

void CKKSEvaluator::add(Ciphertext<Scheme::CKKS>& ct, Plaintext<Scheme::CKKS>& pt){
    while(pt.depth() < ct.depth()){
        operators_.mod_drop_inplace(pt);
    }
    operators_.add_plain_inplace(ct, pt);
}

void CKKSEvaluator::sub(Ciphertext<Scheme::CKKS>& ct1, Ciphertext<Scheme::CKKS>& ct2, 
    Ciphertext<Scheme::CKKS>& dest){   
    while(ct1.depth() != ct2.depth()){
        if( ct1.depth() > ct2.depth() ){
            operators_.mod_drop_inplace(ct2);
        } else {
            operators_.mod_drop_inplace(ct1);
        }
    }
    operators_.sub(ct1, ct2, dest);
}

void CKKSEvaluator::sub(Ciphertext<Scheme::CKKS>& ct1, Ciphertext<Scheme::CKKS>& ct2){
    while(ct1.depth() != ct2.depth()){
        if( ct1.depth() > ct2.depth() ){
            operators_.mod_drop_inplace(ct2);
        } else {
            operators_.mod_drop_inplace(ct1);
        }
    }
    operators_.sub_inplace(ct1, ct2);
}

void CKKSEvaluator::sub(Ciphertext<Scheme::CKKS>& ct, Plaintext<Scheme::CKKS>& pt, 
    Ciphertext<Scheme::CKKS>& dest){
    while(pt.depth() < ct.depth()){
        operators_.mod_drop_inplace(pt);
    }
    operators_.sub_plain(ct, pt, dest);
}

void CKKSEvaluator::sub(Ciphertext<Scheme::CKKS>& ct, Plaintext<Scheme::CKKS>& pt){
    while(pt.depth() < ct.depth()){
        operators_.mod_drop_inplace(pt);
    }
    operators_.sub_plain_inplace(ct, pt);
}

void CKKSEvaluator::negate(Ciphertext<Scheme::CKKS>& ct, Ciphertext<Scheme::CKKS>& dest){
    operators_.negate(ct, dest);
}

void CKKSEvaluator::negate(Ciphertext<Scheme::CKKS>& ct){
    operators_.negate_inplace(ct);
}

void CKKSEvaluator::rotate_rows(Ciphertext<Scheme::CKKS>& ct, int steps, 
    Ciphertext<Scheme::CKKS>& dest){
    steps = (steps + slot_count) % slot_count;
    if (steps == 0) return;

    int temp = 1;
    while (temp <= steps / 2)
        temp *= 2;
    operators_.rotate_rows(ct, dest, galois_key_, temp);
    steps -= temp;

    while (steps > 0) {
        temp = 1;
        while (temp <= steps / 2)
            temp *= 2;
        operators_.rotate_rows_inplace(dest, galois_key_, temp);
        steps -= temp;
    }
}

void CKKSEvaluator::rotate_rows(Ciphertext<Scheme::CKKS>& ct, int steps){
    steps = (steps+slot_count) % slot_count;
    while(steps > 0){
        int temp = 1;
        while(temp <= steps/2)
            temp *= 2;
        operators_.rotate_rows_inplace(ct, galois_key_, temp);
        steps -= temp;
    }
}

void CKKSEvaluator::encrypt(Plaintext<Scheme::CKKS>& pt,  Ciphertext<Scheme::CKKS>& ct){
    encryptor_.encrypt(ct, pt);
}

void CKKSEvaluator::bootstrap(Ciphertext<Scheme::CKKS>& ct, double scale){
    ct = operators_.slim_bootstrapping(ct, galois_key_, relin_key_);
    /*
    Plaintext<Scheme::CKKS> scale_pt;
    encoder_.encode(scale_pt, std::vector<double>(slot_count, scale), scale_);
    multiply(ct, scale_pt);
    ct = operators_.slim_bootstrapping(ct, galois_key_, relin_key_);
    encoder_.encode(scale_pt, std::vector<double>(slot_count, 1/scale), scale_);
    multiply(ct, scale_pt);
    */
}

// ------------------- calculate_MAE (decrypt & decode) -------------------
double CKKSEvaluator::calculate_MAE(std::vector<double> true_vec, Ciphertext<Scheme::CKKS> ct) {
    Plaintext<Scheme::CKKS> plain(ctx_);
    decryptor_.decrypt(plain, ct);
    std::vector<double> y_vec;
    encoder_.decode(y_vec, plain);
    int N = true_vec.size();
    double sum_abs = 0.0;
    for (int i = 0; i < N; ++i) {
        sum_abs += std::abs(true_vec[i] - y_vec[i]);
    }
    return sum_abs / N;
}

double CKKSEvaluator::calculate_MAE(std::vector<double> true_vec, std::vector<double> predict_vec) {
    int N = true_vec.size();
    double sum_abs = 0.0;
    for (int i = 0; i < N; ++i) {
        sum_abs += std::abs(true_vec[i] - predict_vec[i]);
    }
    return sum_abs / N;
}

double CKKSEvaluator::calculate_MAE(std::vector<std::vector<double>> true_mat, std::vector<std::vector<double>> predict_mat){
    if(true_mat.size() != predict_mat.size() || true_mat[0].size() != predict_mat[0].size()){
        throw std::invalid_argument("calculate_MAE error: 输入矩阵维度不匹配");
    }
    int rows = true_mat.size();
    int cols = true_mat[0].size();
    double sum_abs = 0.0;
    for(int r = 0; r < rows; r++){
        for(int c = 0; c < cols; c++){
            sum_abs += std::abs(true_mat[r][c] - predict_mat[r][c]);
        }
    }
    return sum_abs / (rows * cols);
}

void CKKSEvaluator::print_vector(std::vector<double> vec, std::string message, int print_size, int precision) {
    int N = vec.size();
    std::cout << message << "[ ";
    std::cout << std::fixed << std::setprecision(precision);
    if (N <= 2 * print_size) {
        for (double v : vec) std::cout << v << " ";
    } else {
        for (int i = 0; i < print_size; ++i) std::cout << vec[i] << " ";
        std::cout << "... ";
        for (int i = N - print_size; i < N; ++i) std::cout << vec[i] << " ";
    }
    std::cout << "]" << std::endl;
    std::cout.unsetf(std::ios::fixed);
}

void CKKSEvaluator::print_matrix(std::vector<double>& matrix, std::string message, int rows, int cols, 
    int print_row_nums, int print_col_nums, int precision) {
    std::vector<std::vector<double>> mat_2d(rows, std::vector<double>(cols, 0.0));
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            mat_2d[r][c] = matrix[r * cols + c];
        }
    }
    print_matrix(mat_2d, message, rows, cols, print_row_nums, print_col_nums, precision);
}

void CKKSEvaluator::print_matrix(std::vector<std::vector<double>>& matrix, 
    std::string message, int rows, int cols, int print_row_nums, int print_col_nums, int precision) {
    if(matrix.empty() || matrix[0].empty()) {
        throw std::invalid_argument("print_matrix error: 输入矩阵不能为空");
    }
    if(rows > matrix.size() || cols > matrix[0].size()) {
        throw std::invalid_argument("print_matrix error: 打印的行列数与大于实际维度");
    }
    std::cout << message << std::endl;
    std::cout << std::fixed << std::setprecision(precision);
    
    const int actual_print_rows = std::min(print_row_nums, rows);
    const int actual_print_cols = std::min(print_col_nums, cols);
    const int first_row_part = actual_print_rows / 2;
    const int first_col_part = actual_print_cols / 2;
    const int second_row_start = rows - (actual_print_rows - first_row_part);
    const int second_col_start = cols - (actual_print_cols - first_col_part);

    for(size_t r = 0; r < actual_print_rows; r++){
        if(r < first_row_part){
            for(size_t c = 0; c < actual_print_cols; c++){
                if(c < first_col_part){
                    std::cout << matrix[r][c] << "\t";
                }else{
                    if(actual_print_cols < cols && c == first_col_part){
                        std::cout << "...\t";
                    }
                    size_t orig_col = second_col_start + (c - first_col_part);
                    std::cout << matrix[r][orig_col] << "\t";
                }
            }
            std::cout << std::endl;
        }else{
            if(actual_print_rows < rows && r == first_row_part){
                std::cout << "...\t..." << std::endl;
            }
            size_t orig_row = second_row_start + (r - first_row_part);
            for(size_t c = 0; c < actual_print_cols; c++){
                if(c < first_col_part){
                    std::cout << matrix[orig_row][c] << "\t";
                }else{
                    if(actual_print_cols < cols && c == first_col_part){
                        std::cout << "...\t";
                    }
                    size_t orig_col = second_col_start + (c - first_col_part);
                    std::cout << matrix[orig_row][orig_col] << "\t";
                }
            }
            std::cout << std::endl;
        }
    }
    std::cout.unsetf(std::ios::fixed);
}

void CKKSEvaluator::print_matrix(std::vector<std::vector<bool>>& matrix, 
    std::string message, int rows, int cols, int print_row_nums, int print_col_nums) {
    std::vector<std::vector<double>> double_matrix;
    for (auto& row : matrix) {
        std::vector<double> d_row;
        for (bool val : row) d_row.push_back(val ? 1.0 : 0.0);
        double_matrix.push_back(d_row);
    }
    print_matrix(double_matrix, message, rows, cols, print_row_nums, print_col_nums, 0);
}

void CKKSEvaluator::print_cipher(Ciphertext<Scheme::CKKS> ct, std::string message = "", int print_size) {
    Plaintext<Scheme::CKKS> plain(ctx_);
    decryptor_.decrypt(plain, ct);
    std::vector<double> y_vec;
    encoder_.decode(y_vec, plain);
    print_vector(y_vec, message, print_size);
}

void CKKSEvaluator::print_cipher_matrix(Ciphertext<Scheme::CKKS> ct, std::string message, 
    int rows, int cols, int print_row_nums, int print_col_nums, int precision){
    Plaintext<Scheme::CKKS> plain(ctx_);
    decryptor_.decrypt(plain, ct);
    std::vector<double> y_vec;
    encoder_.decode(y_vec, plain);
    print_matrix(y_vec, message, rows, cols, print_row_nums, print_col_nums, precision);
}

std::vector<double> CKKSEvaluator::decrypt(Ciphertext<Scheme::CKKS>& ct){
    Plaintext<Scheme::CKKS> plain(ctx_);
    decryptor_.decrypt(plain, ct);
    std::vector<double> vec;
    encoder_.decode(vec, plain);
    return vec;
}
