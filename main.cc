#include <vector>
#include <cstring>
#include <string>
#include <iostream>
#include <fstream>
#include <set>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <sys/time.h>
#include <omp.h>
#include "hnswlib/hnswlib/hnswlib.h"
#include "flat_scan.h"
#include <queue>
#include <arm_neon.h> 
// 可以自行添加需要的头文件

using namespace hnswlib;
using namespace std;




struct simd8float32 {
    float32x4x2_t data;

    // 默认构造函数 
    simd8float32() = default;

    // 显式构造函数，从浮点数数组初始化 
    explicit simd8float32(const float* x){
        data.val[0]  = vld1q_f32(x);     // 假设调用方已对齐数据 
        data.val[1]  = vld1q_f32(x + 4);
    }

    // 显式构造函数，初始化为单个浮点数 
    explicit simd8float32(float value) {
        float32x4_t val = vdupq_n_f32(value);
        data = { val, val };
    }

    // 乘法运算符重载 
    simd8float32 operator *(const simd8float32& other) const {
        simd8float32 result;
        result.data.val[0] = vmulq_f32(data.val[0], other.data.val[0]);
        result.data.val[1] = vmulq_f32(data.val[1], other.data.val[1]);
        return result;
    }

    // 减法运算符重载 
    simd8float32 operator -(const simd8float32& other) const {
        simd8float32 result;
        result.data.val[0] = vsubq_f32(data.val[0], other.data.val[0]);
        result.data.val[1] = vsubq_f32(data.val[1], other.data.val[1]);
        return result;
    }

    // 加法运算符重载 
    simd8float32 operator + (const simd8float32& other) const {
        simd8float32 result;
        result.data.val[0] = vaddq_f32(data.val[0], other.data.val[0]);
        result.data.val[1] = vaddq_f32(data.val[1], other.data.val[1]);
        return result;
    }

    // 将数据存储到浮点数数组中 
    void storeu(float* ptr) {
        vst1q_f32(ptr, data.val[0]);
        vst1q_f32(ptr + 4, data.val[1]);
    }
};

float InnerProductSIMDNeon(float *b1, float *b2, size_t vecdim) {
    assert(vecdim % 32 == 0); // 调整为32的倍数以支持4次循环展开 
 
    // 使用4个累加器分散依赖 
    simd8float32 sum1(0.0f), sum2(0.0f), sum3(0.0f), sum4(0.0f);
    
    // 每次迭代处理32个元素（4x8）
    for (int i = 0; i < vecdim; i += 32) {
        // 循环展开4次 
        simd8float32 s1_1(b1 + i), s2_1(b2 + i);
        simd8float32 s1_2(b1 + i + 8), s2_2(b2 + i + 8);
        simd8float32 s1_3(b1 + i + 16), s2_3(b2 + i + 16);
        simd8float32 s1_4(b1 + i + 24), s2_4(b2 + i + 24);
 
        sum1 = sum1 + (s1_1 * s2_1);
        sum2 = sum2 + (s1_2 * s2_2);
        sum3 = sum3 + (s1_3 * s2_3);
        sum4 = sum4 + (s1_4 * s2_4);
    }
 
    // 合并累加器 
    simd8float32 sum_total = sum1 + sum2 + sum3 + sum4;
    
    float tmp[8];
    sum_total.storeu(tmp); 
    
    // 水平求和优化（使用NEON加速）
    float32x4_t vsum = vaddq_f32(vld1q_f32(tmp), vld1q_f32(tmp + 4));
    float32x2_t vsum_h = vadd_f32(vget_low_f32(vsum), vget_high_f32(vsum));
    float dis = vget_lane_f32(vpadd_f32(vsum_h, vsum_h), 0);
    
    return 1 - dis;
}

// 修改后的flat_search函数，更名为optimized_flat_search，使用SIMD优化内积运算
std::priority_queue<std::pair<float, uint32_t>> optimized_flat_search(float* base, float* query, size_t n, size_t dim, size_t k) {
    assert(dim % 32 == 0); // 确保维度是32的倍数以支持SIMD优化 
    std::priority_queue<std::pair<float, uint32_t>> q;
    
    #pragma omp parallel for 
    for(int i = 0; i < n; ++i) {
        float dis = InnerProductSIMDNeon(base + i*dim, query, dim);
        
        // 线程安全插入（需互斥锁或原子操作）
        #pragma omp critical 
        {
            if(q.size()  < k || dis < q.top().first)  {
                q.push({dis,  i});
                if(q.size()  > k) q.pop(); 
            }
        }
    }
    return q;
}



// PQEncodedData结构体，存储编码后的数据
struct PQEncodedData {
    uint32_t codes;  // 4x8-bit索引 
    uint32_t id;     // 原始向量ID 
};

// 256个8-bit索引，4个子空间并行计算
struct alignas(32) PQCodebook {
    int m_subspaces;  // 动态子空间数 
    int sub_dim;      // 每个子维度 
    std::vector<simd8float32> centroids; // 替代固定数组 

    void initialize(int m_subspaces, int sub_dim) {
        this->m_subspaces = m_subspaces;
        this->sub_dim = sub_dim;
        centroids.resize(m_subspaces * 256); // 初始化centroids大小
    }

    // 查找最近的质心索引
    // 使用SIMD优化
    uint8_t find_nearest_centroid(const PQCodebook & cb, const float* sub_vec, int m) {
        simd8float32 query(sub_vec);
        float32x4_t min_dist = vdupq_n_f32(std::numeric_limits<float>::max());
        uint8_t best_idx = 0;
        for (int k = 0; k < 256; ++k) {
            simd8float32 diff = query - cb.centroids[m * 256 + k];
            // 计算平方距离
            simd8float32 dist = diff * diff;
            //水平求和
            float32x4_t sum = vaddq_f32(dist.data.val[0], dist.data.val[1]);
            float32x2_t sum_low_high = vadd_f32(vget_low_f32(sum), vget_high_f32(sum));
            float total = vget_lane_f32(vpadd_f32(sum_low_high, sum_low_high), 0);
            // 更新最小距离和索引
            if (total < vgetq_lane_f32(min_dist, 0) || k == 0) {
                min_dist = vdupq_n_f32(total);
                best_idx = k;
            }
        }
        return best_idx;
    }

    // 加载子空间数据并训练质心
    void load_subspace(int m, const float* data, int sub_dim, size_t base_number) {
        // 使用k-means训练质心
        std::vector<std::vector<float>> clusters(256, std::vector<float>(sub_dim, 0.0f));
        std::vector<int> cluster_counts(256, 0);

        // 初始化质心为前256个向量
        for (int k = 0; k < 256; ++k) {
            centroids[m * 256 + k] = simd8float32(data + k * sub_dim);
        }

        // k-means迭代
        for (int iter = 0; iter < 10; ++iter) {
            std::fill(clusters.begin(), clusters.end(), std::vector<float>(sub_dim, 0.0f));
            std::fill(cluster_counts.begin(), cluster_counts.end(), 0);

            // 分配样本到最近质心
            for (size_t i = 0; i < base_number; ++i) {
                const float* vec = data + i * sub_dim;
                uint8_t nearest = find_nearest_centroid(*this, vec, m);
                for (int d = 0; d < sub_dim; ++d) {
                    clusters[nearest][d] += vec[d];
                }
                cluster_counts[nearest]++;
            }

            // 更新质心
            for (int k = 0; k < 256; ++k) {
                if (cluster_counts[k] > 0) {
                    for (int d = 0; d < sub_dim; ++d) {
                        clusters[k][d] /= cluster_counts[k];
                    }
                    centroids[m * 256 + k] = simd8float32(clusters[k].data());
                }
            }
        }
    }

};



// 训练PQ码本并编码原始数据 
void train_and_encode_pq(
    const float* base_data,
    size_t base_number,
    size_t vecdim,
    PQCodebook & codebook,
    std::vector<PQEncodedData> & encoded_base
) {
    // 划分子空间
    const int m_subspaces = 4;
    const int sub_dim = vecdim / m_subspaces;

    codebook.initialize(m_subspaces,  sub_dim);
    // 对每个子空间进行k-means聚类
    for (int m = 0; m < m_subspaces; ++m) {
        // 提取子空间数据并训练码本 
        codebook.load_subspace(m, base_data + m * sub_dim, sub_dim, base_number);
    }

    // 编码全量数据 
    encoded_base.resize(base_number);
#pragma omp parallel for 
    for (size_t i = 0; i < base_number; ++i) {
        PQEncodedData data;
        data.id = i; // 保持ID与原始索引一致 
        for (int m = 0; m < m_subspaces; ++m) {
            // 计算子空间最近质心索引（需实现最近邻搜索）
            uint8_t idx = codebook.find_nearest_centroid(codebook, base_data + i * vecdim + m * sub_dim, m);
            data.codes |= (idx << (8 * m));
        }
        encoded_base[i] = data;
    }
}


// 计算PQ距离的NEON优化版本
float pq_distance_neon(const PQCodebook& codebook, 
    uint32_t encoded_vec, 
    simd8float32& query) {
    // 解码4个8-bit索引 
    alignas(16) uint8_t indices[4] = {
        static_cast<uint8_t>(encoded_vec & 0xFF),
        static_cast<uint8_t>((encoded_vec >> 8) & 0xFF),
        static_cast<uint8_t>((encoded_vec >> 16) & 0xFF),
        static_cast<uint8_t>((encoded_vec >> 24) & 0xFF)
    };

    // NEON并行查表（4子空间并行）
    uint8x8_t idx_vec = vld1_u8(indices);
    simd8float32 sum(0.0f);

    for (int m = 0; m < 4; ++m) {
        // 加载码本中心（4x32维）
        simd8float32 centroid = codebook.centroids[m*256  + indices[m]];

        // SIMD内积计算 
        simd8float32 diff = query - centroid;
        sum = sum + (diff * diff);
    }

    // NEON水平求和 
    float tmp[8];
    sum.storeu(tmp); 
    float32x4_t vsum = vaddq_f32(vld1q_f32(tmp), vld1q_f32(tmp + 4));
    return vgetq_lane_f32(vsum, 0) + vgetq_lane_f32(vsum, 1);
}

// 优化后的flat_search函数，使用PQ距离计算，更名为pq_flat_search_optimized
std::priority_queue<std::pair<float, uint32_t>> 
pq_flat_search_optimized(const PQCodebook& codebook,
                        const PQEncodedData* encoded_base,
                        float* query,
                        size_t base_number,
                        size_t vecdim,
                        size_t k) {
    assert(vecdim % codebook.m_subspaces == 0);
    
    // 查询向量预处理（NEON加速）
    simd8float32 qvec;
    qvec = simd8float32(query); // 使用构造函数加载全部128维数据 
  
    std::priority_queue<std::pair<float, uint32_t>> result_queue;
 
    // 主循环处理（OpenMP）
    for (size_t i = 0; i < base_number; ++i) {
        // 调用pq_distance_neon计算距离 
        float distance = pq_distance_neon(
            codebook,
            encoded_base[i].codes, // 传入编码数据 
            qvec                  // 传入预处理后的SIMD查询向量 
        );
        
        // 维护Top-K队列（IP距离转换）
        const float final_score = 1 - distance;
        if (result_queue.size()  < k || final_score > result_queue.top().first)  {
            result_queue.emplace(final_score,  encoded_base[i].id);
            if (result_queue.size()  > k) result_queue.pop(); 
        }
    }
    return result_queue;
}
// 静态缓存索引
HierarchicalNSW<float>* appr_alg = nullptr;

// 修改后的flat_search函数，更名为proved_search，使用HNSW索引
priority_queue<pair<float, uint32_t>> proved_search(const string& index_path, const float* query, size_t vecdim, size_t k) {
    // 加载HNSW索引
    InnerProductSpace ipspace(vecdim);
    //HierarchicalNSW<float>* appr_alg = new HierarchicalNSW<float>(&ipspace, index_path);

    if (appr_alg == nullptr) {
        appr_alg = new HierarchicalNSW<float>(&ipspace, index_path);
    }
    // 执行搜索
    const size_t dynamic_ef = std::max(k * 2, 128UL); // 动态计算候选集大小 
    appr_alg->setEf(dynamic_ef); // 需确认HNSW库是否支持该API 
    auto res = appr_alg->searchKnn(query, k);
  


    // 将结果转换为priority_queue
    priority_queue<pair<float, uint32_t>> result_queue;
    while (!res.empty()) {
        auto item = res.top();
        result_queue.emplace(item.first, item.second);
        res.pop();
    }
    // delete appr_alg; // 释放内存
    return result_queue;
}


template<typename T>
T *LoadData(std::string data_path, size_t& n, size_t& d)
{
    std::ifstream fin;
    fin.open(data_path, std::ios::in | std::ios::binary);
    fin.read((char*)&n,4);
    fin.read((char*)&d,4);
    T* data = new T[n*d];
    int sz = sizeof(T);
    for(int i = 0; i < n; ++i){
        fin.read(((char*)data + i*d*sz), d*sz);
    }
    fin.close();

    std::cerr<<"load data "<<data_path<<"\n";
    std::cerr<<"dimension: "<<d<<"  number:"<<n<<"  size_per_element:"<<sizeof(T)<<"\n";

    return data;
}

struct SearchResult
{
    float recall;
    int64_t latency; // 单位us
};

void build_index(float* base, size_t base_number, size_t vecdim)
{
    const int efConstruction = 200; // 为防止索引构建时间过长，efc建议设置200以下
    const int M = 16; // M建议设置为16以下

    HierarchicalNSW<float> *appr_alg;
    InnerProductSpace ipspace(vecdim);
    appr_alg = new HierarchicalNSW<float>(&ipspace, base_number, M, efConstruction);

    appr_alg->addPoint(base, 0);
    #pragma omp parallel for
    for(int i = 1; i < base_number; ++i) {
        appr_alg->addPoint(base + 1ll*vecdim*i, i);
    }

    char path_index[1024] = "files/hnsw.index";
    appr_alg->saveIndex(path_index);
}


int main(int argc, char *argv[])
{
    size_t test_number = 0, base_number = 0;
    size_t test_gt_d = 0, vecdim = 0;

    std::string data_path = "/anndata/"; 
    auto test_query = LoadData<float>(data_path + "DEEP100K.query.fbin", test_number, vecdim);
    auto test_gt = LoadData<int>(data_path + "DEEP100K.gt.query.100k.top100.bin", test_number, test_gt_d);
    auto base = LoadData<float>(data_path + "DEEP100K.base.100k.fbin", base_number, vecdim);
    // 只测试前2000条查询
    test_number = 2000;

    const size_t k = 10;

    std::vector<SearchResult> results;
    results.resize(test_number);

    // 如果你需要保存索引，可以在这里添加你需要的函数，你可以将下面的注释删除来查看pbs是否将build.index返回到你的files目录中
    // 要保存的目录必须是files/*
    // 每个人的目录空间有限，不需要的索引请及时删除，避免占空间太大
    // 不建议在正式测试查询时同时构建索引，否则性能波动会较大
    // 下面是一个构建hnsw索引的示例
    //build_index(base, base_number, vecdim);

    
    // 查询测试代码

    // PQ编码预处理
    PQCodebook codebook;
    std::vector<PQEncodedData> encoded_base;
    train_and_encode_pq(base, base_number, vecdim, codebook, encoded_base);

    for(int i = 0; i < test_number; ++i) {
        const unsigned long Converter = 1000 * 1000;
        struct timeval val;
        int ret = gettimeofday(&val, NULL);

        // 该文件已有代码中你只能修改该函数的调用方式
        // 可以任意修改函数名，函数参数或者改为调用成员函数，但是不能修改函数返回值。
        //auto res = flat_search(base, test_query + i*vecdim, base_number, vecdim, k);
        auto res = proved_search("files/hnsw.index", test_query + i*vecdim, vecdim, k);
        //auto res = optimized_flat_search(base, test_query + i*vecdim,base_number, vecdim, k);
        //auto res = pq_flat_search_optimized(codebook, encoded_base.data(), test_query + i*vecdim, base_number, vecdim, k);

        struct timeval newVal;
        ret = gettimeofday(&newVal, NULL);
        int64_t diff = (newVal.tv_sec * Converter + newVal.tv_usec) - (val.tv_sec * Converter + val.tv_usec);

        std::set<uint32_t> gtset;
        for(int j = 0; j < k; ++j){
            int t = test_gt[j + i*test_gt_d];
            gtset.insert(t);
        }

        size_t acc = 0;
        while (res.size()) {   
            int x = res.top().second;
            if(gtset.find(x) != gtset.end()){
                ++acc;
            }
            res.pop();
        }
        float recall = (float)acc/k;

        results[i] = {recall, diff};
    }

    float avg_recall = 0, avg_latency = 0;
    for(int i = 0; i < test_number; ++i) {
        avg_recall += results[i].recall;
        avg_latency += results[i].latency;
    }

    // 浮点误差可能导致一些精确算法平均recall不是1
    std::cout << "average recall: "<<avg_recall / test_number<<"\n";
    std::cout << "average latency (us): "<<avg_latency / test_number<<"\n";
    return 0;
}
