#ifndef EXHAUSTIVE_BENCHMARK_H
#define EXHAUSTIVE_BENCHMARK_H

#include "UserItemStore.h"
#include "LSH.h"
#include "Triplet.h"
#include <vector>
#include <chrono>
#include <map>
#include <set>
#include <string>
#include <iostream>
#include <iomanip>

// Estructura para almacenar resultados de recomendación
struct RecommendationResult {
    int item_id;
    double score;           // Para exhaustivo: similitud coseno
    int hamming_distance;   // Para LSH: distancia Hamming
    int rank;              // Posición en el ranking
    
    RecommendationResult(int id, double s, int h, int r) 
        : item_id(id), score(s), hamming_distance(h), rank(r) {}
};

// Estructura para métricas de evaluación
struct EvaluationMetrics {
    double precision_at_k = 0.0;
    double recall_at_k = 0.0;
    double ndcg_at_k = 0.0;
    double map_score = 0.0;
    double avg_retrieval_time_ms = 0.0;
    double avg_similarity_score = 0.0;
    int total_recommendations = 0;
    
    void print(const std::string& method_name) const {
        std::cout << "\n=== " << method_name << " Metrics ===" << std::endl;
        std::cout << std::fixed << std::setprecision(4);
        std::cout << "  Precision@K:        " << precision_at_k << std::endl;
        std::cout << "  Recall@K:           " << recall_at_k << std::endl;
        std::cout << "  NDCG@K:             " << ndcg_at_k << std::endl;
        std::cout << "  MAP Score:          " << map_score << std::endl;
        std::cout << "  Avg Retrieval Time: " << avg_retrieval_time_ms << " ms" << std::endl;
        std::cout << "  Avg Similarity:     " << avg_similarity_score << std::endl;
        std::cout << "  Total Recs:         " << total_recommendations << std::endl;
    }
};

// Estructura para comparativa de rendimiento
struct PerformanceComparison {
    EvaluationMetrics exhaustive_metrics;
    EvaluationMetrics lsh_metrics;
    double speedup_factor = 0.0;
    double accuracy_loss = 0.0;
    double efficiency_gain = 0.0;
    
    void print_comparison() const {
        std::cout << "\n" << std::string(80, '=') << std::endl;
        std::cout << "COMPARATIVA EXHAUSTIVO vs LSH" << std::endl;
        std::cout << std::string(80, '=') << std::endl;
        
        exhaustive_metrics.print("EXHAUSTIVO");
        lsh_metrics.print("LSH");
        
        std::cout << "\n=== COMPARACIÓN DIRECTA ===" << std::endl;
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "  Speedup Factor:     " << speedup_factor << "x" << std::endl;
        std::cout << "  Accuracy Loss:      " << accuracy_loss * 100 << "%" << std::endl;
        std::cout << "  Efficiency Gain:    " << efficiency_gain << std::endl;
        
        if (speedup_factor > 1.0) {
            std::cout << "  🚀 LSH es " << speedup_factor << "x más rápido!" << std::endl;
        }
        if (accuracy_loss < 0.1) {
            std::cout << "  ✅ Pérdida de precisión mínima (<10%)" << std::endl;
        }
    }
};

class ExhaustiveBenchmark {
public:
    ExhaustiveBenchmark(UserItemStore& store, SRPHasher& hasher);
    
    // === MÉTODOS DE BÚSQUEDA ===
    
    // Búsqueda exhaustiva O(n×d) - compara todos los items
    std::vector<RecommendationResult> exhaustive_search(
        int user_id, 
        int top_k,
        std::chrono::microseconds& retrieval_time
    ) const;
    
    // Búsqueda LSH O(b) - usa códigos binarios y Hamming distance
    std::vector<RecommendationResult> lsh_search(
        int user_id, 
        int top_k,
        std::chrono::microseconds& retrieval_time
    ) const;
    
    // === MÉTODOS DE EVALUACIÓN ===
    
    // Evalúa un conjunto de usuarios con ambos métodos
    PerformanceComparison benchmark_methods(
        const std::vector<int>& test_users,
        int top_k = 10,
        bool verbose = false
    ) const;
    
    // Evalúa precisión comparando con ground truth
    EvaluationMetrics evaluate_recommendations(
        const std::vector<RecommendationResult>& recommendations,
        const std::set<int>& ground_truth_items,
        double avg_retrieval_time_ms
    ) const;
    
    // === ANÁLISIS DE ESCALABILIDAD ===
    
    // Benchmarks con diferentes tamaños de catálogo
    void scalability_analysis(
        const std::vector<int>& catalog_sizes,
        const std::vector<int>& test_users,
        int top_k = 10
    ) const;
    
    // Benchmarks con diferentes configuraciones LSH
    void lsh_configuration_analysis(
        const std::vector<int>& lsh_bits,
        const std::vector<int>& test_users,
        int top_k = 10
    ) const;
    
    // === UTILIDADES ===
    
    // Calcula similitud coseno entre dos vectores
    double cosine_similarity(const Vector& v1, const Vector& v2) const;
    
    // Calcula distancia Hamming entre códigos
    int hamming_distance(const std::string& code1, const std::string& code2) const;
    
    // Genera ground truth basado en ratings reales
    std::map<int, std::set<int>> generate_ground_truth(
        const std::vector<Triplet>& validation_triplets
    ) const;
    
    // === MÉTRICAS ESPECÍFICAS ===
    
    // Calcula Precision@K
    double calculate_precision_at_k(
        const std::vector<RecommendationResult>& recommendations,
        const std::set<int>& ground_truth,
        int k
    ) const;
    
    // Calcula Recall@K
    double calculate_recall_at_k(
        const std::vector<RecommendationResult>& recommendations,
        const std::set<int>& ground_truth,
        int k
    ) const;
    
    // Calcula NDCG@K (Normalized Discounted Cumulative Gain)
    double calculate_ndcg_at_k(
        const std::vector<RecommendationResult>& recommendations,
        const std::set<int>& ground_truth,
        int k
    ) const;
    
    // Calcula MAP (Mean Average Precision)
    double calculate_map(
        const std::vector<RecommendationResult>& recommendations,
        const std::set<int>& ground_truth
    ) const;
    
    // === REPORTES ===
    
    // Genera reporte detallado en formato tabla
    void generate_detailed_report(
        const PerformanceComparison& comparison,
        const std::string& output_file = ""
    ) const;
    
    // Genera gráficos de comparación (ASCII art)
    void generate_ascii_charts(
        const PerformanceComparison& comparison
    ) const;
    
    // === CONFIGURACIÓN ===
    
    struct BenchmarkConfig {
        int top_k = 10;
        int num_test_users = 50;
        bool measure_similarity_correlation = true;
        bool generate_charts = true;
        bool save_detailed_results = false;
        std::string output_directory = "benchmark_results/";
        
        // Parámetros específicos del paper Le et al.
        bool use_paper_metrics = true;  // Usar métricas exactas del paper
        double similarity_threshold = 0.1;  // Umbral para considerar similitud
        int max_catalog_size = 10000;  // Máximo items para prueba escalabilidad
    };
    
    void set_config(const BenchmarkConfig& config) { this->config = config; }
    BenchmarkConfig get_config() const { return config; }

private:
    UserItemStore& store;
    SRPHasher& hasher;
    BenchmarkConfig config;
    
    // === MÉTODOS AUXILIARES PRIVADOS ===
    
    // Convierte ranking LSH a formato estándar
    std::vector<RecommendationResult> convert_lsh_ranking(
        const std::vector<std::pair<int, int>>& lsh_results,
        int user_id
    ) const;
    
    // Convierte ranking exhaustivo a formato estándar
    std::vector<RecommendationResult> convert_exhaustive_ranking(
        const std::vector<std::pair<int, double>>& exhaustive_results
    ) const;
    
    // Calcula estadísticas de distribución de similitudes
    void analyze_similarity_distribution(
        const std::vector<RecommendationResult>& exhaustive_results,
        const std::vector<RecommendationResult>& lsh_results
    ) const;
    
    // Mide correlación entre rankings
    double calculate_ranking_correlation(
        const std::vector<RecommendationResult>& ranking1,
        const std::vector<RecommendationResult>& ranking2
    ) const;
    
    // Genera estadísticas de tiempo por operación
    void time_analysis(
        const std::vector<std::chrono::microseconds>& exhaustive_times,
        const std::vector<std::chrono::microseconds>& lsh_times
    ) const;
    
    // Agrega métricas de múltiples evaluaciones
    EvaluationMetrics aggregate_metrics(
        const std::vector<EvaluationMetrics>& metrics_list
    ) const;
};

#endif // EXHAUSTIVE_BENCHMARK_H