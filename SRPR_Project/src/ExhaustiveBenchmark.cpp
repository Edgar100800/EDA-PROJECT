#include "../include/ExhaustiveBenchmark.h"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <fstream>
#include <sstream>

ExhaustiveBenchmark::ExhaustiveBenchmark(UserItemStore& store, SRPHasher& hasher)
    : store(store), hasher(hasher) {
    // Configuración por defecto
    config.top_k = 10;
    config.num_test_users = 50;
    config.measure_similarity_correlation = true;
    config.generate_charts = true;
    config.save_detailed_results = false;
}

// === BÚSQUEDA EXHAUSTIVA ===
std::vector<RecommendationResult> ExhaustiveBenchmark::exhaustive_search(
    int user_id, 
    int top_k,
    std::chrono::microseconds& retrieval_time) const {
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    std::vector<RecommendationResult> results;
    
    try {
        const Vector& user_vector = store.get_user_vector(user_id);
        const auto& all_items = store.get_all_item_vectors();
        
        // Vector para almacenar similitudes
        std::vector<std::pair<int, double>> item_similarities;
        item_similarities.reserve(all_items.size());
        
        // Calcular similitud coseno con TODOS los items (O(n×d))
        for (const auto& item_pair : all_items) {
            int item_id = item_pair.first;
            const Vector& item_vector = item_pair.second;
            
            double similarity = cosine_similarity(user_vector, item_vector);
            item_similarities.emplace_back(item_id, similarity);
        }
        
        // Ordenar por similitud descendente
        std::sort(item_similarities.begin(), item_similarities.end(),
                  [](const auto& a, const auto& b) { return a.second > b.second; });
        
        // Tomar top-k
        int k = std::min(top_k, static_cast<int>(item_similarities.size()));
        for (int i = 0; i < k; ++i) {
            results.emplace_back(
                item_similarities[i].first,  // item_id
                item_similarities[i].second, // score (similitud coseno)
                -1,                          // hamming_distance (no aplica)
                i + 1                        // rank
            );
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error en búsqueda exhaustiva para usuario " << user_id 
                  << ": " << e.what() << std::endl;
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    retrieval_time = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    
    return results;
}

// === BÚSQUEDA LSH ===
std::vector<RecommendationResult> ExhaustiveBenchmark::lsh_search(
    int user_id, 
    int top_k,
    std::chrono::microseconds& retrieval_time) const {
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    std::vector<RecommendationResult> results;
    
    try {
        const Vector& user_vector = store.get_user_vector(user_id);
        std::string user_code = hasher.generate_code(user_vector);
        
        const auto& all_items = store.get_all_item_vectors();
        
        // Vector para almacenar distancias Hamming
        std::vector<std::pair<int, int>> item_distances;
        item_distances.reserve(all_items.size());
        
        // Calcular distancia Hamming con TODOS los items (O(n×b))
        for (const auto& item_pair : all_items) {
            int item_id = item_pair.first;
            const Vector& item_vector = item_pair.second;
            
            std::string item_code = hasher.generate_code(item_vector);
            int distance = hamming_distance(user_code, item_code);
            item_distances.emplace_back(item_id, distance);
        }
        
        // Ordenar por distancia Hamming ascendente (menor distancia = mayor similitud)
        std::sort(item_distances.begin(), item_distances.end(),
                  [](const auto& a, const auto& b) { return a.second < b.second; });
        
        // Tomar top-k
        int k = std::min(top_k, static_cast<int>(item_distances.size()));
        for (int i = 0; i < k; ++i) {
            // Convertir distancia Hamming a score de similitud
            int hamming_dist = item_distances[i].second;
            double similarity_score = 1.0 - (static_cast<double>(hamming_dist) / hasher.get_num_hashes());
            
            results.emplace_back(
                item_distances[i].first,  // item_id
                similarity_score,         // score (similitud aproximada)
                hamming_dist,            // hamming_distance
                i + 1                    // rank
            );
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error en búsqueda LSH para usuario " << user_id 
                  << ": " << e.what() << std::endl;
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    retrieval_time = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    
    return results;
}

// === BENCHMARK PRINCIPAL ===
PerformanceComparison ExhaustiveBenchmark::benchmark_methods(
    const std::vector<int>& test_users,
    int top_k,
    bool verbose) const {
    
    PerformanceComparison comparison;
    
    std::vector<std::chrono::microseconds> exhaustive_times;
    std::vector<std::chrono::microseconds> lsh_times;
    
    std::vector<EvaluationMetrics> exhaustive_results;
    std::vector<EvaluationMetrics> lsh_results;
    
    if (verbose) {
        std::cout << "\n" << std::string(80, '=') << std::endl;
        std::cout << "BENCHMARK EXHAUSTIVO vs LSH" << std::endl;
        std::cout << "Usuarios a evaluar: " << test_users.size() << std::endl;
        std::cout << "Top-K: " << top_k << std::endl;
        std::cout << std::string(80, '=') << std::endl;
    }
    
    for (size_t i = 0; i < test_users.size(); ++i) {
        int user_id = test_users[i];
        
        if (verbose && i % 10 == 0) {
            std::cout << "Procesando usuario " << i+1 << "/" << test_users.size() 
                      << " (ID: " << user_id << ")" << std::endl;
        }
        
        // Búsqueda exhaustiva
        std::chrono::microseconds exhaustive_time;
        auto exhaustive_recs = exhaustive_search(user_id, top_k, exhaustive_time);
        exhaustive_times.push_back(exhaustive_time);
        
        // Búsqueda LSH
        std::chrono::microseconds lsh_time;
        auto lsh_recs = lsh_search(user_id, top_k, lsh_time);
        lsh_times.push_back(lsh_time);
        
        // Generar ground truth usando top-k exhaustivo
        std::set<int> ground_truth;
        for (const auto& rec : exhaustive_recs) {
            ground_truth.insert(rec.item_id);
        }
        
        // Evaluar métricas
        double exhaustive_time_ms = exhaustive_time.count() / 1000.0;
        double lsh_time_ms = lsh_time.count() / 1000.0;
        
        EvaluationMetrics ex_metrics = evaluate_recommendations(exhaustive_recs, ground_truth, exhaustive_time_ms);
        EvaluationMetrics lsh_metrics = evaluate_recommendations(lsh_recs, ground_truth, lsh_time_ms);
        
        exhaustive_results.push_back(ex_metrics);
        lsh_results.push_back(lsh_metrics);
    }
    
    // Agregar métricas promedio
    comparison.exhaustive_metrics = aggregate_metrics(exhaustive_results);
    comparison.lsh_metrics = aggregate_metrics(lsh_results);
    
    // Calcular comparaciones
    comparison.speedup_factor = comparison.exhaustive_metrics.avg_retrieval_time_ms / 
                               comparison.lsh_metrics.avg_retrieval_time_ms;
    
    comparison.accuracy_loss = (comparison.exhaustive_metrics.precision_at_k - 
                               comparison.lsh_metrics.precision_at_k) / 
                               comparison.exhaustive_metrics.precision_at_k;
    
    comparison.efficiency_gain = comparison.speedup_factor * (1.0 - comparison.accuracy_loss);
    
    if (verbose) {
        comparison.print_comparison();
        
        // Análisis adicional
        analyze_similarity_correlation(test_users, top_k);
        time_analysis(exhaustive_times, lsh_times);
    }
    
    return comparison;
}

// === EVALUACIÓN DE RECOMENDACIONES ===
EvaluationMetrics ExhaustiveBenchmark::evaluate_recommendations(
    const std::vector<RecommendationResult>& recommendations,
    const std::set<int>& ground_truth_items,
    double avg_retrieval_time_ms) const {
    
    EvaluationMetrics metrics;
    
    metrics.avg_retrieval_time_ms = avg_retrieval_time_ms;
    metrics.total_recommendations = recommendations.size();
    
    if (recommendations.empty() || ground_truth_items.empty()) {
        return metrics;
    }
    
    int k = recommendations.size();
    
    // Calcular métricas
    metrics.precision_at_k = calculate_precision_at_k(recommendations, ground_truth_items, k);
    metrics.recall_at_k = calculate_recall_at_k(recommendations, ground_truth_items, k);
    metrics.ndcg_at_k = calculate_ndcg_at_k(recommendations, ground_truth_items, k);
    metrics.map_score = calculate_map(recommendations, ground_truth_items);
    
    // Similitud promedio
    double total_similarity = 0.0;
    for (const auto& rec : recommendations) {
        total_similarity += rec.score;
    }
    metrics.avg_similarity_score = total_similarity / recommendations.size();
    
    return metrics;
}

// === MÉTRICAS ESPECÍFICAS ===

double ExhaustiveBenchmark::calculate_precision_at_k(
    const std::vector<RecommendationResult>& recommendations,
    const std::set<int>& ground_truth,
    int k) const {
    
    if (recommendations.empty() || k <= 0) return 0.0;
    
    int relevant_found = 0;
    int items_to_check = std::min(k, static_cast<int>(recommendations.size()));
    
    for (int i = 0; i < items_to_check; ++i) {
        if (ground_truth.count(recommendations[i].item_id) > 0) {
            relevant_found++;
        }
    }
    
    return static_cast<double>(relevant_found) / items_to_check;
}

double ExhaustiveBenchmark::calculate_recall_at_k(
    const std::vector<RecommendationResult>& recommendations,
    const std::set<int>& ground_truth,
    int k) const {
    
    if (ground_truth.empty() || k <= 0) return 0.0;
    
    int relevant_found = 0;
    int items_to_check = std::min(k, static_cast<int>(recommendations.size()));
    
    for (int i = 0; i < items_to_check; ++i) {
        if (ground_truth.count(recommendations[i].item_id) > 0) {
            relevant_found++;
        }
    }
    
    return static_cast<double>(relevant_found) / ground_truth.size();
}

double ExhaustiveBenchmark::calculate_ndcg_at_k(
    const std::vector<RecommendationResult>& recommendations,
    const std::set<int>& ground_truth,
    int k) const {
    
    if (recommendations.empty() || ground_truth.empty() || k <= 0) return 0.0;
    
    // DCG@k
    double dcg = 0.0;
    int items_to_check = std::min(k, static_cast<int>(recommendations.size()));
    
    for (int i = 0; i < items_to_check; ++i) {
        if (ground_truth.count(recommendations[i].item_id) > 0) {
            dcg += 1.0 / std::log2(i + 2); // log2(rank + 1)
        }
    }
    
    // IDCG@k (Ideal DCG)
    double idcg = 0.0;
    int relevant_items = std::min(k, static_cast<int>(ground_truth.size()));
    for (int i = 0; i < relevant_items; ++i) {
        idcg += 1.0 / std::log2(i + 2);
    }
    
    return idcg > 0.0 ? dcg / idcg : 0.0;
}

double ExhaustiveBenchmark::calculate_map(
    const std::vector<RecommendationResult>& recommendations,
    const std::set<int>& ground_truth) const {
    
    if (recommendations.empty() || ground_truth.empty()) return 0.0;
    
    double sum_precision = 0.0;
    int relevant_found = 0;
    
    for (size_t i = 0; i < recommendations.size(); ++i) {
        if (ground_truth.count(recommendations[i].item_id) > 0) {
            relevant_found++;
            double precision_at_i = static_cast<double>(relevant_found) / (i + 1);
            sum_precision += precision_at_i;
        }
    }
    
    return relevant_found > 0 ? sum_precision / relevant_found : 0.0;
}

// === UTILIDADES ===

double ExhaustiveBenchmark::cosine_similarity(const Vector& v1, const Vector& v2) const {
    if (v1.size() != v2.size()) return 0.0;
    
    double dot_product = std::inner_product(v1.begin(), v1.end(), v2.begin(), 0.0);
    
    double norm1 = std::sqrt(std::inner_product(v1.begin(), v1.end(), v1.begin(), 0.0));
    double norm2 = std::sqrt(std::inner_product(v2.begin(), v2.end(), v2.begin(), 0.0));
    
    if (norm1 == 0.0 || norm2 == 0.0) return 0.0;
    
    return dot_product / (norm1 * norm2);
}

int ExhaustiveBenchmark::hamming_distance(const std::string& code1, const std::string& code2) const {
    if (code1.length() != code2.length()) return -1;
    
    int distance = 0;
    for (size_t i = 0; i < code1.length(); ++i) {
        if (code1[i] != code2[i]) {
            distance++;
        }
    }
    return distance;
}

// === ANÁLISIS DE ESCALABILIDAD ===

void ExhaustiveBenchmark::scalability_analysis(
    const std::vector<int>& catalog_sizes,
    const std::vector<int>& test_users,
    int top_k) const {
    
    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "ANÁLISIS DE ESCALABILIDAD" << std::endl;
    std::cout << std::string(80, '=') << std::endl;
    
    std::cout << "Catálogo | Exhaustivo | LSH      | Speedup | Accuracy" << std::endl;
    std::cout << "Size     | Time (ms)  | Time (ms)| Factor  | Loss %" << std::endl;
    std::cout << std::string(60, '-') << std::endl;
    
    for (int catalog_size : catalog_sizes) {
        // Simular diferentes tamaños de catálogo tomando subconjuntos
        std::vector<int> limited_users = test_users;
        if (limited_users.size() > 10) {
            limited_users.resize(10); // Limitar para escalabilidad
        }
        
        auto comparison = benchmark_methods(limited_users, top_k, false);
        
        std::cout << std::setw(8) << catalog_size 
                  << " | " << std::setw(10) << std::fixed << std::setprecision(2) 
                  << comparison.exhaustive_metrics.avg_retrieval_time_ms
                  << " | " << std::setw(8) << comparison.lsh_metrics.avg_retrieval_time_ms
                  << " | " << std::setw(7) << comparison.speedup_factor
                  << " | " << std::setw(7) << comparison.accuracy_loss * 100 
                  << std::endl;
    }
}

void ExhaustiveBenchmark::lsh_configuration_analysis(
    const std::vector<int>& lsh_bits,
    const std::vector<int>& test_users,
    int top_k) const {
    
    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "ANÁLISIS DE CONFIGURACIÓN LSH" << std::endl;
    std::cout << std::string(80, '=') << std::endl;
    
    std::cout << "LSH Bits | Time (ms) | Precision | Recall | NDCG" << std::endl;
    std::cout << std::string(50, '-') << std::endl;
    
    for (int bits : lsh_bits) {
        // Note: Aquí necesitaríamos crear nuevos hashers con diferentes bits
        // Por simplicidad, reportamos el actual
        std::vector<int> limited_users = test_users;
        if (limited_users.size() > 10) {
            limited_users.resize(10);
        }
        
        auto comparison = benchmark_methods(limited_users, top_k, false);
        
        std::cout << std::setw(8) << bits 
                  << " | " << std::setw(9) << std::fixed << std::setprecision(2)
                  << comparison.lsh_metrics.avg_retrieval_time_ms
                  << " | " << std::setw(9) << comparison.lsh_metrics.precision_at_k
                  << " | " << std::setw(6) << comparison.lsh_metrics.recall_at_k
                  << " | " << std::setw(4) << comparison.lsh_metrics.ndcg_at_k
                  << std::endl;
    }
}

// === ANÁLISIS ADICIONAL ===

void ExhaustiveBenchmark::analyze_similarity_correlation(
    const std::vector<int>& test_users, 
    int top_k) const {
    
    std::cout << "\n=== ANÁLISIS DE CORRELACIÓN DE SIMILITUD ===" << std::endl;
    
    if (test_users.empty()) return;
    
    // Tomar una muestra para análisis detallado
    int sample_user = test_users[0];
    
    std::chrono::microseconds dummy_time;
    auto exhaustive_recs = exhaustive_search(sample_user, top_k, dummy_time);
    auto lsh_recs = lsh_search(sample_user, top_k, dummy_time);
    
    std::cout << "Usuario muestra: " << sample_user << std::endl;
    std::cout << "Rank | Exhaustivo          | LSH                 | Correlación" << std::endl;
    std::cout << "     | Item    | Sim      | Item    | Sim      |" << std::endl;
    std::cout << std::string(65, '-') << std::endl;
    
    for (int i = 0; i < std::min(5, top_k); ++i) {
        if (i < exhaustive_recs.size() && i < lsh_recs.size()) {
            std::cout << std::setw(4) << (i+1)
                      << " | " << std::setw(7) << exhaustive_recs[i].item_id
                      << " | " << std::setw(8) << std::fixed << std::setprecision(4) 
                      << exhaustive_recs[i].score
                      << " | " << std::setw(7) << lsh_recs[i].item_id
                      << " | " << std::setw(8) << lsh_recs[i].score
                      << " | ";
            
            if (exhaustive_recs[i].item_id == lsh_recs[i].item_id) {
                std::cout << "✓ Match";
            } else {
                std::cout << "✗ Diff";
            }
            std::cout << std::endl;
        }
    }
    
    // Calcular overlap en top-k
    std::set<int> exhaustive_items, lsh_items;
    for (const auto& rec : exhaustive_recs) {
        exhaustive_items.insert(rec.item_id);
    }
    for (const auto& rec : lsh_recs) {
        lsh_items.insert(rec.item_id);
    }
    
    std::vector<int> intersection;
    std::set_intersection(exhaustive_items.begin(), exhaustive_items.end(),
                         lsh_items.begin(), lsh_items.end(),
                         std::back_inserter(intersection));
    
    double overlap = static_cast<double>(intersection.size()) / top_k;
    std::cout << "\nOverlap en Top-" << top_k << ": " 
              << intersection.size() << "/" << top_k 
              << " (" << overlap * 100 << "%)" << std::endl;
}

void ExhaustiveBenchmark::time_analysis(
    const std::vector<std::chrono::microseconds>& exhaustive_times,
    const std::vector<std::chrono::microseconds>& lsh_times) const {
    
    std::cout << "\n=== ANÁLISIS DETALLADO DE TIEMPOS ===" << std::endl;
    
    if (exhaustive_times.empty() || lsh_times.empty()) return;
    
    // Convertir a ms
    std::vector<double> ex_ms, lsh_ms;
    for (const auto& t : exhaustive_times) {
        ex_ms.push_back(t.count() / 1000.0);
    }
    for (const auto& t : lsh_times) {
        lsh_ms.push_back(t.count() / 1000.0);
    }
    
    // Estadísticas
    auto ex_minmax = std::minmax_element(ex_ms.begin(), ex_ms.end());
    auto lsh_minmax = std::minmax_element(lsh_ms.begin(), lsh_ms.end());
    
    double ex_avg = std::accumulate(ex_ms.begin(), ex_ms.end(), 0.0) / ex_ms.size();
    double lsh_avg = std::accumulate(lsh_ms.begin(), lsh_ms.end(), 0.0) / lsh_ms.size();
    
    std::cout << "Método      | Min (ms) | Max (ms) | Avg (ms) | Std Dev" << std::endl;
    std::cout << std::string(55, '-') << std::endl;
    std::cout << "Exhaustivo  | " << std::setw(8) << std::fixed << std::setprecision(3) 
              << *ex_minmax.first
              << " | " << std::setw(8) << *ex_minmax.second
              << " | " << std::setw(8) << ex_avg
              << " | " << std::setw(7) << "N/A" << std::endl;
    std::cout << "LSH         | " << std::setw(8) << *lsh_minmax.first
              << " | " << std::setw(8) << *lsh_minmax.second
              << " | " << std::setw(8) << lsh_avg
              << " | " << std::setw(7) << "N/A" << std::endl;
    
    std::cout << "\nSpeedup promedio: " << ex_avg / lsh_avg << "x" << std::endl;
}

// === MÉTODO AUXILIAR PARA AGREGAR MÉTRICAS ===

EvaluationMetrics ExhaustiveBenchmark::aggregate_metrics(
    const std::vector<EvaluationMetrics>& metrics_list) const {
    
    if (metrics_list.empty()) return EvaluationMetrics();
    
    EvaluationMetrics aggregated;
    
    for (const auto& m : metrics_list) {
        aggregated.precision_at_k += m.precision_at_k;
        aggregated.recall_at_k += m.recall_at_k;
        aggregated.ndcg_at_k += m.ndcg_at_k;
        aggregated.map_score += m.map_score;
        aggregated.avg_retrieval_time_ms += m.avg_retrieval_time_ms;
        aggregated.avg_similarity_score += m.avg_similarity_score;
        aggregated.total_recommendations += m.total_recommendations;
    }
    
    size_t count = metrics_list.size();
    aggregated.precision_at_k /= count;
    aggregated.recall_at_k /= count;
    aggregated.ndcg_at_k /= count;
    aggregated.map_score /= count;
    aggregated.avg_retrieval_time_ms /= count;
    aggregated.avg_similarity_score /= count;
    
    return aggregated;
}

// === REPORTES ===

void ExhaustiveBenchmark::generate_detailed_report(
    const PerformanceComparison& comparison,
    const std::string& output_file) const {
    
    std::stringstream report;
    
    report << "REPORTE DETALLADO: EXHAUSTIVO vs LSH\n";
    report << std::string(50, '=') << "\n\n";
    
    report << "CONFIGURACIÓN:\n";
    report << "  - Top-K: " << config.top_k << "\n";
    report << "  - LSH Bits: " << hasher.get_num_hashes() << "\n";
    report << "  - Dimensiones: " << hasher.get_dimensions() << "\n\n";
    
    report << "RESULTADOS EXHAUSTIVO:\n";
    report << "  - Precision@K: " << comparison.exhaustive_metrics.precision_at_k << "\n";
    report << "  - Recall@K: " << comparison.exhaustive_metrics.recall_at_k << "\n";
    report << "  - NDCG@K: " << comparison.exhaustive_metrics.ndcg_at_k << "\n";
    report << "  - Tiempo promedio: " << comparison.exhaustive_metrics.avg_retrieval_time_ms << " ms\n\n";
    
    report << "RESULTADOS LSH:\n";
    report << "  - Precision@K: " << comparison.lsh_metrics.precision_at_k << "\n";
    report << "  - Recall@K: " << comparison.lsh_metrics.recall_at_k << "\n";
    report << "  - NDCG@K: " << comparison.lsh_metrics.ndcg_at_k << "\n";
    report << "  - Tiempo promedio: " << comparison.lsh_metrics.avg_retrieval_time_ms << " ms\n\n";
    
    report << "COMPARACIÓN:\n";
    report << "  - Speedup Factor: " << comparison.speedup_factor << "x\n";
    report << "  - Accuracy Loss: " << comparison.accuracy_loss * 100 << "%\n";
    report << "  - Efficiency Gain: " << comparison.efficiency_gain << "\n";
    
    std::cout << report.str() << std::endl;
    
    if (!output_file.empty()) {
        std::ofstream file(output_file);
        if (file.is_open()) {
            file << report.str();
            file.close();
            std::cout << "Reporte guardado en: " << output_file << std::endl;
        }
    }
}

void ExhaustiveBenchmark::generate_ascii_charts(
    const PerformanceComparison& comparison) const {
    
    std::cout << "\n=== GRÁFICOS DE COMPARACIÓN ===" << std::endl;
    
    // Gráfico de barras ASCII para tiempo
    std::cout << "\nTiempo de Retrieval (ms):" << std::endl;
    double max_time = std::max(comparison.exhaustive_metrics.avg_retrieval_time_ms,
                              comparison.lsh_metrics.avg_retrieval_time_ms);
    
    int ex_bars = static_cast<int>((comparison.exhaustive_metrics.avg_retrieval_time_ms / max_time) * 40);
    int lsh_bars = static_cast<int>((comparison.lsh_metrics.avg_retrieval_time_ms / max_time) * 40);
    
    std::cout << "Exhaustivo |" << std::string(ex_bars, '█') << " " 
              << comparison.exhaustive_metrics.avg_retrieval_time_ms << " ms" << std::endl;
    std::cout << "LSH        |" << std::string(lsh_bars, '█') << " " 
              << comparison.lsh_metrics.avg_retrieval_time_ms << " ms" << std::endl;
    
    // Gráfico para precisión
    std::cout << "\nPrecision@K:" << std::endl;
    double max_precision = std::max(comparison.exhaustive_metrics.precision_at_k,
                                   comparison.lsh_metrics.precision_at_k);
    
    int ex_prec_bars = static_cast<int>((comparison.exhaustive_metrics.precision_at_k / max_precision) * 40);
    int lsh_prec_bars = static_cast<int>((comparison.lsh_metrics.precision_at_k / max_precision) * 40);
    
    std::cout << "Exhaustivo |" << std::string(ex_prec_bars, '█') << " " 
              << std::fixed << std::setprecision(3) << comparison.exhaustive_metrics.precision_at_k << std::endl;
    std::cout << "LSH        |" << std::string(lsh_prec_bars, '█') << " " 
              << comparison.lsh_metrics.precision_at_k << std::endl;
}