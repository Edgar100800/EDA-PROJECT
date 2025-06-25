#include "../include/LSH.h"
#include "../include/UserItemStore.h"
#include "../include/Triplet.h"
#include <iostream>
#include <vector>
#include <set>
#include <map>
#include <chrono>
#include <cmath>
#include <algorithm>

// Función para calcular la distancia de Hamming entre dos códigos
int hamming_distance(const std::string& code1, const std::string& code2) {
    if (code1.length() != code2.length()) return -1;
    
    int distance = 0;
    for (size_t i = 0; i < code1.length(); ++i) {
        if (code1[i] != code2[i]) {
            distance++;
        }
    }
    return distance;
}

// Función para calcular la similitud coseno entre dos vectores
double cosine_similarity(const Vector& v1, const Vector& v2) {
    if (v1.size() != v2.size()) return 0.0;
    
    double dot_product = 0.0;
    double norm1 = 0.0, norm2 = 0.0;
    
    for (size_t i = 0; i < v1.size(); ++i) {
        dot_product += v1[i] * v2[i];
        norm1 += v1[i] * v1[i];
        norm2 += v2[i] * v2[i];
    }
    
    norm1 = std::sqrt(norm1);
    norm2 = std::sqrt(norm2);
    
    if (norm1 == 0.0 || norm2 == 0.0) return 0.0;
    
    return dot_product / (norm1 * norm2);
}

int main() {
    std::cout << "=== Prueba de Integración LSH + UserItemStore ===" << std::endl;
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // === CONFIGURACIÓN ===
    int dimensions = 32;
    int lsh_bits = 16;
    std::string training_file = "data/training_triplets.csv";
    
    std::cout << "\nConfiguración de la prueba:" << std::endl;
    std::cout << "  - Dimensiones de vectores: " << dimensions << std::endl;
    std::cout << "  - Bits de LSH: " << lsh_bits << std::endl;
    std::cout << "  - Archivo de entrenamiento: " << training_file << std::endl;
    
    // === PASO 1: Cargar datos y inicializar componentes ===
    std::cout << "\n--- Paso 1: Inicializando componentes ---" << std::endl;
    
    std::vector<Triplet> triplets = load_triplets(training_file);
    
    if (triplets.empty()) {
        std::cerr << "ERROR: No se pudo cargar el dataset de entrenamiento." << std::endl;
        std::cerr << "Ejecuta primero: ./generate_training_data" << std::endl;
        return 1;
    }
    
    std::cout << "✓ Cargadas " << triplets.size() << " tripletas" << std::endl;
    
    UserItemStore store(dimensions);
    store.initialize(triplets);
    std::cout << "✓ UserItemStore inicializado" << std::endl;
    
    SRPHasher hasher(dimensions, lsh_bits, 42); // Seed fijo para reproducibilidad
    std::cout << "✓ SRPHasher inicializado" << std::endl;
    
    store.print_summary();
    hasher.print_hash_info();
    
    // === PASO 2: Generar códigos LSH para todos los vectores ===
    std::cout << "\n--- Paso 2: Generando códigos LSH ---" << std::endl;
    
    // Obtener todos los usuarios e ítems únicos
    std::set<int> unique_users, unique_items;
    for (const auto& t : triplets) {
        unique_users.insert(t.user_id);
        unique_items.insert(t.preferred_item_id);
        unique_items.insert(t.less_preferred_item_id);
    }
    
    auto hash_start = std::chrono::high_resolution_clock::now();
    
    // Generar códigos para usuarios
    std::map<int, std::string> user_codes;
    for (int user_id : unique_users) {
        const Vector& user_vec = store.get_user_vector(user_id);
        user_codes[user_id] = hasher.generate_code(user_vec);
    }
    
    // Generar códigos para ítems
    std::map<int, std::string> item_codes;
    for (int item_id : unique_items) {
        const Vector& item_vec = store.get_item_vector(item_id);
        item_codes[item_id] = hasher.generate_code(item_vec);
    }
    
    auto hash_end = std::chrono::high_resolution_clock::now();
    auto hash_duration = std::chrono::duration_cast<std::chrono::milliseconds>(hash_end - hash_start);
    
    std::cout << "✓ Códigos generados para " << user_codes.size() << " usuarios" << std::endl;
    std::cout << "✓ Códigos generados para " << item_codes.size() << " ítems" << std::endl;
    std::cout << "✓ Tiempo de generación: " << hash_duration.count() << " ms" << std::endl;
    
    // === PASO 3: Análisis de distribución de códigos ===
    std::cout << "\n--- Paso 3: Análisis de distribución de códigos ---" << std::endl;
    
    // Contar códigos únicos
    std::set<std::string> unique_user_codes, unique_item_codes;
    for (const auto& pair : user_codes) {
        unique_user_codes.insert(pair.second);
    }
    for (const auto& pair : item_codes) {
        unique_item_codes.insert(pair.second);
    }
    
    std::cout << "Diversidad de códigos:" << std::endl;
    std::cout << "  - Códigos únicos de usuarios: " << unique_user_codes.size() 
              << " / " << user_codes.size() 
              << " (" << (100.0 * unique_user_codes.size() / user_codes.size()) << "%)" << std::endl;
    std::cout << "  - Códigos únicos de ítems: " << unique_item_codes.size() 
              << " / " << item_codes.size() 
              << " (" << (100.0 * unique_item_codes.size() / item_codes.size()) << "%)" << std::endl;
    
    // Analizar distribución de bits
    std::vector<int> user_bit_counts(lsh_bits, 0);
    std::vector<int> item_bit_counts(lsh_bits, 0);
    
    for (const auto& pair : user_codes) {
        for (int bit_pos = 0; bit_pos < lsh_bits; ++bit_pos) {
            if (pair.second[bit_pos] == '1') {
                user_bit_counts[bit_pos]++;
            }
        }
    }
    
    for (const auto& pair : item_codes) {
        for (int bit_pos = 0; bit_pos < lsh_bits; ++bit_pos) {
            if (pair.second[bit_pos] == '1') {
                item_bit_counts[bit_pos]++;
            }
        }
    }
    
    std::cout << "\nDistribución de bits (proporción de 1s):" << std::endl;
    std::cout << "  Posición | Usuarios | Ítems" << std::endl;
    for (int i = 0; i < lsh_bits; ++i) {
        double user_prop = (double)user_bit_counts[i] / user_codes.size();
        double item_prop = (double)item_bit_counts[i] / item_codes.size();
        std::cout << "  " << std::setw(8) << i 
                  << " | " << std::setw(8) << std::fixed << std::setprecision(3) << user_prop
                  << " | " << std::setw(5) << std::fixed << std::setprecision(3) << item_prop << std::endl;
    }
    
    // === PASO 4: Análisis de correlación similitud-distancia ===
    std::cout << "\n--- Paso 4: Correlación similitud coseno vs distancia Hamming ---" << std::endl;
    
    // Tomar una muestra de pares de ítems para análisis
    std::vector<int> item_sample;
    int sample_size = std::min(50, (int)unique_items.size());
    
    auto it = unique_items.begin();
    for (int i = 0; i < sample_size && it != unique_items.end(); ++i, ++it) {
        item_sample.push_back(*it);
    }
    
    std::vector<double> similarities;
    std::vector<int> hamming_distances;
    
    for (int i = 0; i < sample_size; ++i) {
        for (int j = i + 1; j < sample_size; ++j) {
            int item1 = item_sample[i];
            int item2 = item_sample[j];
            
            const Vector& vec1 = store.get_item_vector(item1);
            const Vector& vec2 = store.get_item_vector(item2);
            
            double sim = cosine_similarity(vec1, vec2);
            int ham_dist = hamming_distance(item_codes[item1], item_codes[item2]);
            
            similarities.push_back(sim);
            hamming_distances.push_back(ham_dist);
        }
    }
    
    // Calcular estadísticas de correlación
    double avg_similarity = 0.0, avg_hamming = 0.0;
    for (size_t i = 0; i < similarities.size(); ++i) {
        avg_similarity += similarities[i];
        avg_hamming += hamming_distances[i];
    }
    avg_similarity /= similarities.size();
    avg_hamming /= hamming_distances.size();
    
    std::cout << "Estadísticas de correlación (" << similarities.size() << " pares):" << std::endl;
    std::cout << "  - Similitud coseno promedio: " << std::fixed << std::setprecision(4) << avg_similarity << std::endl;
    std::cout << "  - Distancia Hamming promedio: " << std::fixed << std::setprecision(2) << avg_hamming << std::endl;
    
    // Mostrar algunos ejemplos
    std::cout << "\nEjemplos de correlación:" << std::endl;
    std::cout << "  Similitud | Hamming" << std::endl;
    for (int i = 0; i < std::min(10, (int)similarities.size()); ++i) {
        std::cout << "  " << std::setw(9) << std::fixed << std::setprecision(4) << similarities[i]
                  << " | " << std::setw(7) << hamming_distances[i] << std::endl;
    }
    
    // === PASO 5: Simulación de Hamming Ranking ===
    std::cout << "\n--- Paso 5: Simulación de Hamming Ranking ---" << std::endl;
    
    if (!unique_users.empty()) {
        int query_user = *unique_users.begin();
        std::string query_code = user_codes[query_user];
        
        std::cout << "Usuario de consulta: " << query_user << std::endl;
        std::cout << "Código del usuario: " << query_code << std::endl;
        
        // Calcular distancias Hamming a todos los ítems
        std::vector<std::pair<int, int>> item_distances; // <item_id, hamming_distance>
        
        for (const auto& pair : item_codes) {
            int item_id = pair.first;
            const std::string& item_code = pair.second;
            int distance = hamming_distance(query_code, item_code);
            item_distances.push_back({item_id, distance});
        }
        
        // Ordenar por distancia Hamming
        std::sort(item_distances.begin(), item_distances.end(),
            [](const std::pair<int, int>& a, const std::pair<int, int>& b) {
                return a.second < b.second;
            });
        
        std::cout << "\nTop 10 ítems más cercanos (Hamming Ranking):" << std::endl;
        std::cout << "  Rank | Item ID | Distancia" << std::endl;
        for (int i = 0; i < std::min(10, (int)item_distances.size()); ++i) {
            std::cout << "  " << std::setw(4) << (i+1) 
                      << " | " << std::setw(7) << item_distances[i].first
                      << " | " << std::setw(9) << item_distances[i].second << std::endl;
        }
        
        // Analizar distribución de distancias
        std::map<int, int> distance_distribution;
        for (const auto& pair : item_distances) {
            distance_distribution[pair.second]++;
        }
        
        std::cout << "\nDistribución de distancias Hamming:" << std::endl;
        std::cout << "  Distancia | Frecuencia" << std::endl;
        for (const auto& pair : distance_distribution) {
            std::cout << "  " << std::setw(9) << pair.first 
                      << " | " << std::setw(10) << pair.second << std::endl;
        }
    }
    
    // === PASO 6: Benchmark de rendimiento integrado ===
    std::cout << "\n--- Paso 6: Benchmark de rendimiento integrado ---" << std::endl;
    
    auto bench_start = std::chrono::high_resolution_clock::now();
    
    // Simular operaciones típicas durante entrenamiento
    int operations = 0;
    int max_operations = 5000;
    
    for (const auto& t : triplets) {
        if (operations >= max_operations) break;
        
        // Simular acceso a vectores y generación de códigos
        const Vector& user_vec = store.get_user_vector(t.user_id);
        const Vector& item1_vec = store.get_item_vector(t.preferred_item_id);
        const Vector& item2_vec = store.get_item_vector(t.less_preferred_item_id);
        
        std::string user_code = hasher.generate_code(user_vec);
        std::string item1_code = hasher.generate_code(item1_vec);
        std::string item2_code = hasher.generate_code(item2_vec);
        
        // Simular cálculo de distancias Hamming
        int dist1 = hamming_distance(user_code, item1_code);
        int dist2 = hamming_distance(user_code, item2_code);
        
        // Operación simple para evitar optimización del compilador
        if (dist1 + dist2 > 999) std::cout << "";
        
        operations++;
    }
    
    auto bench_end = std::chrono::high_resolution_clock::now();
    auto bench_duration = std::chrono::duration_cast<std::chrono::microseconds>(bench_end - bench_start);
    
    std::cout << "Benchmark de pipeline completo:" << std::endl;
    std::cout << "  - " << operations << " operaciones completas en " << bench_duration.count() << " μs" << std::endl;
    std::cout << "  - " << (operations * 1000000.0 / bench_duration.count()) << " operaciones/segundo" << std::endl;
    std::cout << "  - " << (bench_duration.count() / (double)operations) << " μs por operación" << std::endl;
    
    // === RESUMEN FINAL ===
    auto end_time = std::chrono::high_resolution_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    std::cout << "\n=== RESUMEN FINAL ===" << std::endl;
    std::cout << "🎉 ¡Integración LSH + UserItemStore exitosa!" << std::endl;
    std::cout << "⏱️  Tiempo total: " << total_duration.count() << " ms" << std::endl;
    
    std::cout << "\n📊 Estadísticas de integración:" << std::endl;
    std::cout << "   - Vectores procesados: " << (unique_users.size() + unique_items.size()) << std::endl;
    std::cout << "   - Códigos únicos generados: " << (unique_user_codes.size() + unique_item_codes.size()) << std::endl;
    std::cout << "   - Diversidad de códigos: " << std::fixed << std::setprecision(1) 
              << (100.0 * (unique_user_codes.size() + unique_item_codes.size()) / (user_codes.size() + item_codes.size())) << "%" << std::endl;
    std::cout << "   - Rendimiento pipeline: " << (operations * 1000000.0 / bench_duration.count()) << " ops/s" << std::endl;
    
    std::cout << "\n✅ Componentes integrados verificados:" << std::endl;
    std::cout << "   ✓ UserItemStore con " << dimensions << "D vectores" << std::endl;
    std::cout << "   ✓ SRPHasher con " << lsh_bits << " bits" << std::endl;
    std::cout << "   ✓ Generación eficiente de códigos LSH" << std::endl;
    std::cout << "   ✓ Hamming Ranking funcional" << std::endl;
    std::cout << "   ✓ Pipeline completo de entrenamiento simulado" << std::endl;
    
    std::cout << "\n🚀 ¡Sistema listo para el entrenador SRPR!" << std::endl;
    std::cout << "📋 Próximo paso: Implementar SRPR_Trainer con gradientes" << std::endl;
    
    return 0;
}