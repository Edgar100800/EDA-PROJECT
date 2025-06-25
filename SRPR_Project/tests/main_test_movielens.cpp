#include "../include/Triplet.h"
#include <iostream>
#include <vector>
#include <set>
#include <chrono>

int main() {
    std::cout << "=== Prueba de Carga de Dataset MovieLens ===" << std::endl;
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Configuración de la prueba
    std::string ratings_file = "data/movielens/ml-20m/ratings.csv";
    int max_ratings = 100000;  // Cargar solo 100k ratings para la prueba
    int max_triplets_per_user = 30;
    
    std::cout << "\nConfiguración de la prueba:" << std::endl;
    std::cout << "  - Archivo: " << ratings_file << std::endl;
    std::cout << "  - Máximo ratings: " << max_ratings << std::endl;
    std::cout << "  - Máximo tripletas por usuario: " << max_triplets_per_user << std::endl;
    
    // === PASO 1: Cargar tripletas desde MovieLens ===
    std::cout << "\n--- Paso 1: Cargando datos de MovieLens ---" << std::endl;
    
    std::vector<Triplet> triplets = load_movielens_triplets(
        ratings_file, 
        max_ratings, 
        max_triplets_per_user
    );
    
    if (triplets.empty()) {
        std::cerr << "ERROR: No se pudieron cargar las tripletas de MovieLens." << std::endl;
        std::cerr << "Verifica que el archivo existe en: " << ratings_file << std::endl;
        return 1;
    }
    
    // === PASO 2: Estadísticas básicas ===
    std::cout << "\n--- Paso 2: Análisis de datos cargados ---" << std::endl;
    
    std::set<int> unique_users, unique_movies;
    for (const auto& t : triplets) {
        unique_users.insert(t.user_id);
        unique_movies.insert(t.preferred_item_id);
        unique_movies.insert(t.less_preferred_item_id);
    }
    
    std::cout << "Estadísticas del dataset:" << std::endl;
    std::cout << "  ✓ Total de tripletas: " << triplets.size() << std::endl;
    std::cout << "  ✓ Usuarios únicos: " << unique_users.size() << std::endl;
    std::cout << "  ✓ Películas únicas: " << unique_movies.size() << std::endl;
    std::cout << "  ✓ Promedio de tripletas por usuario: " 
              << (double)triplets.size() / unique_users.size() << std::endl;
    
    // === PASO 3: Mostrar ejemplos ===
    std::cout << "\n--- Paso 3: Ejemplos de tripletas generadas ---" << std::endl;
    
    int examples_to_show = std::min(10, (int)triplets.size());
    std::cout << "Primeras " << examples_to_show << " tripletas:" << std::endl;
    
    for (int i = 0; i < examples_to_show; ++i) {
        const auto& t = triplets[i];
        std::cout << "  " << (i+1) << ". Usuario " << t.user_id 
                  << " prefiere película " << t.preferred_item_id
                  << " sobre película " << t.less_preferred_item_id << std::endl;
    }
    
    // === PASO 4: Verificación de calidad de datos ===
    std::cout << "\n--- Paso 4: Verificación de calidad ---" << std::endl;
    
    // Verificar que no hay preferencias auto-referenciadas
    int self_preferences = 0;
    for (const auto& t : triplets) {
        if (t.preferred_item_id == t.less_preferred_item_id) {
            self_preferences++;
        }
    }
    
    if (self_preferences > 0) {
        std::cerr << "ADVERTENCIA: " << self_preferences 
                  << " tripletas tienen auto-referencias." << std::endl;
    } else {
        std::cout << "  ✓ No hay auto-referencias en las tripletas." << std::endl;
    }
    
    // Verificar distribución de usuarios
    std::map<int, int> user_triplet_count;
    for (const auto& t : triplets) {
        user_triplet_count[t.user_id]++;
    }
    
    int min_triplets = INT_MAX, max_triplets = 0;
    for (const auto& pair : user_triplet_count) {
        min_triplets = std::min(min_triplets, pair.second);
        max_triplets = std::max(max_triplets, pair.second);
    }
    
    std::cout << "  ✓ Distribución de tripletas por usuario:" << std::endl;
    std::cout << "    - Mínimo: " << min_triplets << " tripletas" << std::endl;
    std::cout << "    - Máximo: " << max_triplets << " tripletas" << std::endl;
    
    // === PASO 5: Tiempo de ejecución ===
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    std::cout << "\n--- Resumen de rendimiento ---" << std::endl;
    std::cout << "  ✓ Tiempo total de carga: " << duration.count() << " ms" << std::endl;
    std::cout << "  ✓ Tripletas por segundo: " 
              << (triplets.size() * 1000.0) / duration.count() << std::endl;
    
    // === PASO 6: Guardar muestra para pruebas futuras ===
    std::cout << "\n--- Paso 6: Guardando muestra para pruebas ---" << std::endl;
    
    std::ofstream sample_file("data/movielens_sample.csv");
    if (sample_file.is_open()) {
        int sample_size = std::min(1000, (int)triplets.size());
        for (int i = 0; i < sample_size; ++i) {
            const auto& t = triplets[i];
            sample_file << t.user_id << "," << t.preferred_item_id 
                       << "," << t.less_preferred_item_id << "\n";
        }
        sample_file.close();
        std::cout << "  ✓ Guardadas " << sample_size 
                  << " tripletas en data/movielens_sample.csv" << std::endl;
    }
    
    std::cout << "\n🎉 ¡Prueba de MovieLens completada exitosamente!" << std::endl;
    std::cout << "✅ Los datos están listos para ser usados en el entrenamiento SRPR." << std::endl;
    
    return 0;
}