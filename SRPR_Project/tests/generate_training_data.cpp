#include "../include/Triplet.h"
#include <iostream>
#include <vector>
#include <set>
#include <chrono>
#include <fstream>

int main(int argc, char* argv[]) {
    std::cout << "=== Generador de Dataset de Entrenamiento SRPR ===" << std::endl;
    
    // Configuración por defecto
    int max_ratings = 500000;  // 500K ratings para dataset de entrenamiento
    int max_triplets_per_user = 100;
    double min_rating_diff = 1.0;  // Diferencia mínima de rating más estricta
    std::string output_file = "data/training_triplets.csv";
    
    // Procesar argumentos de línea de comandos
    if (argc > 1) {
        max_ratings = std::atoi(argv[1]);
    }
    if (argc > 2) {
        max_triplets_per_user = std::atoi(argv[2]);
    }
    if (argc > 3) {
        min_rating_diff = std::atof(argv[3]);
    }
    if (argc > 4) {
        output_file = argv[4];
    }
    
    std::cout << "\nConfiguración del generador:" << std::endl;
    std::cout << "  - Máximo ratings a procesar: " << max_ratings << std::endl;
    std::cout << "  - Máximo tripletas por usuario: " << max_triplets_per_user << std::endl;
    std::cout << "  - Diferencia mínima de rating: " << min_rating_diff << std::endl;
    std::cout << "  - Archivo de salida: " << output_file << std::endl;
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // === PASO 1: Cargar ratings de MovieLens ===
    std::cout << "\n--- Paso 1: Cargando ratings de MovieLens ---" << std::endl;
    
    std::vector<Rating> ratings = load_movielens_ratings(
        "data/movielens/ml-20m/ratings.csv", 
        max_ratings
    );
    
    if (ratings.empty()) {
        std::cerr << "ERROR: No se pudieron cargar los ratings de MovieLens." << std::endl;
        std::cerr << "Verifica que existe el archivo: data/movielens/ml-20m/ratings.csv" << std::endl;
        return 1;
    }
    
    // === PASO 2: Análisis de ratings cargados ===
    std::cout << "\n--- Paso 2: Análisis de ratings ---" << std::endl;
    
    std::set<int> unique_users_ratings, unique_movies_ratings;
    std::map<double, int> rating_distribution;
    
    for (const auto& r : ratings) {
        unique_users_ratings.insert(r.user_id);
        unique_movies_ratings.insert(r.movie_id);
        rating_distribution[r.rating]++;
    }
    
    std::cout << "Estadísticas de ratings cargados:" << std::endl;
    std::cout << "  ✓ Total de ratings: " << ratings.size() << std::endl;
    std::cout << "  ✓ Usuarios únicos: " << unique_users_ratings.size() << std::endl;
    std::cout << "  ✓ Películas únicas: " << unique_movies_ratings.size() << std::endl;
    
    std::cout << "  ✓ Distribución de ratings:" << std::endl;
    for (const auto& pair : rating_distribution) {
        std::cout << "    " << pair.first << " estrellas: " << pair.second 
                  << " (" << (100.0 * pair.second / ratings.size()) << "%)" << std::endl;
    }
    
    // === PASO 3: Convertir a tripletas con configuración optimizada ===
    std::cout << "\n--- Paso 3: Convirtiendo a tripletas ---" << std::endl;
    
    std::vector<Triplet> triplets;
    
    // Agrupar ratings por usuario
    std::map<int, std::vector<Rating>> user_ratings;
    for (const auto& rating : ratings) {
        user_ratings[rating.user_id].push_back(rating);
    }
    
    std::mt19937 rng(42); // Seed fijo para reproducibilidad
    int users_processed = 0;
    int users_with_sufficient_ratings = 0;
    
    for (const auto& user_pair : user_ratings) {
        int user_id = user_pair.first;
        const auto& user_movie_ratings = user_pair.second;
        users_processed++;
        
        // Solo procesar usuarios con suficientes ratings
        if (user_movie_ratings.size() < 5) {
            continue;
        }
        users_with_sufficient_ratings++;
        
        std::vector<Triplet> user_triplets;
        
        // Generar tripletas para este usuario
        for (size_t i = 0; i < user_movie_ratings.size(); ++i) {
            for (size_t j = i + 1; j < user_movie_ratings.size(); ++j) {
                const auto& rating_i = user_movie_ratings[i];
                const auto& rating_j = user_movie_ratings[j];
                
                // Crear tripleta solo si hay diferencia significativa
                if (std::abs(rating_i.rating - rating_j.rating) >= min_rating_diff) {
                    if (rating_i.rating > rating_j.rating) {
                        user_triplets.push_back({user_id, rating_i.movie_id, rating_j.movie_id});
                    } else {
                        user_triplets.push_back({user_id, rating_j.movie_id, rating_i.movie_id});
                    }
                }
            }
        }
        
        // Limitar y mezclar tripletas
        if (user_triplets.size() > max_triplets_per_user) {
            std::shuffle(user_triplets.begin(), user_triplets.end(), rng);
            user_triplets.resize(max_triplets_per_user);
        }
        
        triplets.insert(triplets.end(), user_triplets.begin(), user_triplets.end());
        
        // Mostrar progreso cada 100 usuarios
        if (users_processed % 100 == 0) {
            std::cout << "  Procesados " << users_processed << " usuarios..." << std::endl;
        }
    }
    
    std::cout << "Conversión completada:" << std::endl;
    std::cout << "  ✓ Usuarios procesados: " << users_processed << std::endl;
    std::cout << "  ✓ Usuarios con suficientes ratings: " << users_with_sufficient_ratings << std::endl;
    std::cout << "  ✓ Tripletas generadas: " << triplets.size() << std::endl;
    
    // === PASO 4: Estadísticas finales ===
    std::cout << "\n--- Paso 4: Estadísticas del dataset final ---" << std::endl;
    
    std::set<int> final_users, final_movies;
    for (const auto& t : triplets) {
        final_users.insert(t.user_id);
        final_movies.insert(t.preferred_item_id);
        final_movies.insert(t.less_preferred_item_id);
    }
    
    std::cout << "Dataset de entrenamiento generado:" << std::endl;
    std::cout << "  ✓ Tripletas totales: " << triplets.size() << std::endl;
    std::cout << "  ✓ Usuarios únicos: " << final_users.size() << std::endl;
    std::cout << "  ✓ Películas únicas: " << final_movies.size() << std::endl;
    std::cout << "  ✓ Densidad promedio: " 
              << (double)triplets.size() / (final_users.size() * final_movies.size()) << std::endl;
    std::cout << "  ✓ Tripletas por usuario: " << (double)triplets.size() / final_users.size() << std::endl;
    
    // === PASO 5: Guardar dataset ===
    std::cout << "\n--- Paso 5: Guardando dataset ---" << std::endl;
    
    std::ofstream output(output_file);
    if (!output.is_open()) {
        std::cerr << "ERROR: No se pudo crear el archivo: " << output_file << std::endl;
        return 1;
    }
    
    // Escribir header
    output << "user_id,preferred_item_id,less_preferred_item_id\n";
    
    // Escribir tripletas
    for (const auto& t : triplets) {
        output << t.user_id << "," << t.preferred_item_id << "," << t.less_preferred_item_id << "\n";
    }
    
    output.close();
    
    // === PASO 6: Crear dataset de validación ===
    std::cout << "\n--- Paso 6: Creando dataset de validación ---" << std::endl;
    
    // Tomar 10% para validación
    int validation_size = triplets.size() * 0.1;
    std::shuffle(triplets.begin(), triplets.end(), rng);
    
    std::string validation_file = "data/validation_triplets.csv";
    std::ofstream validation_output(validation_file);
    if (validation_output.is_open()) {
        validation_output << "user_id,preferred_item_id,less_preferred_item_id\n";
        for (int i = 0; i < validation_size; ++i) {
            const auto& t = triplets[i];
            validation_output << t.user_id << "," << t.preferred_item_id << "," << t.less_preferred_item_id << "\n";
        }
        validation_output.close();
        std::cout << "  ✓ Dataset de validación guardado: " << validation_file 
                  << " (" << validation_size << " tripletas)" << std::endl;
    }
    
    // === PASO 7: Resumen final ===
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
    
    std::cout << "\n=== RESUMEN FINAL ===" << std::endl;
    std::cout << "✅ Dataset de entrenamiento generado exitosamente!" << std::endl;
    std::cout << "📁 Archivos creados:" << std::endl;
    std::cout << "   - " << output_file << " (" << triplets.size() - validation_size << " tripletas)" << std::endl;
    std::cout << "   - " << validation_file << " (" << validation_size << " tripletas)" << std::endl;
    std::cout << "⏱️  Tiempo total: " << duration.count() << " segundos" << std::endl;
    std::cout << "🎯 El dataset está listo para entrenar el modelo SRPR!" << std::endl;
    
    // Instrucciones de uso
    std::cout << "\n📋 INSTRUCCIONES DE USO:" << std::endl;
    std::cout << "   Para generar datasets con diferentes parámetros:" << std::endl;
    std::cout << "   ./generate_training_data [max_ratings] [max_triplets_per_user] [min_rating_diff] [output_file]" << std::endl;
    std::cout << "   Ejemplo: ./generate_training_data 1000000 50 0.5 data/large_training.csv" << std::endl;
    
    return 0;
}