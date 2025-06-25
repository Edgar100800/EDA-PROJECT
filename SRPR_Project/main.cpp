#include "include/Triplet.h"
#include "include/UserItemStore.h"
#include "include/SRPR_Trainer.h"
#include "include/LSH.h"
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <map>
#include <set>
#include <fstream>
#include <sstream>

// Estructura para información de películas
struct Movie {
    int movie_id;
    std::string title;
    std::vector<std::string> genres;
    int year;
    
    Movie() : movie_id(0), year(0) {}
};

// Función para cargar información de películas desde movies.csv
std::map<int, Movie> load_movies_info(const std::string& movies_file) {
    std::map<int, Movie> movies;
    std::ifstream file(movies_file);
    
    if (!file.is_open()) {
        std::cerr << "Advertencia: No se pudo abrir " << movies_file << std::endl;
        return movies;
    }
    
    std::string line;
    std::getline(file, line); // Skip header
    
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string cell;
        Movie movie;
        
        // Parse movieId
        std::getline(ss, cell, ',');
        movie.movie_id = std::stoi(cell);
        
        // Parse title (may contain commas within quotes)
        std::string title_and_genres;
        std::getline(ss, title_and_genres);
        
        // Find the last comma to separate title from genres
        size_t last_comma = title_and_genres.rfind(',');
        if (last_comma != std::string::npos) {
            movie.title = title_and_genres.substr(0, last_comma);
            std::string genres_str = title_and_genres.substr(last_comma + 1);
            
            // Remove quotes if present
            if (movie.title.front() == '"' && movie.title.back() == '"') {
                movie.title = movie.title.substr(1, movie.title.length() - 2);
            }
            
            // Extract year from title
            size_t year_start = movie.title.rfind('(');
            size_t year_end = movie.title.rfind(')');
            if (year_start != std::string::npos && year_end != std::string::npos && year_end > year_start) {
                std::string year_str = movie.title.substr(year_start + 1, year_end - year_start - 1);
                try {
                    movie.year = std::stoi(year_str);
                } catch (...) {
                    movie.year = 0;
                }
            }
            
            // Parse genres
            std::stringstream genres_ss(genres_str);
            std::string genre;
            while (std::getline(genres_ss, genre, '|')) {
                if (!genre.empty()) {
                    movie.genres.push_back(genre);
                }
            }
        } else {
            movie.title = title_and_genres;
        }
        
        movies[movie.movie_id] = movie;
    }
    
    file.close();
    return movies;
}

// Función para mostrar el banner del sistema
void show_banner() {
    std::cout << "=================================================================================================" << std::endl;
    std::cout << "   ____  ____  ____  ____     ____                                                 _           " << std::endl;
    std::cout << "  / ___||  _ \\|  _ \\|  _ \\   |  _ \\ ___  ___ ___  _ __ ___  _ __ ___   ___ _ __   __| | ___ _ __ " << std::endl;
    std::cout << "  \\___ \\| |_) | |_) | |_) |  | |_) / _ \\/ __/ _ \\| '_ ` _ \\| '_ ` _ \\ / _ \\ '_ \\ / _` |/ _ \\ '__|" << std::endl;
    std::cout << "   ___) |  _ <|  __/|  _ <   |  _ <  __/ (_| (_) | | | | | | | | | | |  __/ | | | (_| |  __/ |   " << std::endl;
    std::cout << "  |____/|_| \\_\\_|   |_| \\_\\  |_| \\_\\___|\\___\\___/|_| |_| |_|_| |_| |_|\\___|_| |_|\\__,_|\\___|_|   " << std::endl;
    std::cout << "                                                                                                " << std::endl;
    std::cout << "  Stochastically Robust Personalized Ranking for LSH Recommendation Retrieval                " << std::endl;
    std::cout << "  Implementación en C++ con Dataset MovieLens ML-20M (20M ratings, 27K películas)           " << std::endl;
    std::cout << "=================================================================================================" << std::endl;
    std::cout << std::endl;
}

// Función para mostrar ayuda
void show_help() {
    std::cout << "Uso: ./srpr_system [opciones]" << std::endl;
    std::cout << std::endl;
    std::cout << "Opciones:" << std::endl;
    std::cout << "  --help, -h              Mostrar esta ayuda" << std::endl;
    std::cout << "  --train                 Entrenar modelo SRPR" << std::endl;
    std::cout << "  --recommend USER_ID     Generar recomendaciones para usuario" << std::endl;
    std::cout << "  --evaluate              Evaluar modelo entrenado" << std::endl;
    std::cout << "  --analyze               Analizar dataset MovieLens completo" << std::endl;
    std::cout << "  --generate-data         Generar tripletas desde MovieLens raw" << std::endl;
    std::cout << "  --data-file FILE        Archivo de datos (default: data/training_triplets.csv)" << std::endl;
    std::cout << "  --val-file FILE         Archivo de validación (default: data/validation_triplets.csv)" << std::endl;
    std::cout << "  --movies-file FILE      Archivo de películas (default: data/movielens/ml-20m/movies.csv)" << std::endl;
    std::cout << "  --ratings-file FILE     Archivo de ratings (default: data/movielens/ml-20m/ratings.csv)" << std::endl;
    std::cout << "  --epochs N              Número de epochs (default: 20)" << std::endl;
    std::cout << "  --lr RATE               Learning rate (default: 0.005)" << std::endl;
    std::cout << "  --dimensions N          Dimensiones de vectores (default: 32)" << std::endl;
    std::cout << "  --lsh-bits N            Bits de LSH (default: 16)" << std::endl;
    std::cout << "  --top-k N               Top-K recomendaciones (default: 10)" << std::endl;
    std::cout << "  --max-ratings N         Máximo ratings a procesar (default: 500000)" << std::endl;
    std::cout << "  --triplets-per-user N   Máximo tripletas por usuario (default: 50)" << std::endl;
    std::cout << "  --min-rating-diff D     Diferencia mínima de rating (default: 1.0)" << std::endl;
    std::cout << "  --genre GENRE           Filtrar recomendaciones por género" << std::endl;
    std::cout << "  --year-range START-END  Filtrar por rango de años (ej: 2000-2010)" << std::endl;
    std::cout << "  --verbose               Modo verboso" << std::endl;
    std::cout << std::endl;
    std::cout << "Ejemplos:" << std::endl;
    std::cout << "  ./srpr_system --generate-data --max-ratings 1000000 --triplets-per-user 100" << std::endl;
    std::cout << "  ./srpr_system --train --epochs 30 --lr 0.01 --verbose" << std::endl;
    std::cout << "  ./srpr_system --recommend 1 --top-k 20 --genre Action --year-range 2000-2020" << std::endl;
    std::cout << "  ./srpr_system --analyze --verbose" << std::endl;
    std::cout << "  ./srpr_system --evaluate --verbose" << std::endl;
}

// Función para generar recomendaciones usando Hamming Ranking con metadatos
std::vector<std::pair<int, int>> hamming_ranking_recommendations(
    int user_id, 
    const UserItemStore& store, 
    const SRPHasher& hasher, 
    const std::map<int, Movie>& movies,
    int top_k,
    const std::string& genre_filter = "",
    int year_start = 0,
    int year_end = 9999) {
    
    std::vector<std::pair<int, int>> recommendations; // <item_id, hamming_distance>
    
    try {
        const Vector& user_vector = store.get_user_vector(user_id);
        std::string user_code = hasher.generate_code(user_vector);
        
        // Obtener todos los vectores de items
        const auto& all_items = store.get_all_item_vectors();
        
        for (const auto& item_pair : all_items) {
            int item_id = item_pair.first;
            const Vector& item_vector = item_pair.second;
            
            // Aplicar filtros de metadatos
            auto movie_it = movies.find(item_id);
            if (movie_it != movies.end()) {
                const Movie& movie = movie_it->second;
                
                // Filtro por género
                if (!genre_filter.empty()) {
                    bool has_genre = false;
                    for (const std::string& genre : movie.genres) {
                        if (genre == genre_filter) {
                            has_genre = true;
                            break;
                        }
                    }
                    if (!has_genre) continue;
                }
                
                // Filtro por año
                if (movie.year < year_start || movie.year > year_end) {
                    continue;
                }
            }
            
            std::string item_code = hasher.generate_code(item_vector);
            
            // Calcular distancia de Hamming
            int distance = 0;
            for (size_t i = 0; i < user_code.length(); ++i) {
                if (user_code[i] != item_code[i]) {
                    distance++;
                }
            }
            
            recommendations.push_back({item_id, distance});
        }
        
        // Ordenar por distancia Hamming (menor distancia = mayor similitud)
        std::sort(recommendations.begin(), recommendations.end(),
            [](const std::pair<int, int>& a, const std::pair<int, int>& b) {
                return a.second < b.second;
            });
        
        // Tomar solo top-k
        if (recommendations.size() > top_k) {
            recommendations.resize(top_k);
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error generando recomendaciones para usuario " << user_id << ": " << e.what() << std::endl;
    }
    
    return recommendations;
}

// Función para calcular precisión de ranking
double calculate_ranking_precision(const std::vector<Triplet>& test_triplets, 
                                  const UserItemStore& store) {
    int correct_rankings = 0;
    int total_rankings = 0;
    
    for (const auto& triplet : test_triplets) {
        try {
            const Vector& user_vec = store.get_user_vector(triplet.user_id);
            const Vector& preferred_vec = store.get_item_vector(triplet.preferred_item_id);
            const Vector& less_preferred_vec = store.get_item_vector(triplet.less_preferred_item_id);
            
            // Calcular productos punto (similitudes)
            double score_preferred = 0.0, score_less_preferred = 0.0;
            for (size_t i = 0; i < user_vec.size(); ++i) {
                score_preferred += user_vec[i] * preferred_vec[i];
                score_less_preferred += user_vec[i] * less_preferred_vec[i];
            }
            
            if (score_preferred > score_less_preferred) {
                correct_rankings++;
            }
            total_rankings++;
            
        } catch (const std::exception& e) {
            continue;
        }
    }
    
    return total_rankings > 0 ? (double)correct_rankings / total_rankings : 0.0;
}

// Función para analizar el dataset MovieLens completo
int analyze_dataset(const std::string& ratings_file, const std::string& movies_file, bool verbose) {
    std::cout << "=== ANÁLISIS DEL DATASET MOVIELENS ML-20M ===" << std::endl;
    std::cout << std::endl;
    
    // Cargar información de películas
    std::cout << "Cargando información de películas..." << std::endl;
    auto movies = load_movies_info(movies_file);
    std::cout << "✓ Cargadas " << movies.size() << " películas con metadatos" << std::endl;
    
    // Analizar géneros
    std::map<std::string, int> genre_count;
    std::map<int, int> year_count;
    
    for (const auto& movie_pair : movies) {
        const Movie& movie = movie_pair.second;
        
        // Contar géneros
        for (const std::string& genre : movie.genres) {
            genre_count[genre]++;
        }
        
        // Contar años por década
        if (movie.year > 0) {
            int decade = (movie.year / 10) * 10;
            year_count[decade]++;
        }
    }
    
    std::cout << "\n--- Análisis de Géneros ---" << std::endl;
    std::cout << "Género                | Películas" << std::endl;
    std::cout << "----------------------|----------" << std::endl;
    
    // Ordenar géneros por popularidad
    std::vector<std::pair<std::string, int>> sorted_genres(genre_count.begin(), genre_count.end());
    std::sort(sorted_genres.begin(), sorted_genres.end(),
        [](const std::pair<std::string, int>& a, const std::pair<std::string, int>& b) { return a.second > b.second; });
    
    for (const auto& genre_pair : sorted_genres) {
        std::cout << std::setw(21) << std::left << genre_pair.first 
                  << "| " << std::setw(8) << genre_pair.second << std::endl;
    }
    
    std::cout << "\n--- Análisis Temporal ---" << std::endl;
    std::cout << "Década  | Películas" << std::endl;
    std::cout << "--------|----------" << std::endl;
    
    for (const auto& year_pair : year_count) {
        if (year_pair.first >= 1900) {
            std::cout << year_pair.first << "s | " << std::setw(8) << year_pair.second << std::endl;
        }
    }
    
    // Analizar ratings si está disponible
    if (!ratings_file.empty()) {
        std::cout << "\n--- Análisis de Ratings ---" << std::endl;
        
        std::ifstream ratings_stream(ratings_file);
        if (ratings_stream.is_open()) {
            std::string line;
            std::getline(ratings_stream, line); // Skip header
            
            std::map<double, int> rating_dist;
            std::set<int> unique_users, unique_movies;
            int total_ratings = 0;
            
            while (std::getline(ratings_stream, line) && total_ratings < 100000) { // Muestra de 100K
                std::stringstream ss(line);
                std::string cell;
                
                std::getline(ss, cell, ',');
                int user_id = std::stoi(cell);
                unique_users.insert(user_id);
                
                std::getline(ss, cell, ',');
                int movie_id = std::stoi(cell);
                unique_movies.insert(movie_id);
                
                std::getline(ss, cell, ',');
                double rating = std::stod(cell);
                rating_dist[rating]++;
                
                total_ratings++;
            }
            
            std::cout << "Muestra analizada: " << total_ratings << " ratings" << std::endl;
            std::cout << "Usuarios únicos: " << unique_users.size() << std::endl;
            std::cout << "Películas únicas: " << unique_movies.size() << std::endl;
            
            std::cout << "\nDistribución de ratings:" << std::endl;
            std::cout << "Rating | Frecuencia | Porcentaje" << std::endl;
            std::cout << "-------|------------|----------" << std::endl;
            
            for (const auto& rating_pair : rating_dist) {
                double percentage = (double)rating_pair.second / total_ratings * 100.0;
                std::cout << std::setw(6) << rating_pair.first 
                          << " | " << std::setw(10) << rating_pair.second
                          << " | " << std::setw(8) << std::fixed << std::setprecision(2) << percentage << "%" << std::endl;
            }
            
            ratings_stream.close();
        } else {
            std::cout << "⚠️ No se pudo abrir el archivo de ratings para análisis detallado" << std::endl;
        }
    }
    
    std::cout << "\n=== RESUMEN DEL DATASET ===" << std::endl;
    std::cout << "📊 Total de películas: " << movies.size() << std::endl;
    std::cout << "📊 Géneros únicos: " << genre_count.size() << std::endl;
    std::cout << "📊 Décadas representadas: " << year_count.size() << std::endl;
    std::cout << "📊 Género más popular: " << sorted_genres[0].first << " (" << sorted_genres[0].second << " películas)" << std::endl;
    
    return 0;
}

// Función para generar datos desde MovieLens raw
int generate_training_data(const std::string& ratings_file, int max_ratings, 
                          int triplets_per_user, double min_rating_diff, bool verbose) {
    std::cout << "=== GENERANDO DATASET DE ENTRENAMIENTO ===" << std::endl;
    std::cout << "Configuración:" << std::endl;
    std::cout << "  - Archivo de ratings: " << ratings_file << std::endl;
    std::cout << "  - Máximo ratings: " << max_ratings << std::endl;
    std::cout << "  - Tripletas por usuario: " << triplets_per_user << std::endl;
    std::cout << "  - Diferencia mínima rating: " << min_rating_diff << std::endl;
    std::cout << std::endl;
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Generar tripletas
    std::vector<Triplet> triplets = load_movielens_triplets(ratings_file, max_ratings, triplets_per_user);
    
    if (triplets.empty()) {
        std::cerr << "ERROR: No se pudieron generar tripletas." << std::endl;
        return 1;
    }
    
    // Dividir en entrenamiento y validación
    std::mt19937 rng(42);
    std::shuffle(triplets.begin(), triplets.end(), rng);
    
    int split_point = triplets.size() * 0.9;
    std::vector<Triplet> training(triplets.begin(), triplets.begin() + split_point);
    std::vector<Triplet> validation(triplets.begin() + split_point, triplets.end());
    
    // Guardar archivos
    std::ofstream train_file("data/training_triplets.csv");
    std::ofstream val_file("data/validation_triplets.csv");
    
    if (train_file.is_open()) {
        train_file << "user_id,preferred_item_id,less_preferred_item_id\n";
        for (const auto& t : training) {
            train_file << t.user_id << "," << t.preferred_item_id << "," << t.less_preferred_item_id << "\n";
        }
        train_file.close();
        std::cout << "✓ Guardado training_triplets.csv (" << training.size() << " tripletas)" << std::endl;
    }
    
    if (val_file.is_open()) {
        val_file << "user_id,preferred_item_id,less_preferred_item_id\n";
        for (const auto& t : validation) {
            val_file << t.user_id << "," << t.preferred_item_id << "," << t.less_preferred_item_id << "\n";
        }
        val_file.close();
        std::cout << "✓ Guardado validation_triplets.csv (" << validation.size() << " tripletas)" << std::endl;
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
    
    std::cout << "\n🎉 Dataset generado exitosamente en " << duration.count() << " segundos!" << std::endl;
    
    return 0;
}

// Función principal de entrenamiento
int train_model(const std::string& data_file, const std::string& val_file,
                int epochs, double learning_rate, int dimensions, 
                int lsh_bits, bool verbose) {
    
    std::cout << "=== INICIANDO ENTRENAMIENTO SRPR ===" << std::endl;
    std::cout << "Configuración:" << std::endl;
    std::cout << "  - Archivo de datos: " << data_file << std::endl;
    std::cout << "  - Archivo de validación: " << val_file << std::endl;
    std::cout << "  - Epochs: " << epochs << std::endl;
    std::cout << "  - Learning rate: " << learning_rate << std::endl;
    std::cout << "  - Dimensiones: " << dimensions << std::endl;
    std::cout << "  - LSH bits: " << lsh_bits << std::endl;
    std::cout << std::endl;
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Cargar datos
    std::cout << "Cargando datos de entrenamiento..." << std::endl;
    std::vector<Triplet> training_triplets = load_triplets(data_file);
    if (training_triplets.empty()) {
        std::cerr << "ERROR: No se pudieron cargar los datos de entrenamiento." << std::endl;
        std::cerr << "Ejecuta: ./srpr_system --generate-data" << std::endl;
        return 1;
    }
    std::cout << "✓ Cargadas " << training_triplets.size() << " tripletas de entrenamiento" << std::endl;
    
    std::vector<Triplet> validation_triplets;
    if (!val_file.empty()) {
        validation_triplets = load_triplets(val_file);
        std::cout << "✓ Cargadas " << validation_triplets.size() << " tripletas de validación" << std::endl;
    }
    
    // Inicializar sistema
    std::cout << "\nInicializando UserItemStore..." << std::endl;
    UserItemStore store(dimensions);
    store.initialize(training_triplets);
    store.print_summary();
    
    std::cout << "\nInicializando SRPR_Trainer..." << std::endl;
    SRPR_Trainer trainer(store);
    
    // Configurar parámetros de entrenamiento
    SRPR_Trainer::TrainingParams params;
    params.epochs = epochs;
    params.learning_rate = learning_rate;
    params.b_lsh_length = lsh_bits;
    params.regularization = 0.0005;
    params.verbose = verbose;
    params.validation_freq = std::max(1, epochs / 5);
    
    // Evaluación inicial
    if (verbose) {
        std::cout << "\nEvaluación inicial..." << std::endl;
        double initial_loss = trainer.calculate_total_loss(training_triplets, params);
        std::cout << "✓ Pérdida inicial: " << std::fixed << std::setprecision(6) << initial_loss << std::endl;
        
        if (!validation_triplets.empty()) {
            double initial_precision = calculate_ranking_precision(validation_triplets, store);
            std::cout << "✓ Precisión inicial: " << std::fixed << std::setprecision(4) 
                      << (initial_precision * 100) << "%" << std::endl;
        }
    }
    
    // Entrenamiento
    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "INICIANDO ENTRENAMIENTO" << std::endl;
    std::cout << std::string(80, '=') << std::endl;
    
    auto training_stats = trainer.train(training_triplets, params, validation_triplets);
    
    // Evaluación final
    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "EVALUACIÓN FINAL" << std::endl;
    std::cout << std::string(80, '=') << std::endl;
    
    double final_loss = trainer.calculate_total_loss(training_triplets, params);
    std::cout << "✓ Pérdida final: " << std::fixed << std::setprecision(6) << final_loss << std::endl;
    
    if (!validation_triplets.empty()) {
        double final_precision = calculate_ranking_precision(validation_triplets, store);
        std::cout << "✓ Precisión final: " << std::fixed << std::setprecision(4) 
                  << (final_precision * 100) << "%" << std::endl;
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
    
    std::cout << "\n🎉 ENTRENAMIENTO COMPLETADO" << std::endl;
    std::cout << "⏱️  Tiempo total: " << total_duration.count() << " segundos" << std::endl;
    std::cout << "📊 Actualizaciones: " << training_stats.total_updates << std::endl;
    std::cout << "🚀 Velocidad: " << std::fixed << std::setprecision(0) 
              << (training_stats.total_updates * 1000.0 / training_stats.training_time_ms) 
              << " actualizaciones/s" << std::endl;
    
    if (training_stats.converged) {
        std::cout << "✅ El modelo convergió exitosamente" << std::endl;
    } else {
        std::cout << "⚠️  El modelo no convergió completamente - considerar más epochs" << std::endl;
    }
    
    return 0;
}

// Función para generar recomendaciones
int generate_recommendations(int user_id, int top_k, int dimensions, int lsh_bits, 
                           const std::string& data_file, const std::string& movies_file,
                           const std::string& genre_filter, const std::string& year_range,
                           bool verbose) {
    
    std::cout << "=== GENERANDO RECOMENDACIONES ===" << std::endl;
    std::cout << "Usuario: " << user_id << std::endl;
    std::cout << "Top-K: " << top_k << std::endl;
    if (!genre_filter.empty()) {
        std::cout << "Filtro de género: " << genre_filter << std::endl;
    }
    if (!year_range.empty()) {
        std::cout << "Filtro de años: " << year_range << std::endl;
    }
    std::cout << std::endl;
    
    // Parsear rango de años
    int year_start = 0, year_end = 9999;
    if (!year_range.empty()) {
        size_t dash_pos = year_range.find('-');
        if (dash_pos != std::string::npos) {
            year_start = std::stoi(year_range.substr(0, dash_pos));
            year_end = std::stoi(year_range.substr(dash_pos + 1));
        }
    }
    
    // Cargar metadatos de películas
    auto movies = load_movies_info(movies_file);
    if (verbose) {
        std::cout << "✓ Cargados metadatos de " << movies.size() << " películas" << std::endl;
    }
    
    // Cargar datos para inicializar el modelo
    std::vector<Triplet> triplets = load_triplets(data_file);
    if (triplets.empty()) {
        std::cerr << "ERROR: No se pudieron cargar los datos." << std::endl;
        return 1;
    }
    
    // Inicializar sistema
    UserItemStore store(dimensions);
    store.initialize(triplets);
    
    if (verbose) {
        store.print_summary();
    }
    
    // Verificar que el usuario existe
    try {
        store.get_user_vector(user_id);
    } catch (const std::exception& e) {
        std::cerr << "ERROR: Usuario " << user_id << " no encontrado en el dataset." << std::endl;
        std::cerr << "Usuarios disponibles: ";
        
        std::set<int> available_users;
        for (const auto& triplet : triplets) {
            available_users.insert(triplet.user_id);
        }
        
        int count = 0;
        for (int uid : available_users) {
            if (count++ < 10) {
                std::cerr << uid << " ";
            }
        }
        if (available_users.size() > 10) {
            std::cerr << "... (y " << (available_users.size() - 10) << " más)";
        }
        std::cerr << std::endl;
        return 1;
    }
    
    // Crear hasher LSH
    SRPHasher hasher(dimensions, lsh_bits, 42);
    
    // Generar recomendaciones
    std::cout << "Generando recomendaciones usando Hamming Ranking..." << std::endl;
    auto recommendations = hamming_ranking_recommendations(user_id, store, hasher, movies, top_k, 
                                                         genre_filter, year_start, year_end);
    
    if (recommendations.empty()) {
        std::cout << "No se pudieron generar recomendaciones para el usuario " << user_id;
        if (!genre_filter.empty() || !year_range.empty()) {
            std::cout << " con los filtros aplicados";
        }
        std::cout << std::endl;
        return 1;
    }
    
    // Mostrar recomendaciones
    std::cout << "\n" << std::string(100, '=') << std::endl;
    std::cout << "RECOMENDACIONES PARA USUARIO " << user_id << std::endl;
    std::cout << std::string(100, '=') << std::endl;
    
    std::cout << "Rank | Item ID | Distancia | Similitud | Título                                     | Géneros" << std::endl;
    std::cout << "-----|---------|-----------|-----------|--------------------------------------------|---------" << std::endl;
    
    for (size_t i = 0; i < recommendations.size(); ++i) {
        int item_id = recommendations[i].first;
        int distance = recommendations[i].second;
        double similarity = 100.0 * (1.0 - (double)distance / lsh_bits);
        
        std::string title = "Película " + std::to_string(item_id);
        std::string genres = "";
        
        auto movie_it = movies.find(item_id);
        if (movie_it != movies.end()) {
            const Movie& movie = movie_it->second;
            title = movie.title;
            if (title.length() > 40) {
                title = title.substr(0, 37) + "...";
            }
            
            if (!movie.genres.empty()) {
                genres = movie.genres[0];
                if (movie.genres.size() > 1) {
                    genres += ", " + movie.genres[1];
                }
                if (movie.genres.size() > 2) {
                    genres += "...";
                }
            }
        }
        
        std::cout << std::setw(4) << (i + 1) << " | "
                  << std::setw(7) << item_id << " | "
                  << std::setw(9) << distance << " | "
                  << std::setw(8) << std::fixed << std::setprecision(1) << similarity << "% | "
                  << std::setw(42) << std::left << title << " | "
                  << genres << std::endl;
    }
    
    std::cout << std::endl;
    std::cout << "✅ Recomendaciones generadas exitosamente!" << std::endl;
    std::cout << "💡 Tip: Items con menor distancia Hamming son más similares al usuario" << std::endl;
    
    return 0;
}

// Función para evaluar el modelo
int evaluate_model(const std::string& data_file, const std::string& val_file,
                  const std::string& movies_file, int dimensions, int lsh_bits, bool verbose) {
    
    std::cout << "=== EVALUANDO MODELO SRPR ===" << std::endl;
    std::cout << std::endl;
    
    // Cargar metadatos
    auto movies = load_movies_info(movies_file);
    if (verbose && !movies.empty()) {
        std::cout << "✓ Cargados metadatos de " << movies.size() << " películas" << std::endl;
    }
    
    // Cargar datos
    std::vector<Triplet> training_triplets = load_triplets(data_file);
    std::vector<Triplet> validation_triplets = load_triplets(val_file);
    
    if (training_triplets.empty()) {
        std::cerr << "ERROR: No se pudieron cargar los datos de entrenamiento." << std::endl;
        return 1;
    }
    
    // Inicializar sistema
    UserItemStore store(dimensions);
    store.initialize(training_triplets);
    
    if (verbose) {
        store.print_summary();
    }
    
    SRPR_Trainer trainer(store);
    SRPR_Trainer::TrainingParams params;
    params.b_lsh_length = lsh_bits;
    
    // Evaluación en entrenamiento
    std::cout << "Evaluando en datos de entrenamiento..." << std::endl;
    double train_loss = trainer.calculate_total_loss(training_triplets, params);
    double train_precision = calculate_ranking_precision(training_triplets, store);
    
    std::cout << "✓ Pérdida (entrenamiento): " << std::fixed << std::setprecision(6) << train_loss << std::endl;
    std::cout << "✓ Precisión (entrenamiento): " << std::fixed << std::setprecision(4) 
              << (train_precision * 100) << "%" << std::endl;
    
    // Evaluación en validación
    if (!validation_triplets.empty()) {
        std::cout << "\nEvaluando en datos de validación..." << std::endl;
        double val_loss = trainer.calculate_total_loss(validation_triplets, params);
        double val_precision = calculate_ranking_precision(validation_triplets, store);
        
        std::cout << "✓ Pérdida (validación): " << std::fixed << std::setprecision(6) << val_loss << std::endl;
        std::cout << "✓ Precisión (validación): " << std::fixed << std::setprecision(4) 
                  << (val_precision * 100) << "%" << std::endl;
    }
    
    // Evaluación LSH
    std::cout << "\nEvaluando sistema LSH..." << std::endl;
    SRPHasher hasher(dimensions, lsh_bits, 42);
    
    // Generar códigos para todos los vectores
    std::set<int> unique_users, unique_items;
    for (const auto& triplet : training_triplets) {
        unique_users.insert(triplet.user_id);
        unique_items.insert(triplet.preferred_item_id);
        unique_items.insert(triplet.less_preferred_item_id);
    }
    
    std::set<std::string> unique_codes;
    for (int user_id : unique_users) {
        try {
            const Vector& vec = store.get_user_vector(user_id);
            std::string code = hasher.generate_code(vec);
            unique_codes.insert(code);
        } catch (const std::exception& e) {
            continue;
        }
    }
    
    for (int item_id : unique_items) {
        try {
            const Vector& vec = store.get_item_vector(item_id);
            std::string code = hasher.generate_code(vec);
            unique_codes.insert(code);
        } catch (const std::exception& e) {
            continue;
        }
    }
    
    double diversity = (double)unique_codes.size() / (unique_users.size() + unique_items.size());
    std::cout << "✓ Diversidad de códigos LSH: " << std::fixed << std::setprecision(3) 
              << (diversity * 100) << "%" << std::endl;
    
    // Análisis por géneros si hay metadatos
    if (!movies.empty() && verbose) {
        std::cout << "\nAnálisis por géneros en el modelo..." << std::endl;
        std::map<std::string, int> genre_items;
        
        for (int item_id : unique_items) {
            auto movie_it = movies.find(item_id);
            if (movie_it != movies.end()) {
                for (const std::string& genre : movie_it->second.genres) {
                    genre_items[genre]++;
                }
            }
        }
        
        std::cout << "Géneros representados en el modelo:" << std::endl;
        for (const auto& genre_pair : genre_items) {
            if (genre_pair.second >= 10) { // Solo géneros con al menos 10 películas
                std::cout << "  " << std::setw(15) << std::left << genre_pair.first 
                          << ": " << genre_pair.second << " películas" << std::endl;
            }
        }
    }
    
    // Resumen
    std::cout << "\n" << std::string(50, '=') << std::endl;
    std::cout << "RESUMEN DE EVALUACIÓN" << std::endl;
    std::cout << std::string(50, '=') << std::endl;
    std::cout << "📊 Usuarios: " << unique_users.size() << std::endl;
    std::cout << "📊 Items: " << unique_items.size() << std::endl;
    std::cout << "📊 Películas con metadatos: " << movies.size() << std::endl;
    std::cout << "📊 Tripletas entrenamiento: " << training_triplets.size() << std::endl;
    if (!validation_triplets.empty()) {
        std::cout << "📊 Tripletas validación: " << validation_triplets.size() << std::endl;
    }
    std::cout << "📊 Dimensiones vectores: " << dimensions << std::endl;
    std::cout << "📊 Bits LSH: " << lsh_bits << std::endl;
    std::cout << "📊 Diversidad códigos: " << std::fixed << std::setprecision(1) 
              << (diversity * 100) << "%" << std::endl;
    
    return 0;
}

int main(int argc, char* argv[]) {
    // Mostrar banner
    show_banner();
    
    // Configuración por defecto
    std::string data_file = "data/training_triplets.csv";
    std::string val_file = "data/validation_triplets.csv";
    std::string movies_file = "data/movielens/ml-20m/movies.csv";
    std::string ratings_file = "data/movielens/ml-20m/ratings.csv";
    int epochs = 20;
    double learning_rate = 0.005;
    int dimensions = 32;
    int lsh_bits = 16;
    int top_k = 10;
    int max_ratings = 500000;
    int triplets_per_user = 50;
    double min_rating_diff = 1.0;
    std::string genre_filter = "";
    std::string year_range = "";
    bool verbose = false;
    
    // Modos de operación
    bool train_mode = false;
    bool recommend_mode = false;
    bool evaluate_mode = false;
    bool analyze_mode = false;
    bool generate_data_mode = false;
    int recommend_user_id = -1;
    
    // Procesar argumentos
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        
        if (arg == "--help" || arg == "-h") {
            show_help();
            return 0;
        }
        else if (arg == "--train") {
            train_mode = true;
        }
        else if (arg == "--recommend") {
            recommend_mode = true;
            if (i + 1 < argc) {
                recommend_user_id = std::atoi(argv[++i]);
            } else {
                std::cerr << "ERROR: --recommend requiere USER_ID" << std::endl;
                return 1;
            }
        }
        else if (arg == "--evaluate") {
            evaluate_mode = true;
        }
        else if (arg == "--analyze") {
            analyze_mode = true;
        }
        else if (arg == "--generate-data") {
            generate_data_mode = true;
        }
        else if (arg == "--data-file") {
            if (i + 1 < argc) {
                data_file = argv[++i];
            } else {
                std::cerr << "ERROR: --data-file requiere un archivo" << std::endl;
                return 1;
            }
        }
        else if (arg == "--val-file") {
            if (i + 1 < argc) {
                val_file = argv[++i];
            } else {
                std::cerr << "ERROR: --val-file requiere un archivo" << std::endl;
                return 1;
            }
        }
        else if (arg == "--movies-file") {
            if (i + 1 < argc) {
                movies_file = argv[++i];
            } else {
                std::cerr << "ERROR: --movies-file requiere un archivo" << std::endl;
                return 1;
            }
        }
        else if (arg == "--ratings-file") {
            if (i + 1 < argc) {
                ratings_file = argv[++i];
            } else {
                std::cerr << "ERROR: --ratings-file requiere un archivo" << std::endl;
                return 1;
            }
        }
        else if (arg == "--epochs") {
            if (i + 1 < argc) {
                epochs = std::atoi(argv[++i]);
            } else {
                std::cerr << "ERROR: --epochs requiere un número" << std::endl;
                return 1;
            }
        }
        else if (arg == "--lr") {
            if (i + 1 < argc) {
                learning_rate = std::atof(argv[++i]);
            } else {
                std::cerr << "ERROR: --lr requiere un número" << std::endl;
                return 1;
            }
        }
        else if (arg == "--dimensions") {
            if (i + 1 < argc) {
                dimensions = std::atoi(argv[++i]);
            } else {
                std::cerr << "ERROR: --dimensions requiere un número" << std::endl;
                return 1;
            }
        }
        else if (arg == "--lsh-bits") {
            if (i + 1 < argc) {
                lsh_bits = std::atoi(argv[++i]);
            } else {
                std::cerr << "ERROR: --lsh-bits requiere un número" << std::endl;
                return 1;
            }
        }
        else if (arg == "--top-k") {
            if (i + 1 < argc) {
                top_k = std::atoi(argv[++i]);
            } else {
                std::cerr << "ERROR: --top-k requiere un número" << std::endl;
                return 1;
            }
        }
        else if (arg == "--max-ratings") {
            if (i + 1 < argc) {
                max_ratings = std::atoi(argv[++i]);
            } else {
                std::cerr << "ERROR: --max-ratings requiere un número" << std::endl;
                return 1;
            }
        }
        else if (arg == "--triplets-per-user") {
            if (i + 1 < argc) {
                triplets_per_user = std::atoi(argv[++i]);
            } else {
                std::cerr << "ERROR: --triplets-per-user requiere un número" << std::endl;
                return 1;
            }
        }
        else if (arg == "--min-rating-diff") {
            if (i + 1 < argc) {
                min_rating_diff = std::atof(argv[++i]);
            } else {
                std::cerr << "ERROR: --min-rating-diff requiere un número" << std::endl;
                return 1;
            }
        }
        else if (arg == "--genre") {
            if (i + 1 < argc) {
                genre_filter = argv[++i];
            } else {
                std::cerr << "ERROR: --genre requiere un género" << std::endl;
                return 1;
            }
        }
        else if (arg == "--year-range") {
            if (i + 1 < argc) {
                year_range = argv[++i];
            } else {
                std::cerr << "ERROR: --year-range requiere un rango (ej: 2000-2010)" << std::endl;
                return 1;
            }
        }
        else if (arg == "--verbose") {
            verbose = true;
        }
        else {
            std::cerr << "ERROR: Argumento desconocido: " << arg << std::endl;
            show_help();
            return 1;
        }
    }
    
    // Validar que se seleccionó un modo
    if (!train_mode && !recommend_mode && !evaluate_mode && !analyze_mode && !generate_data_mode) {
        std::cout << "Por favor, selecciona un modo de operación:" << std::endl;
        std::cout << "  --generate-data    Para generar dataset desde MovieLens raw" << std::endl;
        std::cout << "  --analyze          Para analizar el dataset MovieLens" << std::endl;
        std::cout << "  --train            Para entrenar el modelo" << std::endl;
        std::cout << "  --recommend        Para generar recomendaciones" << std::endl;
        std::cout << "  --evaluate         Para evaluar el modelo" << std::endl;
        std::cout << std::endl;
        std::cout << "Usa --help para ver todas las opciones disponibles." << std::endl;
        return 1;
    }
    
    // Ejecutar modo seleccionado
    try {
        if (generate_data_mode) {
            return generate_training_data(ratings_file, max_ratings, triplets_per_user, min_rating_diff, verbose);
        }
        else if (analyze_mode) {
            return analyze_dataset(ratings_file, movies_file, verbose);
        }
        else if (train_mode) {
            return train_model(data_file, val_file, epochs, learning_rate, 
                             dimensions, lsh_bits, verbose);
        }
        else if (recommend_mode) {
            return generate_recommendations(recommend_user_id, top_k, dimensions, 
                                          lsh_bits, data_file, movies_file, genre_filter, year_range, verbose);
        }
        else if (evaluate_mode) {
            return evaluate_model(data_file, val_file, movies_file, dimensions, lsh_bits, verbose);
        }
    }
    catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}