#define _USE_MATH_DEFINES
#include <cmath>
#include "../include/SRPR_Trainer.h"
#include <iostream>
#include <numeric>
#include <algorithm>
#include <chrono>
#include <iomanip>

// Función de utilidad para calcular la norma de un vector
static double norm(const Vector& v) {
    return std::sqrt(std::inner_product(v.begin(), v.end(), v.begin(), 0.0));
}

// Función de utilidad para el producto punto
static double dot_product(const Vector& v1, const Vector& v2) {
    return std::inner_product(v1.begin(), v1.end(), v2.begin(), 0.0);
}

SRPR_Trainer::SRPR_Trainer(UserItemStore& data_store) : store(data_store) {}

double SRPR_Trainer::calculate_p_srp(const Vector& v1, const Vector& v2) const {
    double n1 = norm(v1);
    double n2 = norm(v2);
    
    // Manejar casos extremos
    if (n1 < 1e-12 || n2 < 1e-12) {
        return 0.5; // Probabilidad neutral si algún vector es casi cero
    }

    double cosine_sim = dot_product(v1, v2) / (n1 * n2);
    
    // Asegurar que el valor esté en [-1, 1] para acos
    cosine_sim = std::max(-1.0, std::min(1.0, cosine_sim));
    
    // p_ui^srp = (1 - arccos(cosine_similarity) / π)
    // Esto es equivalente a (π - arccos(cos_sim)) / π
    return 1.0 - (std::acos(cosine_sim) / M_PI);
}

double SRPR_Trainer::calculate_gamma(double p_ui, double p_uj, int b) const {
    // Evitar división por cero y valores extremos
    p_ui = std::max(1e-12, std::min(1.0 - 1e-12, p_ui));
    p_uj = std::max(1e-12, std::min(1.0 - 1e-12, p_uj));
    
    double numerator = p_uj - p_ui;
    double variance_ui = p_ui * (1.0 - p_ui);
    double variance_uj = p_uj * (1.0 - p_uj);
    double denominator = std::sqrt(variance_ui + variance_uj);
    
    if (denominator < 1e-12) {
        return 0.0; // Gamma neutral si la varianza es muy pequeña
    }
    
    return numerator / denominator;
}

std::pair<double, double> SRPR_Trainer::calculate_gamma_derivatives(double p_ui, double p_uj, int b) const {
    // Clamp probabilities para estabilidad numérica
    p_ui = std::max(1e-12, std::min(1.0 - 1e-12, p_ui));
    p_uj = std::max(1e-12, std::min(1.0 - 1e-12, p_uj));
    
    double var_ui = p_ui * (1.0 - p_ui);
    double var_uj = p_uj * (1.0 - p_uj);
    double sigma = std::sqrt(var_ui + var_uj);
    
    if (sigma < 1e-12) {
        return {0.0, 0.0};
    }
    
    double numerator = p_uj - p_ui;
    
    // Derivada respecto a p_ui
    double dgamma_dpui_num = -sigma - numerator * (1.0 - 2.0 * p_ui) / (2.0 * sigma);
    double dgamma_dpui = dgamma_dpui_num / (sigma * sigma);
    
    // Derivada respecto a p_uj
    double dgamma_dpuj_num = sigma - numerator * (1.0 - 2.0 * p_uj) / (2.0 * sigma);
    double dgamma_dpuj = dgamma_dpuj_num / (sigma * sigma);
    
    return {dgamma_dpui, dgamma_dpuj};
}

std::pair<Vector, Vector> SRPR_Trainer::calculate_p_srp_derivatives(const Vector& v1, const Vector& v2) const {
    int d = v1.size();
    Vector grad_v1(d, 0.0);
    Vector grad_v2(d, 0.0);
    
    double n1 = norm(v1);
    double n2 = norm(v2);
    
    if (n1 < 1e-12 || n2 < 1e-12) {
        return {grad_v1, grad_v2}; // Gradientes cero si normas muy pequeñas
    }
    
    double dot_prod = dot_product(v1, v2);
    double cosine_sim = dot_prod / (n1 * n2);
    cosine_sim = std::max(-1.0, std::min(1.0, cosine_sim));
    
    // dp/d(cos_sim) = 1/π * 1/sqrt(1 - cos_sim²)
    double sin_theta = std::sqrt(1.0 - cosine_sim * cosine_sim);
    if (sin_theta < 1e-12) {
        return {grad_v1, grad_v2}; // Gradientes cero si vectores son paralelos
    }
    
    double dp_dcos = 1.0 / (M_PI * sin_theta);
    
    // Derivadas del coseno respecto a los vectores
    for (int i = 0; i < d; ++i) {
        // Derivada respecto a v1[i]
        double dcos_dv1i = v2[i] / (n1 * n2) - (cosine_sim * v1[i]) / (n1 * n1);
        grad_v1[i] = dp_dcos * dcos_dv1i;
        
        // Derivada respecto a v2[i]
        double dcos_dv2i = v1[i] / (n1 * n2) - (cosine_sim * v2[i]) / (n2 * n2);
        grad_v2[i] = dp_dcos * dcos_dv2i;
    }
    
    return {grad_v1, grad_v2};
}

void SRPR_Trainer::compute_gradients(const Triplet& triplet, const TrainingParams& params,
                                    Vector& grad_xu, Vector& grad_yi, Vector& grad_yj) const {
    
    const Vector& xu = store.get_user_vector(triplet.user_id);
    const Vector& yi = store.get_item_vector(triplet.preferred_item_id);
    const Vector& yj = store.get_item_vector(triplet.less_preferred_item_id);
    
    int d = xu.size();
    grad_xu.assign(d, 0.0);
    grad_yi.assign(d, 0.0);
    grad_yj.assign(d, 0.0);
    
    // Calcular probabilidades de colisión
    double p_ui = calculate_p_srp(xu, yi);
    double p_uj = calculate_p_srp(xu, yj);
    
    // Calcular gamma
    double gamma = calculate_gamma(p_ui, p_uj, params.b_lsh_length);
    double sqrt_b_gamma = std::sqrt(params.b_lsh_length) * gamma;
    
    // Calcular derivadas de gamma
    std::pair<double, double> gamma_derivs = calculate_gamma_derivatives(p_ui, p_uj, params.b_lsh_length);
    double dgamma_dpui = gamma_derivs.first;
    double dgamma_dpuj = gamma_derivs.second;
    
    // Calcular phi(sqrt(b) * gamma) y phi'(sqrt(b) * gamma)
    double phi_val = phi(sqrt_b_gamma);
    double phi_prime_val = phi_prime(sqrt_b_gamma);
    
    if (phi_val < 1e-12) {
        return; // Evitar división por cero
    }
    
    // Factor común: phi'(sqrt(b) * gamma) / phi(sqrt(b) * gamma) * sqrt(b)
    double common_factor = (phi_prime_val / phi_val) * std::sqrt(params.b_lsh_length);
    
    // Calcular derivadas de p_ui y p_uj respecto a los vectores
    std::pair<Vector, Vector> dpui_derivs = calculate_p_srp_derivatives(xu, yi);
    Vector dpui_dxu = dpui_derivs.first;
    Vector dpui_dyi = dpui_derivs.second;
    
    std::pair<Vector, Vector> dpuj_derivs = calculate_p_srp_derivatives(xu, yj);
    Vector dpuj_dxu = dpuj_derivs.first;
    Vector dpuj_dyj = dpuj_derivs.second;
    
    // Aplicar regla de la cadena para obtener gradientes finales
    for (int k = 0; k < d; ++k) {
        // Gradiente respecto a xu
        grad_xu[k] = common_factor * (dgamma_dpui * dpui_dxu[k] + dgamma_dpuj * dpuj_dxu[k]);
        
        // Gradiente respecto a yi
        grad_yi[k] = common_factor * dgamma_dpui * dpui_dyi[k];
        
        // Gradiente respecto a yj
        grad_yj[k] = common_factor * dgamma_dpuj * dpuj_dyj[k];
    }
}

void SRPR_Trainer::update_vectors(const Triplet& triplet, const Vector& grad_xu, 
                                 const Vector& grad_yi, const Vector& grad_yj,
                                 const TrainingParams& params) {
    
    Vector& xu = store.get_user_vector(triplet.user_id);
    Vector& yi = store.get_item_vector(triplet.preferred_item_id);
    Vector& yj = store.get_item_vector(triplet.less_preferred_item_id);
    
    int d = xu.size();
    
    // Actualizar vectores usando gradiente ascendente (maximizar log-likelihood)
    for (int k = 0; k < d; ++k) {
        xu[k] += params.learning_rate * grad_xu[k];
        yi[k] += params.learning_rate * grad_yi[k];
        yj[k] += params.learning_rate * grad_yj[k];
    }
    
    // Aplicar regularización
    apply_regularization(xu, params.regularization, params.learning_rate);
    apply_regularization(yi, params.regularization, params.learning_rate);
    apply_regularization(yj, params.regularization, params.learning_rate);
}

double SRPR_Trainer::phi(double x) const {
    // Función de distribución acumulativa normal estándar usando aproximación
    return 0.5 * (1.0 + std::erf(x / std::sqrt(2.0)));
}

double SRPR_Trainer::phi_prime(double x) const {
    // Densidad de probabilidad normal estándar
    return (1.0 / std::sqrt(2.0 * M_PI)) * std::exp(-0.5 * x * x);
}

double SRPR_Trainer::safe_acos(double x) const {
    return std::acos(std::max(-1.0, std::min(1.0, x)));
}

void SRPR_Trainer::apply_regularization(Vector& vector, double reg_factor, double learning_rate) const {
    for (double& val : vector) {
        val -= learning_rate * reg_factor * val;
    }
}

bool SRPR_Trainer::check_convergence(const std::vector<double>& losses, double tolerance) const {
    if (losses.size() < 3) return false;
    
    size_t n = losses.size();
    double recent_change = std::abs(losses[n-1] - losses[n-2]);
    return recent_change < tolerance;
}

double SRPR_Trainer::evaluate_triplet(const Triplet& triplet, const TrainingParams& params) const {
    const Vector& xu = store.get_user_vector(triplet.user_id);
    const Vector& yi = store.get_item_vector(triplet.preferred_item_id);
    const Vector& yj = store.get_item_vector(triplet.less_preferred_item_id);
    
    double p_ui = calculate_p_srp(xu, yi);
    double p_uj = calculate_p_srp(xu, yj);
    double gamma = calculate_gamma(p_ui, p_uj, params.b_lsh_length);
    double sqrt_b_gamma = std::sqrt(params.b_lsh_length) * gamma;
    
    return std::log(phi(sqrt_b_gamma) + 1e-12); // Evitar log(0)
}

double SRPR_Trainer::calculate_total_loss(const std::vector<Triplet>& triplets, const TrainingParams& params) const {
    double total_loss = 0.0;
    
    for (const auto& triplet : triplets) {
        total_loss += evaluate_triplet(triplet, params);
    }
    
    return total_loss / triplets.size();
}

SRPR_Trainer::TrainingStats SRPR_Trainer::train(const std::vector<Triplet>& training_triplets, 
                                               const TrainingParams& params,
                                               const std::vector<Triplet>& validation_triplets) {
    
    TrainingStats stats;
    auto start_time = std::chrono::high_resolution_clock::now();
    
    if (params.verbose) {
        std::cout << "=== Iniciando Entrenamiento SRPR ===" << std::endl;
        std::cout << "Configuración:" << std::endl;
        std::cout << "  - Epochs: " << params.epochs << std::endl;
        std::cout << "  - Learning rate: " << params.learning_rate << std::endl;
        std::cout << "  - LSH bits: " << params.b_lsh_length << std::endl;
        std::cout << "  - Regularización: " << params.regularization << std::endl;
        std::cout << "  - Tripletas entrenamiento: " << training_triplets.size() << std::endl;
        std::cout << "  - Tripletas validación: " << validation_triplets.size() << std::endl;
        std::cout << std::endl;
    }
    
    for (int epoch = 0; epoch < params.epochs; ++epoch) {
        double epoch_loss = 0.0;
        int updates = 0;
        
        auto epoch_start = std::chrono::high_resolution_clock::now();
        
        // Entrenar con todas las tripletas
        for (const auto& triplet : training_triplets) {
            Vector grad_xu, grad_yi, grad_yj;
            
            // Calcular gradientes
            compute_gradients(triplet, params, grad_xu, grad_yi, grad_yj);
            
            // Actualizar vectores
            update_vectors(triplet, grad_xu, grad_yi, grad_yj, params);
            
            // Acumular pérdida
            epoch_loss += evaluate_triplet(triplet, params);
            updates++;
        }
        
        epoch_loss /= training_triplets.size();
        stats.epoch_losses.push_back(epoch_loss);
        stats.total_updates += updates;
        
        auto epoch_end = std::chrono::high_resolution_clock::now();
        auto epoch_duration = std::chrono::duration_cast<std::chrono::milliseconds>(epoch_end - epoch_start);
        
        // Evaluación en validación
        double validation_loss = 0.0;
        if (!validation_triplets.empty() && (epoch + 1) % params.validation_freq == 0) {
            validation_loss = calculate_total_loss(validation_triplets, params);
            stats.validation_scores.push_back(validation_loss);
        }
        
        // Mostrar progreso
        if (params.verbose) {
            std::cout << "Epoch " << std::setw(3) << (epoch + 1) << "/" << params.epochs 
                      << " | Loss: " << std::fixed << std::setprecision(6) << epoch_loss
                      << " | Time: " << std::setw(4) << epoch_duration.count() << "ms";
            
            if (!validation_triplets.empty() && (epoch + 1) % params.validation_freq == 0) {
                std::cout << " | Val Loss: " << std::fixed << std::setprecision(6) << validation_loss;
            }
            std::cout << std::endl;
        }
        
        // Verificar convergencia
        if (check_convergence(stats.epoch_losses)) {
            if (params.verbose) {
                std::cout << "Convergencia detectada en epoch " << (epoch + 1) << std::endl;
            }
            stats.converged = true;
            break;
        }
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    stats.final_loss = stats.epoch_losses.back();
    stats.training_time_ms = total_duration.count();
    
    if (params.verbose) {
        std::cout << "\n=== Entrenamiento Completado ===" << std::endl;
        print_training_summary(stats);
    }
    
    return stats;
}

void SRPR_Trainer::print_training_summary(const TrainingStats& stats) const {
    std::cout << "Resumen del entrenamiento:" << std::endl;
    std::cout << "  - Pérdida final: " << std::fixed << std::setprecision(6) << stats.final_loss << std::endl;
    std::cout << "  - Tiempo total: " << stats.training_time_ms << " ms" << std::endl;
    std::cout << "  - Total de actualizaciones: " << stats.total_updates << std::endl;
    std::cout << "  - Convergió: " << (stats.converged ? "Sí" : "No") << std::endl;
    
    if (stats.epoch_losses.size() >= 2) {
        double improvement = stats.epoch_losses[0] - stats.final_loss;
        std::cout << "  - Mejora total: " << std::fixed << std::setprecision(6) << improvement << std::endl;
    }
    
    if (!stats.validation_scores.empty()) {
        double best_val = *std::min_element(stats.validation_scores.begin(), stats.validation_scores.end());
        std::cout << "  - Mejor pérdida validación: " << std::fixed << std::setprecision(6) << best_val << std::endl;
    }
    
    std::cout << "  - Velocidad: " << (stats.total_updates * 1000.0 / stats.training_time_ms) << " actualizaciones/s" << std::endl;
}

std::vector<double> SRPR_Trainer::get_gradient_norms(const std::vector<Triplet>& sample_triplets, 
                                                     const TrainingParams& params) const {
    std::vector<double> norms;
    
    for (const auto& triplet : sample_triplets) {
        Vector grad_xu, grad_yi, grad_yj;
        compute_gradients(triplet, params, grad_xu, grad_yi, grad_yj);
        
        double norm_xu = norm(grad_xu);
        double norm_yi = norm(grad_yi);
        double norm_yj = norm(grad_yj);
        
        norms.push_back(norm_xu);
        norms.push_back(norm_yi);
        norms.push_back(norm_yj);
    }
    
    return norms;
}