#ifndef SRPR_TRAINER_H
#define SRPR_TRAINER_H

#include "UserItemStore.h"
#include "Triplet.h"
#include "LSH.h"
#include <vector>
#include <cmath>

class SRPR_Trainer {
public:
    struct TrainingParams {
        int epochs = 10;
        double learning_rate = 0.01;
        int b_lsh_length = 16; // Número de funciones hash a considerar en el entrenamiento
        double regularization = 0.001; // Factor de regularización
        bool verbose = true; // Mostrar progreso durante entrenamiento
        int validation_freq = 5; // Frecuencia de validación (cada N epochs)
    };

    struct TrainingStats {
        std::vector<double> epoch_losses;
        std::vector<double> validation_scores;
        double final_loss = 0.0;
        double training_time_ms = 0.0;
        int total_updates = 0;
        bool converged = false;
    };

    SRPR_Trainer(UserItemStore& store);

    // El método principal de entrenamiento
    TrainingStats train(const std::vector<Triplet>& training_triplets, 
                       const TrainingParams& params,
                       const std::vector<Triplet>& validation_triplets = {});

    // Métodos de utilidad para análisis
    double evaluate_triplet(const Triplet& triplet, const TrainingParams& params) const;
    double calculate_total_loss(const std::vector<Triplet>& triplets, const TrainingParams& params) const;
    
    // Métodos para debugging y análisis
    void print_training_summary(const TrainingStats& stats) const;
    std::vector<double> get_gradient_norms(const std::vector<Triplet>& sample_triplets, 
                                         const TrainingParams& params) const;

private:
    UserItemStore& store;

    // Función para calcular p_ui, la probabilidad de colisión para SRP-LSH
    double calculate_p_srp(const Vector& v1, const Vector& v2) const;
    
    // Función para calcular gamma según Ecuación 5 del paper
    double calculate_gamma(double p_ui, double p_uj, int b) const;
    
    // Función para calcular la derivada de gamma respecto a p_ui y p_uj
    std::pair<double, double> calculate_gamma_derivatives(double p_ui, double p_uj, int b) const;
    
    // Función para calcular la derivada de p_srp respecto a los vectores
    std::pair<Vector, Vector> calculate_p_srp_derivatives(const Vector& v1, const Vector& v2) const;
    
    // Función principal de cálculo de gradientes
    void compute_gradients(const Triplet& triplet, const TrainingParams& params,
                          Vector& grad_xu, Vector& grad_yi, Vector& grad_yj) const;
    
    // Función para actualizar vectores con gradientes
    void update_vectors(const Triplet& triplet, const Vector& grad_xu, 
                       const Vector& grad_yi, const Vector& grad_yj,
                       const TrainingParams& params);
    
    // Funciones de utilidad matemática
    double phi(double x) const; // Función de distribución normal estándar
    double phi_prime(double x) const; // Derivada de phi (densidad normal estándar)
    double safe_acos(double x) const; // acos seguro para evitar errores numéricos
    
    // Función para aplicar regularización
    void apply_regularization(Vector& vector, double reg_factor, double learning_rate) const;
    
    // Función para verificar convergencia
    bool check_convergence(const std::vector<double>& losses, double tolerance = 1e-6) const;
};

#endif // SRPR_TRAINER_H