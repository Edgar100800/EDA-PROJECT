# Proyecto: Implementación de SRPR en C++

Este documento sirve como guía para un agente de IA para implementar el paper "Stochastically Robust Personalized Ranking for LSH Recommendation Retrieval" en C++.

## 1. Visión General del Proyecto

El objetivo es construir un sistema de recomendación que utiliza el framework SRPR. El desarrollo se hará de manera incremental, enfocándose en construir y probar cada estructura de datos y componente lógico de forma independiente antes de integrarlos.

El flujo de trabajo sigue las dos fases descritas en el paper:

- **Fase I (Aprendizaje Offline)**: Entrenar un modelo para aprender vectores de usuario e ítem que sean robustos a la aleatoriedad de LSH.
- **Fase II (Recuperación Online)**: Usar los vectores aprendidos junto con LSH para recuperar eficientemente las k mejores recomendaciones.

## 2. Estructura de Directorios

Organiza el proyecto de la siguiente manera para mantener el código limpio y modular:

```
SRPR_Project/
├── include/              # Archivos de cabecera (.h, .hpp)
│   ├── Triplet.h
│   ├── VectorUtils.h
│   ├── UserItemStore.h
│   ├── LSH.h
│   ├── SRPR_Trainer.h
│   └── ItemIndex.h
├── src/                  # Archivos de implementación (.cpp)
│   ├── UserItemStore.cpp
│   ├── LSH.cpp
│   ├── SRPR_Trainer.cpp
│   └── ItemIndex.cpp
├── tests/                # Archivos `main` para pruebas unitarias de cada componente
│   ├── main_test_triplet.cpp
│   ├── main_test_useritemstore.cpp
│   ├── main_test_lsh.cpp
│   ├── main_test_srpr_trainer.cpp
│   └── main_test_itemindex.cpp
├── data/                 # Datos de ejemplo (e.g., tripletas en CSV)
│   └── triplets.csv
└── main.cpp              # Archivo principal para la integración final
```

## 3. Plan de Implementación Incremental

A continuación se detallan los pasos para construir y probar cada componente. Para cada paso, se proporciona el código del archivo de cabecera y un main de prueba.

### Paso 1: Estructura de Tripletas

Esta es la estructura de datos más básica que representa una preferencia observada (usuario, item_preferido, item_no_preferido).

#### include/Triplet.h

```cpp
#ifndef TRIPLET_H
#define TRIPLET_H

#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>

// Representa una única observación de preferencia: usuario u prefiere item i sobre item j.
struct Triplet {
    int user_id;
    int preferred_item_id;
    int less_preferred_item_id;
};

// Función de utilidad para cargar tripletas desde un archivo CSV.
static std::vector<Triplet> load_triplets(const std::string& filepath) {
    std::vector<Triplet> triplets;
    std::ifstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "Error: No se pudo abrir el archivo " << filepath << std::endl;
        return triplets;
    }

    std::string line;
    // Opcional: Omitir la cabecera del CSV
    // std::getline(file, line);

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string cell;

        Triplet t;

        std::getline(ss, cell, ',');
        t.user_id = std::stoi(cell);

        std::getline(ss, cell, ',');
        t.preferred_item_id = std::stoi(cell);

        std::getline(ss, cell, ',');
        t.less_preferred_item_id = std::stoi(cell);

        triplets.push_back(t);
    }

    file.close();
    return triplets;
}

#endif // TRIPLET_H
```

#### tests/main_test_triplet.cpp

```cpp
#include "../include/Triplet.h"
#include <iostream>
#include <vector>

int main() {
    std::cout << "--- Probando Carga de Tripletas ---" << std::endl;

    // Crear un archivo CSV de prueba
    std::ofstream test_file("triplets_test.csv");
    test_file << "101,1,2\n";
    test_file << "101,1,3\n";
    test_file << "102,5,6\n";
    test_file.close();

    std::vector<Triplet> triplets = load_triplets("triplets_test.csv");

    if (triplets.empty()) {
        std::cerr << "Prueba fallida: No se cargaron las tripletas." << std::endl;
        return 1;
    }

    std::cout << "Se cargaron " << triplets.size() << " tripletas." << std::endl;
    for (const auto& t : triplets) {
        std::cout << "Usuario: " << t.user_id
                  << ", Prefiere: " << t.preferred_item_id
                  << " sobre: " << t.less_preferred_item_id << std::endl;
    }

    std::cout << "Prueba de Tripletas completada con éxito." << std::endl;
    return 0;
}
```

### Paso 2: Almacén de Vectores de Usuario e Ítem

Esta clase gestionará las matrices X (usuarios) e Y (ítems).

#### include/UserItemStore.h

```cpp
#ifndef USER_ITEM_STORE_H
#define USER_ITEM_STORE_H

#include <vector>
#include <unordered_map>
#include <random>
#include "Triplet.h"

using Vector = std::vector<double>;

class UserItemStore {
public:
    UserItemStore(int dimensions);

    // Inicializa los vectores para todos los usuarios e ítems encontrados en las tripletas.
    void initialize(const std::vector<Triplet>& triplets);

    // Obtiene una referencia modificable a un vector.
    Vector& get_user_vector(int user_id);
    Vector& get_item_vector(int item_id);

    // Obtiene una referencia constante.
    const Vector& get_user_vector(int user_id) const;
    const Vector& get_item_vector(int item_id) const;

    const std::unordered_map<int, Vector>& get_all_item_vectors() const;

    void print_summary() const;

private:
    int d; // Dimensionalidad de los vectores latentes
    std::unordered_map<int, Vector> user_vectors; // Matriz X
    std::unordered_map<int, Vector> item_vectors; // Matriz Y

    // Generador de números aleatorios para la inicialización.
    std::mt19937 rng;
    std::normal_distribution<double> dist;
};

#endif // USER_ITEM_STORE_H
```

#### src/UserItemStore.cpp

```cpp
#include "../include/UserItemStore.h"
#include <iostream>
#include <set>

UserItemStore::UserItemStore(int dimensions) : d(dimensions), rng(std::random_device{}()), dist(0.0, 0.1) {}

void UserItemStore::initialize(const std::vector<Triplet>& triplets) {
    std::set<int> user_ids;
    std::set<int> item_ids;

    for (const auto& t : triplets) {
        user_ids.insert(t.user_id);
        item_ids.insert(t.preferred_item_id);
        item_ids.insert(t.less_preferred_item_id);
    }

    for (int id : user_ids) {
        user_vectors[id] = Vector(d);
        for (int i = 0; i < d; ++i) {
            user_vectors[id][i] = dist(rng);
        }
    }

    for (int id : item_ids) {
        item_vectors[id] = Vector(d);
        for (int i = 0; i < d; ++i) {
            item_vectors[id][i] = dist(rng);
        }
    }
}

Vector& UserItemStore::get_user_vector(int user_id) {
    return user_vectors.at(user_id);
}

Vector& UserItemStore::get_item_vector(int item_id) {
    return item_vectors.at(item_id);
}

const Vector& UserItemStore::get_user_vector(int user_id) const {
    return user_vectors.at(user_id);
}

const Vector& UserItemStore::get_item_vector(int item_id) const {
    return item_vectors.at(item_id);
}

const std::unordered_map<int, Vector>& UserItemStore::get_all_item_vectors() const {
    return item_vectors;
}

void UserItemStore::print_summary() const {
    std::cout << "UserItemStore Resumen:" << std::endl;
    std::cout << "  - " << user_vectors.size() << " usuarios." << std::endl;
    std::cout << "  - " << item_vectors.size() << " items." << std::endl;
    std::cout << "  - Dimensiones: " << d << std::endl;
}
```

#### tests/main_test_useritemstore.cpp

```cpp
#include "../include/UserItemStore.h"
#include "../include/Triplet.h"
#include <iostream>

int main() {
    std::cout << "--- Probando UserItemStore ---" << std::endl;
    int dimensions = 8;

    std::vector<Triplet> triplets = {
        {101, 1, 2},
        {101, 1, 3},
        {102, 5, 6}
    };

    UserItemStore store(dimensions);
    store.initialize(triplets);
    store.print_summary();

    try {
        Vector& user_vec = store.get_user_vector(101);
        std::cout << "Vector original para usuario 101[0]: " << user_vec[0] << std::endl;

        // Modificamos el vector
        user_vec[0] = 99.9;

        const Vector& updated_vec = store.get_user_vector(101);
        std::cout << "Vector actualizado para usuario 101[0]: " << updated_vec[0] << std::endl;

        if (updated_vec[0] != 99.9) {
             std::cerr << "Prueba fallida: La actualización del vector no persistió." << std::endl;
             return 1;
        }

    } catch (const std::out_of_range& e) {
        std::cerr << "Prueba fallida: ID no encontrado. " << e.what() << std::endl;
        return 1;
    }

    std::cout << "Prueba de UserItemStore completada con éxito." << std::endl;
    return 0;
}
```

### Paso 3: LSH Hasher

Implementaremos una clase base abstracta LSH y una implementación concreta para SRP-LSH (Sign Random Projection).

#### include/LSH.h

```cpp
#ifndef LSH_H
#define LSH_H

#include <vector>
#include <string>
#include <random>
#include <numeric> // Para std::inner_product

using Vector = std::vector<double>;

// Clase base abstracta para familias LSH
class LSH {
public:
    LSH(int dimensions, int num_hashes) : d(dimensions), b(num_hashes) {}
    virtual ~LSH() = default;

    // Genera un código binario de longitud b para un vector dado.
    std::string generate_code(const Vector& vec) const;

protected:
    // Genera un único bit de hash (0 o 1). Debe ser implementado por las clases hijas.
    virtual char hash_to_bit(const Vector& vec, int hash_function_index) const = 0;

    int d; // Dimensiones del vector
    int b; // Número de funciones de hash (longitud del código)
};

// Implementación de Sign Random Projection (SRP)
class SRPHasher : public LSH {
public:
    SRPHasher(int dimensions, int num_hashes);

protected:
    char hash_to_bit(const Vector& vec, int hash_function_index) const override;

private:
    std::vector<Vector> random_vectors; // Parámetros 'a' de las funciones de hash
};

#endif // LSH_H
```

#### src/LSH.cpp

```cpp
#include "../include/LSH.h"

std::string LSH::generate_code(const Vector& vec) const {
    std::string code = "";
    for (int i = 0; i < b; ++i) {
        code += hash_to_bit(vec, i);
    }
    return code;
}

// --- SRPHasher Implementación ---

SRPHasher::SRPHasher(int dimensions, int num_hashes) : LSH(dimensions, num_hashes) {
    std::mt19937 rng(std::random_device{}());
    std::normal_distribution<double> dist(0.0, 1.0);

    random_vectors.resize(num_hashes);
    for (int i = 0; i < num_hashes; ++i) {
        random_vectors[i].resize(dimensions);
        for (int j = 0; j < dimensions; ++j) {
            random_vectors[i][j] = dist(rng);
        }
    }
}

char SRPHasher::hash_to_bit(const Vector& vec, int hash_function_index) const {
    double projection = std::inner_product(vec.begin(), vec.end(), random_vectors[hash_function_index].begin(), 0.0);
    // h(x) = sign(a^T * x). Mapeamos a '1' si es >= 0, y a '0' si es < 0.
    return (projection >= 0) ? '1' : '0';
}
```

#### tests/main_test_lsh.cpp

```cpp
#include "../include/LSH.h"
#include <iostream>

int main() {
    std::cout << "--- Probando LSH (SRPHasher) ---" << std::endl;
    int dimensions = 16;
    int num_hashes = 8;

    SRPHasher hasher(dimensions, num_hashes);

    Vector test_vector(dimensions, 1.0);
    Vector test_vector_neg(dimensions, -1.0);

    std::string code1 = hasher.generate_code(test_vector);
    std::string code2 = hasher.generate_code(test_vector_neg);
    std::string code3 = hasher.generate_code(test_vector); // Debería ser igual a code1

    std::cout << "Vector de 1.0s   -> Código: " << code1 << std::endl;
    std::cout << "Vector de -1.0s  -> Código: " << code2 << std::endl;
    std::cout << "Vector de 1.0s (2) -> Código: " << code3 << std::endl;

    if (code1.length() != num_hashes || code2.length() != num_hashes) {
        std::cerr << "Prueba fallida: La longitud del código es incorrecta." << std::endl;
        return 1;
    }

    if (code1 != code3) {
        std::cerr << "Prueba fallida: El hash no es determinista para la misma entrada." << std::endl;
        return 1;
    }

    std::cout << "Prueba de LSH completada con éxito." << std::endl;
    return 0;
}
```

### Paso 4: Entrenador SRPR (Fase I)

Este componente encapsula la lógica de entrenamiento. Por ahora, será un esqueleto que muestra el flujo, con un TODO para la compleja lógica del gradiente, que es la tarea principal a resolver por el agente.

#### include/SRPR_Trainer.h

```cpp
#ifndef SRPR_TRAINER_H
#define SRPR_TRAINER_H

#include "UserItemStore.h"
#include "Triplet.h"
#include "LSH.h"

class SRPR_Trainer {
public:
    struct TrainingParams {
        int epochs = 10;
        double learning_rate = 0.01;
        int b_lsh_length = 16; // Número de funciones hash a considerar en el entrenamiento
    };

    SRPR_Trainer(UserItemStore& store);

    // El método principal de entrenamiento.
    void train(const std::vector<Triplet>& triplets, const TrainingParams& params);

private:
    UserItemStore& store;

    // Función para calcular p_ui, la probabilidad de colisión para SRP-LSH.
    double calculate_p_srp(const Vector& v1, const Vector& v2) const;
};

#endif // SRPR_TRAINER_H
```

#### src/SRPR_Trainer.cpp

```cpp
#define _USE_MATH_DEFINES
#include <cmath>
#include "../include/SRPR_Trainer.h"
#include <iostream>
#include <numeric> // Para inner_product
#include <algorithm> // Para std::max/min

// Función de utilidad para calcular la norma de un vector
static double norm(const Vector& v) {
    return std::sqrt(std::inner_product(v.begin(), v.end(), v.begin(), 0.0));
}

SRPR_Trainer::SRPR_Trainer(UserItemStore& data_store) : store(data_store) {}

double SRPR_Trainer::calculate_p_srp(const Vector& v1, const Vector& v2) const {
    double n1 = norm(v1);
    double n2 = norm(v2);
    if (n1 == 0.0 || n2 == 0.0) return 0.5; // Evitar división por cero

    double cosine_sim = std::inner_product(v1.begin(), v1.end(), v2.begin(), 0.0) / (n1 * n2);
    // Asegurarse de que el valor esté en [-1, 1] para acos
    cosine_sim = std::max(-1.0, std::min(1.0, cosine_sim));
    // p_ui^srp = (1/pi) * arccos(cosine_similarity)
    return std::acos(cosine_sim) / M_PI;
}

void SRPR_Trainer::train(const std::vector<Triplet>& triplets, const TrainingParams& params) {
    std::cout << "Iniciando entrenamiento SRPR..." << std::endl;
    for (int epoch = 0; epoch < params.epochs; ++epoch) {
        int count = 0;
        for (const auto& t : triplets) {
            Vector& xu = store.get_user_vector(t.user_id);
            Vector& yi = store.get_item_vector(t.preferred_item_id);
            Vector& yj = store.get_item_vector(t.less_preferred_item_id);

            // TAREA PRINCIPAL PARA EL AGENTE:
            // TODO: Implementar el cálculo del gradiente y la actualización de vectores.
            // 1. Calcular p_ui y p_uj usando calculate_p_srp.
            // double p_ui = calculate_p_srp(xu, yi);
            // double p_uj = calculate_p_srp(xu, yj);

            // 2. Calcular gamma_uij (Ecuación 5 del paper).
            // double gamma_numerator = p_uj - p_ui;
            // double gamma_denominator = std::sqrt(p_uj * (1 - p_uj) + p_ui * (1 - p_ui));
            // double gamma = (gamma_denominator > 1e-9) ? gamma_numerator / gamma_denominator : 0.0;

            // 3. Calcular el gradiente de la función de verosimilitud log(Φ(sqrt(b)*gamma))
            //    con respecto a xu, yi, yj. Esto es la parte más compleja y requiere aplicar la regla de la cadena.
            //    ∇L = (Φ'(sqrt(b)γ) / Φ(sqrt(b)γ)) * sqrt(b) * ∇γ
            //    Necesitarás la derivada de Φ (función de densidad de probabilidad normal estándar)
            //    y la derivada de γ (que a su vez depende de las derivadas de p_ui y p_uj).

            // 4. Actualizar los vectores.
            //    xu += params.learning_rate * gradient_xu;
            //    yi += params.learning_rate * gradient_yi;
            //    yj += params.learning_rate * gradient_yj;

            // Simulación de actualización para la prueba (eliminar al implementar el gradiente real):
            xu[0] += 0.0001;
            yi[0] += 0.0001;
            yj[0] -= 0.0001;
        }
        std::cout << "Epoch " << epoch + 1 << "/" << params.epochs << " completado." << std::endl;
    }
     std::cout << "Entrenamiento finalizado." << std::endl;
}
```

#### tests/main_test_srpr_trainer.cpp

```cpp
#include "../include/UserItemStore.h"
#include "../include/SRPR_Trainer.h"
#include <iostream>

int main() {
    std::cout << "--- Probando SRPR_Trainer ---" << std::endl;
    int dimensions = 8;

    std::vector<Triplet> triplets = {{101, 1, 2}};
    UserItemStore store(dimensions);
    store.initialize(triplets);

    Vector initial_vec = store.get_user_vector(101);
    std::cout << "Vector inicial usuario 101[0]: " << initial_vec[0] << std::endl;

    SRPR_Trainer trainer(store);
    SRPR_Trainer::TrainingParams params;
    params.epochs = 5;

    trainer.train(triplets, params);

    Vector final_vec = store.get_user_vector(101);
    std::cout << "Vector final usuario 101[0]: " << final_vec[0] << std::endl;

    if (initial_vec[0] >= final_vec[0]) {
        std::cerr << "Prueba fallida: El vector no cambió como se esperaba después del entrenamiento." << std::endl;
        return 1;
    }

    std::cout << "Prueba de SRPR_Trainer completada con éxito." << std::endl;
    return 0;
}
```

### Paso 5: Ensamblaje Final

Una vez que todos los componentes anteriores han sido implementados y probados (especialmente la lógica del gradiente en SRPR_Trainer), usa el siguiente main.cpp para simular el flujo completo.

#### main.cpp

```cpp
#include "include/Triplet.h"
#include "include/UserItemStore.h"
#include "include/SRPR_Trainer.h"
#include "include/LSH.h"
#include <iostream>
#include <vector>
#include <string>
#include <algorithm> // para std::sort

// Función para la estrategia Hamming Ranking
void hamming_ranking_retrieval(int user_id_query, int k, const UserItemStore& store, const LSH& hasher) {
    std::cout << "\n--- Iniciando Recuperación Top-" << k << " para Usuario " << user_id_query << " (Hamming Ranking) ---" << std::endl;

    // 1. Obtener vector y código del usuario
    const Vector& user_vector = store.get_user_vector(user_id_query);
    std::string user_code = hasher.generate_code(user_vector);
    std::cout << "Código LSH del usuario: " << user_code << std::endl;

    // 2. Generar códigos para todos los ítems y calcular distancia
    std::vector<std::pair<int, int>> item_distances; // <item_id, hamming_distance>
    const auto& all_items = store.get_all_item_vectors();

    for (const auto& pair : all_items) {
        int item_id = pair.first;
        const Vector& item_vector = pair.second;
        std::string item_code = hasher.generate_code(item_vector);

        // Calcular distancia de Hamming
        int distance = 0;
        for (size_t i = 0; i < user_code.length(); ++i) {
            if (user_code[i] != item_code[i]) {
                distance++;
            }
        }
        item_distances.push_back({item_id, distance});
    }

    // 3. Ordenar por distancia y obtener top-k
    std::sort(item_distances.begin(), item_distances.end(),
        [](const auto& a, const auto& b) {
            return a.second < b.second;
        });

    std::cout << "Top " << k << " ítems recomendados:" << std::endl;
    for (int i = 0; i < k && i < item_distances.size(); ++i) {
        std::cout << "  - Ítem: " << item_distances[i].first
                  << " (Distancia: " << item_distances[i].second << ")" << std::endl;
    }
}

int main(int argc, char* argv[]) {
    std::cout << "===== Ejecución del Pipeline SRPR Completo =====" << std::endl;

    // --- Configuración ---
    int dimensions = 20;
    int lsh_code_length = 32;
    int top_k = 3;

    // --- Fase I: Aprendizaje ---
    std::cout << "\n--- Fase I: Aprendizaje ---" << std::endl;
    std::vector<Triplet> triplets = {
        {1, 101, 102}, {1, 101, 103}, {1, 104, 102},
        {2, 102, 101}, {2, 102, 104},
        {3, 105, 101}, {3, 106, 102}
    };
    std::cout << "Datos de entrenamiento: " << triplets.size() << " tripletas." << std::endl;

    UserItemStore store(dimensions);
    store.initialize(triplets);
    std::cout << "UserItemStore inicializado." << std::endl;

    SRPR_Trainer trainer(store);
    SRPR_Trainer::TrainingParams params;
    params.epochs = 20;
    trainer.train(triplets, params);

    std::cout << "Entrenamiento completado. Vectores latentes aprendidos." << std::endl;

    // --- Fase II: Recuperación ---
    std::cout << "\n--- Fase II: Recuperación ---" << std::endl;
    SRPHasher hasher(dimensions, lsh_code_length);
    std::cout << "Hasher SRP creado con longitud de código " << lsh_code_length << "." << std::endl;

    // Realizar una consulta de ejemplo para el usuario 1
    int user_id_query = 1;
    hamming_ranking_retrieval(user_id_query, top_k, store, hasher);

    std::cout << "\n===== Pipeline SRPR completado. =====" << std::endl;

    return 0;
}
```

## 4. Notas Importantes

### Objetivo Principal
La **tarea principal** para el agente de IA es implementar la lógica del gradiente en el método `SRPR_Trainer::train()`. Esta es la parte más compleja del algoritmo y requiere:

1. Calcular las probabilidades de colisión LSH `p_ui` y `p_uj`
2. Calcular el valor gamma según la Ecuación 5 del paper
3. Implementar el gradiente de la función de verosimilitud
4. Actualizar los vectores usando el gradiente calculado

### Compilación y Pruebas
Para cada paso, compile y ejecute las pruebas individuales antes de continuar:

```bash
g++ -std=c++11 tests/main_test_triplet.cpp -o test_triplet
./test_triplet
```

### Consideraciones de Implementación
- Use vectores de doble precisión para mayor estabilidad numérica
- Implemente verificaciones de límites y manejo de errores
- La inicialización aleatoria debe ser reproducible para pruebas
- Considere optimizaciones de rendimiento solo después de que la funcionalidad esté completa