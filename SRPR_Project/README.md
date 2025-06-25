# SRPR Recommender System

**Stochastically Robust Personalized Ranking for LSH Recommendation Retrieval**

Una implementación completa en C++ del algoritmo SRPR para sistemas de recomendación basados en LSH (Locality Sensitive Hashing).

## 📋 Descripción

Este proyecto implementa el algoritmo SRPR descrito en el paper "Stochastically Robust Personalized Ranking for LSH Recommendation Retrieval". El sistema utiliza Sign Random Projection (SRP-LSH) y optimización de gradientes para aprender representaciones vectoriales robustas que mejoran la efectividad de los sistemas de recomendación basados en hashing.

### Características Principales

- ✅ **Algoritmo SRPR completo** con gradientes implementados
- ✅ **LSH (SRP-LSH)** para códigos binarios eficientes
- ✅ **Entrenamiento escalable** con datos reales (MovieLens)
- ✅ **Hamming Ranking** para recomendaciones rápidas
- ✅ **Sistema end-to-end** con interfaz de línea de comandos
- ✅ **Alto rendimiento**: 58K actualizaciones/segundo

## 🏗️ Arquitectura del Sistema

### Componentes Principales

1. **Triplet.h/cpp** - Manejo de datos y conversión de ratings a tripletas
2. **UserItemStore.h/cpp** - Gestión de vectores latentes de usuarios e ítems
3. **LSH.h/cpp** - Sistema LSH con Sign Random Projection
4. **SRPR_Trainer.h/cpp** - Algoritmo de entrenamiento con gradientes
5. **main.cpp** - Sistema integrado con CLI

### Flujo de Trabajo

```
Datos MovieLens → Tripletas → UserItemStore → SRPR Training → LSH Codes → Recomendaciones
```

## 🚀 Instalación y Compilación

### Requisitos

- Compilador C++11 o superior
- Sistema operativo: Linux, macOS, Windows (con WSL)

### Compilación

```bash
# Navegar al directorio del proyecto
cd ProyectoFinal/SRPR_Project

# Compilar el sistema completo
g++ -std=c++11 -O2 src/*.cpp main.cpp -o srpr_system

# O compilar componentes individuales para testing
g++ -std=c++11 src/UserItemStore.cpp tests/main_test_useritemstore.cpp -o test_useritemstore
g++ -std=c++11 src/LSH.cpp src/UserItemStore.cpp tests/main_test_lsh.cpp -o test_lsh
g++ -std=c++11 src/SRPR_Trainer.cpp src/UserItemStore.cpp tests/main_test_srpr_trainer.cpp -o test_srpr_trainer
```

## 📊 Preparación de Datos

### Generar Dataset de Entrenamiento

```bash
# Compilar el generador de datos
g++ -std=c++11 tests/generate_training_data.cpp src/UserItemStore.cpp -o generate_training_data

# Generar dataset desde MovieLens (requiere datos en data/movielens/ml-20m/)
./generate_training_data 500000 50 1.0

# Esto genera:
# - data/training_triplets.csv (dataset principal)
# - data/validation_triplets.csv (dataset de validación)
```

### Formato de Datos

Las tripletas tienen el formato: `user_id,preferred_item_id,less_preferred_item_id`

```csv
user_id,preferred_item_id,less_preferred_item_id
1,1266,7482
1,5952,2194
1,6774,1525
```

## 🎯 Uso del Sistema

### 1. Entrenamiento

```bash
# Entrenamiento básico
./srpr_system --train

# Entrenamiento con parámetros personalizados
./srpr_system --train --epochs 30 --lr 0.01 --dimensions 64 --lsh-bits 32 --verbose

# Entrenamiento con archivos específicos
./srpr_system --train --data-file mi_dataset.csv --val-file mi_validacion.csv
```

### 2. Generar Recomendaciones

```bash
# Recomendaciones para usuario específico
./srpr_system --recommend 1 --top-k 20

# Con configuración personalizada
./srpr_system --recommend 42 --top-k 10 --dimensions 32 --lsh-bits 16 --verbose
```

### 3. Evaluación del Modelo

```bash
# Evaluación completa
./srpr_system --evaluate --verbose

# Evaluación con archivos específicos
./srpr_system --evaluate --data-file dataset.csv --val-file validacion.csv
```

### Opciones de Línea de Comandos

| Opción | Descripción | Valor por Defecto |
|--------|-------------|-------------------|
| `--help, -h` | Mostrar ayuda | - |
| `--train` | Entrenar modelo SRPR | - |
| `--recommend USER_ID` | Generar recomendaciones | - |
| `--evaluate` | Evaluar modelo | - |
| `--data-file FILE` | Archivo de datos | `data/training_triplets.csv` |
| `--val-file FILE` | Archivo de validación | `data/validation_triplets.csv` |
| `--epochs N` | Número de epochs | 20 |
| `--lr RATE` | Learning rate | 0.005 |
| `--dimensions N` | Dimensiones de vectores | 32 |
| `--lsh-bits N` | Bits de LSH | 16 |
| `--top-k N` | Top-K recomendaciones | 10 |
| `--verbose` | Modo verboso | false |

## 📈 Configuración y Rendimiento

### Configuración Recomendada

- **Dimensiones**: 32-64 (balance entre precisión y velocidad)
- **LSH bits**: 16-32 (16 para velocidad, 32 para precisión)
- **Learning rate**: 0.001-0.01 (0.005 por defecto)
- **Epochs**: 15-30 (dependiendo del dataset)

### Métricas de Rendimiento

- ⚡ **Entrenamiento**: ~58,000 actualizaciones/segundo
- ⚡ **Generación LSH**: ~321,000 códigos/segundo
- ⚡ **Acceso vectores**: ~15.4M accesos/segundo
- 💾 **Memoria**: ~0.8 MB para 370 usuarios + 2,769 ítems (32D)

## 🧪 Testing

### Ejecutar Todas las Pruebas

```bash
# Pruebas de componentes individuales
./test_useritemstore
./test_lsh
./test_srpr_trainer

# Pruebas de integración
g++ -std=c++11 src/LSH.cpp src/UserItemStore.cpp tests/test_lsh_integration.cpp -o test_integration
./test_integration

# Prueba completa con datos reales
g++ -std=c++11 src/SRPR_Trainer.cpp src/LSH.cpp src/UserItemStore.cpp tests/test_srpr_trainer_real_data.cpp -o test_real
./test_real
```

### Estructura de Testing

```
tests/
├── main_test_triplet.cpp              # Pruebas de carga de datos
├── main_test_useritemstore.cpp        # Pruebas de vectores latentes
├── main_test_lsh.cpp                  # Pruebas de LSH
├── main_test_srpr_trainer.cpp         # Pruebas de entrenamiento
├── test_lsh_integration.cpp           # Pruebas de integración
├── test_srpr_trainer_real_data.cpp    # Pruebas con datos reales
└── generate_training_data.cpp         # Generador de datasets
```

## 📁 Estructura del Proyecto

```
SRPR_Project/
├── include/                    # Headers
│   ├── Triplet.h              # Estructuras de datos y carga
│   ├── UserItemStore.h        # Gestión de vectores latentes
│   ├── LSH.h                  # LSH y SRP-LSH
│   └── SRPR_Trainer.h         # Algoritmo de entrenamiento
├── src/                       # Implementaciones
│   ├── UserItemStore.cpp      # Gestión de vectores
│   ├── LSH.cpp                # Sistema LSH
│   └── SRPR_Trainer.cpp       # Algoritmo SRPR completo
├── tests/                     # Pruebas unitarias e integración
├── data/                      # Datasets
│   ├── training_triplets.csv  # Dataset principal
│   ├── validation_triplets.csv # Dataset de validación
│   └── movielens/             # Datos originales MovieLens
├── main.cpp                   # Sistema integrado
├── README.md                  # Esta documentación
└── requirements.md            # Especificaciones técnicas
```

## 🔬 Algoritmo SRPR

### Funcionamiento

1. **Entrada**: Tripletas de preferencia (user, item_preferred, item_less_preferred)
2. **Vectores**: Representaciones latentes de usuarios e ítems (32D)
3. **LSH**: Códigos binarios usando Sign Random Projection
4. **Optimización**: Gradiente ascendente para maximizar log-verosimilitud
5. **Recomendación**: Hamming Ranking basado en códigos LSH

### Ecuaciones Clave

- **Probabilidad LSH**: `p_ui^srp = 1 - arccos(cosine_similarity) / π`
- **Gamma**: `γ = (p_uj - p_ui) / sqrt(var_ui + var_uj)`
- **Objetivo**: Maximizar `log Φ(sqrt(b) * γ)`

## 📊 Ejemplo de Resultados

### Entrenamiento Típico

```
Epoch  1/15 | Loss: -0.584898 | Time:  126ms
Epoch  2/15 | Loss: -0.453816 | Time:  126ms
Epoch  3/15 | Loss: -0.374542 | Time:  126ms | Val Loss: -0.373470
...
Epoch 15/15 | Loss: -0.149695 | Time:  127ms | Val Loss: -0.154647

Mejora en pérdida: 0.435 puntos
Velocidad: 58,206 actualizaciones/s
```

### Recomendaciones Generadas

```
RECOMENDACIONES PARA USUARIO 1
Rank | Item ID | Distancia Hamming | Similitud %
-----|---------|-------------------|-------------
   1 |     907 |                 1 |       93.8%
   2 |    2138 |                 1 |       93.8%
   3 |    1171 |                 1 |       93.8%
   4 |   30825 |                 2 |       87.5%
   5 |    3178 |                 2 |       87.5%
```

## 🐛 Troubleshooting

### Problemas Comunes

**Error: "No se pudo cargar el dataset"**
- Verifica que existan los archivos `data/training_triplets.csv`
- Ejecuta `./generate_training_data` para crear los datasets

**Error: "Usuario no encontrado"**
- Verifica que el usuario exista en el dataset
- Usa `--evaluate` para ver usuarios disponibles

**Rendimiento lento**
- Reduce el número de epochs con `--epochs`
- Usa menos dimensiones con `--dimensions`
- Reduce el dataset de entrenamiento

### Optimización

- **Para velocidad**: Usa `--dimensions 16 --lsh-bits 8`
- **Para precisión**: Usa `--dimensions 64 --lsh-bits 32`
- **Para memoria**: Reduce el tamaño del dataset

## 📚 Referencias

- Paper original: "Stochastically Robust Personalized Ranking for LSH Recommendation Retrieval"
- Dataset: MovieLens ML-20M
- Técnica LSH: Sign Random Projection (SRP)

## 👥 Contribuciones

Este proyecto fue desarrollado como implementación del algoritmo SRPR en C++. 

### Características Implementadas

- ✅ Algoritmo SRPR completo con gradientes
- ✅ Sistema LSH con SRP
- ✅ Integración con datos reales MovieLens
- ✅ Sistema de recomendación end-to-end
- ✅ Interfaz de línea de comandos
- ✅ Suite completa de pruebas

---

¡El sistema SRPR está listo para usar! 🚀