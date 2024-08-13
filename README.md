# FindMyPic: Sistema de Recuperación de Información Basado en Imágenes

## Introducción

El objetivo de este proyecto es diseñar y desarrollar FindMyPic, un sistema de recuperación de imágenes que permite a los usuarios realizar consultas utilizando imágenes en lugar de texto. Este sistema está diseñado para encontrar imágenes similares dentro de una base de datos dada. El proyecto se divide en varias fases, que se describen a continuación.

## Fases del Proyecto

### 1. Adquisición de Datos
- **Objetivo:** Obtener y preparar el dataset Caltech101.
- **Tareas:** Descargar, descomprimir y organizar el dataset.

### 2. Preprocesamiento
- **Objetivo:** Preparar las imágenes para su análisis.
- **Tareas:** Normalizar, reducir tamaño y eliminar ruido de las imágenes; documentar el proceso.

### 3. Extracción de Características
- **Objetivo:** Extraer características procesables de las imágenes.
- **Tareas:** Utilizar una CNN para extraer características, entrenar o aplicar transfer learning, documentar métodos y resultados.

### 4. Indexación
- **Objetivo:** Crear un índice para búsquedas eficientes.
- **Tareas:** Desarrollar un sistema de indexación usando técnicas como k-NN, KD-Trees o LSH; documentar el proceso.

### 5. Diseño del Motor de Búsqueda
- **Objetivo:** Implementar la funcionalidad de búsqueda.
- **Tareas:** Desarrollar la lógica de consulta, el algoritmo de ranking y documentar la arquitectura y los algoritmos.

### 6. Evaluación del Sistema
- **Objetivo:** Medir la efectividad del sistema.
- **Tareas:** Definir métricas de evaluación, establecer benchmarks, comparar configuraciones y documentar resultados.

### 7. Interfaz Web de Usuario
- **Objetivo:** Crear una interfaz para la interacción con el sistema.
- **Tareas:** Diseñar una interfaz web para subir imágenes y mostrar resultados, asegurar una experiencia intuitiva y documentar el diseño.

## Entorno Virtual

Para garantizar la consistencia del entorno de desarrollo, se ha configurado un entorno virtual. Asegúrate de activarlo antes de trabajar en el proyecto.

```bash
# Activar el entorno virtual
source .venv/bin/activate
```
## Instalación de Dependencias

Las dependencias necesarias para el proyecto están listadas en el archivo `requirements.txt`. Puedes instalarlas ejecutando el siguiente comando:

```bash
pip install -r requirements.txt
```