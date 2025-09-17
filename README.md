# kedroEv1




#  EV1 Machine Learning Project
**Framework:** Kedro | **Lenguaje:** Python | **Entorno:** Jupyter

---

## 📋 Descripción del Proyecto
Este proyecto implementa un flujo completo de Machine Learning utilizando el framework **Kedro**, siguiendo las primeras tres fases de la metodología **CRISP-DM**.  
El objetivo es analizar y preparar datos transaccionales (clientes, productos y compras) para construir un dataset limpio, enriquecido y listo para modelado, aplicando **pipelines reproducibles** y **buenas prácticas de ingeniería de datos**.

---

## 🎯 Objetivos
- Comprensión del negocio y definición de hipótesis.
- Análisis exploratorio de datos (EDA).
- Integración de múltiples fuentes (mínimo 3 datasets).
- Limpieza y transformación de datos.
- Feature engineering: creación de variables derivadas, ratios e interacciones.
- Preparación de conjuntos `X_train`, `X_test`, `y_train`, `y_test` listos para modelado.

---

## 📊 Datasets
- `customer.csv`: Información demográfica y de registro de clientes.
- `product.csv`: Catálogo de productos con categoría y precio.
- `purchase.csv`: Historial de compras con fechas y montos.
  
  Link de descarga del data set (Es estrictamente obligatorio que se usen los nombres de arriba. Han sido modificados.)
- https://www.kaggle.com/datasets/svbstan/sales-product-and-customer-insight-repository

> Los archivos deben colocarse en la carpeta `data/01_raw/`.

---

## 🚀 Instalación y Configuración

### ✅ Prerrequisitos
- Python 3.8+
- Git
- Kedro

### ✅ Pasos
```bash
# 1. Clonar el repositorio
git clone https://github.com/prxska/ev1ML.git
cd ev1ML

# 2. Crear y activar entorno virtual
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Verificar instalación
kedro info

----------------------------------
# En caso de caso de querer tener una visualizacion interactiva

kedro viz


[![Powered by Kedro](https://img.shields.io/badge/powered_by-kedro-ffc900?logo=kedro)](https://kedro.org)

## Overview

This is your new Kedro project, which was generated using `kedro 1.0.0`.

Take a look at the [Kedro documentation](https://docs.kedro.org) to get started.

## Rules and guidelines

In order to get the best out of the template:

* Don't remove any lines from the `.gitignore` file we provide
* Make sure your results can be reproduced by following a data engineering convention
* Don't commit data to your repository
* Don't commit any credentials or your local configuration to your repository. Keep all your credentials and local configuration in `conf/local/`

## How to install dependencies

Declare any dependencies in `requirements.txt` for `pip` installation.

To install them, run:

```
pip install -r requirements.txt
```

## How to run your Kedro pipeline

You can run your Kedro project with:

```
kedro run
```

## How to test your Kedro project

Have a look at the file `tests/test_run.py` for instructions on how to write your tests. You can run your tests as follows:

```
pytest
```

You can configure the coverage threshold in your project's `pyproject.toml` file under the `[tool.coverage.report]` section.


## Project dependencies

To see and update the dependency requirements for your project use `requirements.txt`. You can install the project requirements with `pip install -r requirements.txt`.

[Further information about project dependencies](https://docs.kedro.org/en/stable/kedro_project_setup/dependencies.html#project-specific-dependencies)

## How to work with Kedro and notebooks

> Note: Using `kedro jupyter` or `kedro ipython` to run your notebook provides these variables in scope: `context`, 'session', `catalog`, and `pipelines`.
>
> Jupyter, JupyterLab, and IPython are already included in the project requirements by default, so once you have run `pip install -r requirements.txt` you will not need to take any extra steps before you use them.

### Jupyter
To use Jupyter notebooks in your Kedro project, you need to install Jupyter:

```
pip install jupyter
```

After installing Jupyter, you can start a local notebook server:

```
kedro jupyter notebook
```

### JupyterLab
To use JupyterLab, you need to install it:

```
pip install jupyterlab
```

You can also start JupyterLab:

```
kedro jupyter lab
```

### IPython
And if you want to run an IPython session:

```
kedro ipython
```

### How to ignore notebook output cells in `git`
To automatically strip out all output cell contents before committing to `git`, you can use tools like [`nbstripout`](https://github.com/kynan/nbstripout). For example, you can add a hook in `.git/config` with `nbstripout --install`. This will run `nbstripout` before anything is committed to `git`.

> *Note:* Your output cells will be retained locally.

## Package your Kedro project

[Further information about building project documentation and packaging your project](https://docs.kedro.org/en/stable/tutorial/package_a_project.html)
