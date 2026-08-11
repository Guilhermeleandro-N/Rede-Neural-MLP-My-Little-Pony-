# Multilayer Perceptron (MLP) - Implementação do Zero em Python

Uma implementação educacional de uma **Rede Neural Artificial do tipo Multilayer Perceptron (MLP)** desenvolvida do zero utilizando apenas **NumPy**, sem o uso de bibliotecas de Deep Learning.

O projeto foi desenvolvido para apoiar a disciplina de **Inteligência Computacional**, permitindo compreender as etapas envolvidas no treinamento de uma rede neural.

---

# Funcionalidades

- Rede Neural Multilayer Perceptron (MLP)
- Camadas totalmente conectadas (Dense)
- Forward Propagation
- Backpropagation
- Inicialização de pesos
    - He Initialization
    - Xavier Initialization
- Funções de ativação
    - Sigmoid
    - Tanh
    - ReLU
    - Leaky ReLU
    - ELU
    - SELU
    - Softplus
    - Swish
    - Mish
    - GELU
    - Softmax
- Funções de perda
    - Mean Squared Error (MSE)
    - Cross Entropy
- Otimizadores
    - SGD
    - Momentum
    - RMSProp
    - Adam
    - AdamW
    - Nadam
- Grid Search para ajuste de hiperparâmetros
- Salvamento e carregamento de modelos
- Métricas de avaliação
- Estrutura modular para facilitar extensões

---

# 📁 Estrutura do Projeto

```text
.
├── datasets/
│   └── xor.py
│
├── mlp/
│   ├── activations.py
│   ├── checkpoint.py
│   ├── grid_search.py
│   ├── initializers.py
│   ├── layers.py
│   ├── losses.py
│   ├── metrics.py
│   ├── network.py
│   └── optimizers.py
│
├── main.py
├── requirements.txt
└── README.md
```

---

# ⚙️ Pré-requisitos

Antes de executar o projeto é necessário possuir:

- Python 3.11 ou superior
- pip
- Git (opcional)

Verifique sua instalação:

```bash
python --version
```

ou

```bash
python3 --version
```

---

# 🚀 Instalação

## 1. Clone o repositório

```bash
git clone https://github.com/SEU-USUARIO/SEU-REPOSITORIO.git
```

Entre na pasta do projeto.

```bash
cd SEU-REPOSITORIO
```

---

## 2. Crie um ambiente virtual

### Windows

Crie o ambiente virtual:

```powershell
python -m venv .venv
```

Ative o ambiente:

**CMD**

```cmd
.venv\Scripts\activate
```

**PowerShell**

```powershell
.venv\Scripts\Activate.ps1
```

Caso apareça erro relacionado à política de execução:

```powershell
Set-ExecutionPolicy -Scope Process RemoteSigned
```

Depois execute novamente:

```powershell
.venv\Scripts\Activate.ps1
```

---

### Linux

Crie o ambiente virtual:

```bash
python3 -m venv .venv
```

Ative:

```bash
source .venv/bin/activate
```

---

## 3. Atualize o pip

```bash
pip install --upgrade pip
```

---

## 4. Instale as dependências

Se existir um arquivo `requirements.txt`:

```bash
pip install -r requirements.txt
```

Caso contrário:

```bash
pip install numpy scipy pandas matplotlib scikit-learn
```

---

# ▶️ Executando o Projeto

Na pasta principal execute:

### Windows

```bash
python main.py


### Linux

```bash
python3 main.py
```

---

# 📈 Exemplo de Saída

```text
============================================================
GRID SEARCH - XOR
============================================================

Experimento 1

Epoch 0 Loss=0.27654

Epoch 100 Loss=0.14531

Epoch 200 Loss=0.03184

...

Accuracy = 1.0000

Loss = 0.000003
```

Ao final será exibida a melhor configuração encontrada.

---

# 🏗 Arquitetura da Rede

O exemplo padrão utiliza a base XOR.

```text
Entrada (2 atributos)
        │
        ▼
Dense
        │
        ▼
Função de Ativação
        │
        ▼
Dense
        │
        ▼
Sigmoid
        │
        ▼
Saída
```

A arquitetura pode ser modificada livremente no arquivo `main.py`.

---

# 🔍 Grid Search

O projeto possui implementação própria de **Grid Search**, permitindo testar automaticamente diferentes combinações de hiperparâmetros.

Exemplo de parâmetros avaliados:

- Número de neurônios
- Função de ativação
- Otimizador
- Número de épocas
- Função de perda

Ao final é selecionado automaticamente o melhor modelo.

---

# 📚 Conteúdo Implementado

## Funções de Ativação

- Sigmoid
- Tanh
- ReLU
- Leaky ReLU
- ELU
- SELU
- Softplus
- Swish
- Mish
- GELU
- Softmax

---

## Inicialização dos Pesos

- He Initialization
- Xavier Initialization

---

## Funções de Perda

- Mean Squared Error (MSE)
- Cross Entropy

---

## Otimizadores

- SGD
- Momentum
- RMSProp
- Adam
- AdamW
- Nadam

---

# 🛠 Desenvolvendo Novos Experimentos

Como este projeto possui caráter educacional, ele foi estruturado para facilitar a implementação de novos recursos, como:

- Dropout
- Early Stopping
- Batch Normalization
- Regularização L1
- Regularização L2
- Mini-Batch Gradient Descent
- Novas funções de ativação
- Novos otimizadores
- Novas funções de perda
- Novas métricas
- Novas arquiteturas

---

# Autor

**Prof. Dr. Pedro Luis Saraiva Barbosa**

Professor do Instituto Federal de Educação, Ciência e Tecnologia do Ceará (IFCE)

Disciplina: **Inteligência Computacional**

---

# Licença

Este projeto foi desenvolvido exclusivamente para fins acadêmicos e educacionais.