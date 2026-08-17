#para garantir a reprodutibilidade dos experimentos, foi utilizada uma semente aleatória fixa
import numpy as np
np.random.seed(42)

from datasets.adult import (
    X_train,
    X_test,
    y_train,
    y_test
)

from mlp.network import NeuralNetwork
from mlp.layers import Dense
from mlp.initializers import (
    HeInitializer,
    XavierInitializer
)

from mlp.activations import (
    ReLU,
    Tanh,
    LeakyReLU,
    GELU,
    ELU,
    SELU,
    Swish,
    Softplus,
    Mish,
    Sigmoid
)

from mlp.losses import MSE

from mlp.optimizers import (
    Adam
)

from mlp.grid_search import GridSearch

from mlp.metrics import (
    Accuracy,
    Precision,
    Recall,
    F1Score
)


def build_network(params):
    """
    Constrói uma rede com os parâmetros
    recebidos pelo Grid Search.
    """

    network = NeuralNetwork()

    activation = params["activation"]()

    arquitetura = params["arquitetura"]

    initializer = params.get(
        "initializer",
        HeInitializer
    )

    # Primeira camada oculta
    network.add(
        Dense(
            input_size=X_train.shape[1],
            output_size=arquitetura[0],
            activation=activation,
            initializer=initializer
        )
    )

    # Demais camadas ocultas
    for i in range(1, len(arquitetura)):

        network.add(
            Dense(
                input_size=arquitetura[i - 1],
                output_size=arquitetura[i],
                activation=activation,
                initializer=initializer
            )
        )

    # Camada de saída
    network.add(
        Dense(
            input_size=arquitetura[-1],
            output_size=1,
            activation=Sigmoid(),
            initializer=initializer
        )
    )

    return network


def main():

    print("=" * 60)
    print("GRID SEARCH - ADULT INCOME")
    print("=" * 60)

    print("\nDados:")
    print(f"X_train: {X_train.shape}")
    print(f"y_train: {y_train.shape}")
    print(f"X_test:  {X_test.shape}")
    print(f"y_test:  {y_test.shape}")

# # EXPERIMENTO 1 - NÚMERO DE NEURÔNIOS
#     param_grid = {

#         "arquitetura": [
#             [4],
#             [8],
#             [16],
#             [32],
#             [64]
#         ],

#         "activation": [
#             ReLU
#         ],

#         "optimizer": [
#             lambda: Adam(
#                 lr=0.001
#             )
#         ],

#         "epochs": [
#             100
#         ],

#         "loss": [
#             MSE
#         ]
#     }



# # EXPERIMENTO 2 - NÚMERO DE CAMADAS OCULTAS
#     param_grid = {
#         "arquitetura": [
#             [32],
#             [32, 16],
#             [32, 16, 8],
#             [32, 16, 8, 4]
#         ],
#         "activation": [
#             ReLU
#         ],

#         "optimizer": [
#             lambda: Adam(
#                 lr=0.001
#             )
#         ],

#         "epochs": [
#             100
#         ],

#         "loss": [
#             MSE
#         ]
#     }



    param_grid = {

        "arquitetura": [
            [32, 16, 8]
        ],

        "activation": [
            ReLU,
            Tanh,
            Mish

        ],

        "initializer": [
        HeInitializer,
        XavierInitializer
        ],

        "optimizer": [
            lambda: Adam(
                lr=0.001
            )
        ],

        "epochs": [
            100
        ],

        "loss": [
            MSE
        ]
    }

        # ==========================================
    # EXPERIMENTO — INICIALIZADORES
    # ==========================================

    print("\n")
    print("=" * 60)
    print("EXPERIMENTO - COMPARAÇÃO DE INICIALIZADORES")
    print("=" * 60)

    resultados_inicializadores = []

    initializers = [
        ("He", HeInitializer),
        ("Xavier", XavierInitializer)
    ]

    for initializer_name, initializer in initializers:

        print("\n")
        print("-" * 60)
        print(f"Inicializador: {initializer_name}")
        print("-" * 60)

        # Reinicia a seed para tornar a comparação reprodutível
        np.random.seed(42)

        experiment_params = {
            "arquitetura": [32, 16, 8],
            "activation": ReLU,
            "initializer": initializer,
            "optimizer": lambda: Adam(lr=0.001),
            "epochs": 100,
            "loss": MSE
        }

        network = build_network(
            experiment_params
        )

        optimizer = (
            experiment_params["optimizer"]()
        )

        loss_function = (
            experiment_params["loss"]()
        )

        network.fit(
            X_train,
            y_train,
            epochs=experiment_params["epochs"],
            loss_function=loss_function,
            optimizer=optimizer
        )

        predictions = network.forward(
            X_train
        )

        accuracy = Accuracy.calculate(
            y_train,
            predictions
        )

        precision = Precision.calculate(
            y_train,
            predictions
        )

        recall = Recall.calculate(
            y_train,
            predictions
        )

        f1 = F1Score.calculate(
            y_train,
            predictions
        )

        loss = loss_function.forward(
            y_train,
            predictions
        )

        resultados_inicializadores.append({
            "Experimento": initializer_name,
            "Arquitetura": "32-16-8",
            "Ativação": "ReLU",
            "Inicializador": initializer_name,
            "Otimizador": "Adam",
            "LR": 0.001,
            "Épocas": 100,
            "Acurácia": accuracy,
            "F1": f1
        })

        print(f"\nResultado - {initializer_name}")
        print(f"Accuracy  = {accuracy:.4f}")
        print(f"Precision = {precision:.4f}")
        print(f"Recall    = {recall:.4f}")
        print(f"F1-score  = {f1:.4f}")
        print(f"Loss      = {loss:.6f}")

    # ==========================================
    # TABELA RESUMO — INICIALIZADORES
    # ==========================================

    print("\n")
    print("=" * 100)
    print("TABELA RESUMO - EXPERIMENTO DE INICIALIZADORES")
    print("=" * 100)

    print(
        f"{'Experimento':<12}"
        f"{'Arquitetura':<15}"
        f"{'Ativação':<12}"
        f"{'Inicializador':<15}"
        f"{'Otimizador':<12}"
        f"{'LR':<10}"
        f"{'Épocas':<10}"
        f"{'Acurácia':<12}"
        f"{'F1':<10}"
    )

    print("-" * 100)

    for resultado in resultados_inicializadores:
        print(
            f"{resultado['Experimento']:<12}"
            f"{resultado['Arquitetura']:<15}"
            f"{resultado['Ativação']:<12}"
            f"{resultado['Inicializador']:<15}"
            f"{resultado['Otimizador']:<12}"
            f"{resultado['LR']:<10}"
            f"{resultado['Épocas']:<10}"
            f"{resultado['Acurácia']:<12.4f}"
            f"{resultado['F1']:<10.4f}"
        )

    search = GridSearch(
        network_builder=build_network,
        param_grid=param_grid
    )

    print("\n")
    print("=" * 60)
    print("INICIANDO GRID SEARCH")
    print("=" * 60)

    search.fit(
        X_train,
        y_train
    )

    search.summary()

    # ==========================================
    # Avaliação no conjunto de teste
    # ==========================================

    print("\n")
    print("=" * 60)
    print("AVALIAÇÃO NO CONJUNTO DE TESTE")
    print("=" * 60)

    best_network = search.best_model

    predictions = best_network.forward(
        X_test
    )

    test_accuracy = Accuracy.calculate(
        y_test,
        predictions
    )
    test_precision = Precision.calculate(
        y_test,
        predictions
    )

    test_recall = Recall.calculate(
        y_test,
        predictions
    )

    test_f1 = F1Score.calculate(
        y_test,
        predictions
    )

    test_loss = MSE().forward(
        y_test,
        predictions
    )

    print("\nMétricas finais no conjunto de teste:")
    print(f"Accuracy  = {test_accuracy:.4f}")
    print(f"Precision = {test_precision:.4f}")
    print(f"Recall    = {test_recall:.4f}")
    print(f"F1-score  = {test_f1:.4f}")
    print(f"Loss      = {test_loss:.6f}")



    classes = (
        predictions > 0.5
    ).astype(int)

        # ==========================================
    # MATRIZ DE CONFUSÃO
    # ==========================================

    tn = np.sum(
        (y_test == 0) & (classes == 0)
    )

    fp = np.sum(
        (y_test == 0) & (classes == 1)
    )

    fn = np.sum(
        (y_test == 1) & (classes == 0)
    )

    tp = np.sum(
        (y_test == 1) & (classes == 1)
    )

    print("\n")
    print("=" * 60)
    print("MATRIZ DE CONFUSÃO")
    print("=" * 60)

    print("\n                 Predito")
    print("              0          1")
    print(f"Real  0     {tn:6d}     {fp:6d}")
    print(f"      1     {fn:6d}     {tp:6d}")

    print("\nComponentes:")
    print(f"Verdadeiros Negativos (TN): {tn}")
    print(f"Falsos Positivos      (FP): {fp}")
    print(f"Falsos Negativos      (FN): {fn}")
    print(f"Verdadeiros Positivos (TP): {tp}")

    print("\nPrimeiras 20 predições:")
    print(predictions[:20])

    print("\nPrimeiras 20 classes previstas:")
    print(classes[:20])

    print("\nPrimeiras 20 classes reais:")
    print(y_test[:20])


if __name__ == "__main__":
    main()