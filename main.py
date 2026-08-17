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

    # Primeira camada oculta
    network.add(
        Dense(
            input_size=X_train.shape[1],
            output_size=arquitetura[0],
            activation=activation
        )
    )

    # Demais camadas ocultas
    for i in range(1, len(arquitetura)):

        network.add(
            Dense(
                input_size=arquitetura[i - 1],
                output_size=arquitetura[i],
                activation=activation
            )
        )

    # Camada de saída
    network.add(
        Dense(
            input_size=arquitetura[-1],
            output_size=1,
            activation=Sigmoid()
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

    print("\n")
    print("=" * 60)
    print("MELHOR CONFIGURAÇÃO")
    print("=" * 60)

    print(
        f"Accuracy de treino: "
        f"{search.best_score:.4f}"
    )

    print(
        f"Loss de treino: "
        f"{search.best_loss:.6f}"
    )

    print(
        f"Parâmetros: "
        f"{search.best_params}"
    )

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

    print("\nPrimeiras 20 predições:")
    print(predictions[:20])

    print("\nPrimeiras 20 classes previstas:")
    print(classes[:20])

    print("\nPrimeiras 20 classes reais:")
    print(y_test[:20])


if __name__ == "__main__":
    main()