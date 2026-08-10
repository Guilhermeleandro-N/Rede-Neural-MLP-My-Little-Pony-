from datasets.xor import X, y

from mlp.network import NeuralNetwork
from mlp.layers import Dense

from mlp.activations import (
    ReLU,
    Tanh,
    Sigmoid
)

from mlp.losses import MSE

from mlp.optimizers import (
    SGD,
    Momentum,
    RMSProp,
    Adam,
    AdamW,
    Nadam
)

from mlp.grid_search import GridSearch


def build_network(params):
    """
    Constrói uma rede com os parâmetros
    recebidos pelo Grid Search.
    """

    network = NeuralNetwork()

    activation = params["activation"]()

    network.add(
        Dense(
            input_size=2,
            output_size=params["neurons"],
            activation=activation
        )
    )

    network.add(
        Dense(
            input_size=params["neurons"],
            output_size=1,
            activation=Sigmoid()
        )
    )

    return network


def main():

    print("=" * 60)
    print("GRID SEARCH - XOR")
    print("=" * 60)

    param_grid = {

        "neurons": [
            4,
            8,
            16
        ],

        # classes, não instâncias
        "activation": [
            ReLU,
            Tanh
        ],

        # fábricas de otimizadores
        "optimizer": [

            lambda: SGD(
                lr=0.01
            ),

            lambda: Momentum(
                lr=0.01,
                momentum=0.9
            ),

            lambda: RMSProp(
                lr=0.001
            ),

            lambda: Adam(
                lr=0.001
            ),

            lambda: AdamW(
                lr=0.001,
                weight_decay=0.01
            ),

            lambda: Nadam(
                lr=0.001
            )
        ],

        "epochs": [
            1000,
            5000
        ],

        "loss": [
            MSE
        ]
    }

    search = GridSearch(
        network_builder=build_network,
        param_grid=param_grid
    )

    search.fit(X, y)

    print("\n")
    print("=" * 60)
    print("MELHOR CONFIGURAÇÃO")
    print("=" * 60)

    print(f"Acurácia: {search.best_score:.4f}")
    print(f"Parâmetros: {search.best_params}")

    print("\n")
    print("=" * 60)
    print("TODOS OS RESULTADOS")
    print("=" * 60)

    search.summary()

    print("\n")
    print("=" * 60)
    print("TREINANDO MELHOR MODELO")
    print("=" * 60)

    print("\n")
    print("=" * 60)
    print("MELHOR MODELO")
    print("=" * 60)

    best_network = search.best_model

    predictions = (
        best_network.forward(X)
    )

    print("\nPredições:")

    print(predictions)

    print("\nClasses:")

    print(
        (predictions > 0.5)
        .astype(int)
    )

    best_network.fit(
        X,
        y,
        epochs=search.best_params["epochs"],
        loss_function=search.best_params["loss"](),
        optimizer=search.best_params["optimizer"]()
    )

    predictions = best_network.forward(X)

    print("\nPredições:")

    print(predictions)

    print("\nClasses:")

    print(
        (predictions > 0.5)
        .astype(int)
    )


if __name__ == "__main__":
    main()