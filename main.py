import numpy as np

# Seed global para reprodutibilidade
np.random.seed(42)


from datasets.adult import (
    X_train,
    X_val,
    X_test,
    y_train,
    y_val,
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
    Mish,
    Sigmoid
)


from mlp.losses import MSE


from mlp.optimizers import (
    Adam,
    RMSProp
)


from mlp.grid_search import GridSearch


from mlp.metrics import (
    Accuracy,
    Precision,
    Recall,
    F1Score
)


# ============================================================
# CONSTRUÇÃO DA REDE
# ============================================================

def build_network(params):

    network = NeuralNetwork()

    activation = (
        params["activation"]()
    )

    arquitetura = (
        params["arquitetura"]
    )

    initializer = params.get(
        "initializer",
        HeInitializer
    )


    # ==========================================
    # Primeira camada oculta
    # ==========================================

    network.add(
        Dense(
            input_size=X_train.shape[1],
            output_size=arquitetura[0],
            activation=activation,
            initializer=initializer
        )
    )


    # ==========================================
    # Demais camadas ocultas
    # ==========================================

    for i in range(
        1,
        len(arquitetura)
    ):

        network.add(
            Dense(
                input_size=arquitetura[i - 1],
                output_size=arquitetura[i],
                activation=activation,
                initializer=initializer
            )
        )


    # ==========================================
    # Camada de saída
    # ==========================================

    network.add(
        Dense(
            input_size=arquitetura[-1],
            output_size=1,
            activation=Sigmoid(),
            initializer=initializer
        )
    )


    return network


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("GRID SEARCH - ADULT INCOME")
    print("=" * 60)


    # ========================================================
    # DADOS
    # ========================================================

    print("\nDados:")

    print(
        f"X_train: {X_train.shape}"
    )

    print(
        f"y_train: {y_train.shape}"
    )

    print(
        f"X_val:   {X_val.shape}"
    )

    print(
        f"y_val:   {y_val.shape}"
    )

    print(
        f"X_test:  {X_test.shape}"
    )

    print(
        f"y_test:  {y_test.shape}"
    )


    # ========================================================
    # EXPERIMENTO - COMPARAÇÃO DE INICIALIZADORES
    # ========================================================

    print("\n")
    print("=" * 60)
    print("EXPERIMENTO - COMPARAÇÃO DE INICIALIZADORES")
    print("=" * 60)


    resultados_inicializadores = []


    initializers = [
        (
            "He",
            HeInitializer
        ),

        (
            "Xavier",
            XavierInitializer
        )
    ]


    for initializer_name, initializer in initializers:

        print("\n")
        print("-" * 60)

        print(
            f"Inicializador: {initializer_name}"
        )

        print("-" * 60)


        # Mesma seed para cada inicializador
        np.random.seed(42)


        experiment_params = {

            "arquitetura": [
                32,
                16,
                8
            ],

            "activation":
                ReLU,

            "initializer":
                initializer,

            "optimizer":
                Adam,

            "learning_rate":
                0.001,

            "epochs":
                100,

            "loss":
                MSE
        }


        network = build_network(
            experiment_params
        )


        optimizer = Adam(
            lr=experiment_params[
                "learning_rate"
            ]
        )


        loss_function = (
            experiment_params[
                "loss"
            ]()
        )


        # Treina apenas no conjunto de treino
        network.fit(
            X_train,
            y_train,
            epochs=experiment_params[
                "epochs"
            ],
            loss_function=loss_function,
            optimizer=optimizer
        )


        # Avalia no conjunto de validação
        predictions = network.forward(
            X_val
        )


        accuracy = Accuracy.calculate(
            y_val,
            predictions
        )


        precision = Precision.calculate(
            y_val,
            predictions
        )


        recall = Recall.calculate(
            y_val,
            predictions
        )


        f1 = F1Score.calculate(
            y_val,
            predictions
        )


        loss = loss_function.forward(
            y_val,
            predictions
        )


        resultados_inicializadores.append({

            "Experimento":
                initializer_name,

            "Arquitetura":
                "32-16-8",

            "Ativação":
                "ReLU",

            "Inicializador":
                initializer_name,

            "Otimizador":
                "Adam",

            "LR":
                0.001,

            "Épocas":
                100,

            "Acurácia":
                accuracy,

            "F1":
                f1
        })


        print(
            f"\nResultado - {initializer_name}"
        )

        print(
            f"Accuracy  = {accuracy:.4f}"
        )

        print(
            f"Precision = {precision:.4f}"
        )

        print(
            f"Recall    = {recall:.4f}"
        )

        print(
            f"F1-score  = {f1:.4f}"
        )

        print(
            f"Loss      = {loss:.6f}"
        )


    # ========================================================
    # TABELA RESUMO DOS INICIALIZADORES
    # ========================================================

    print("\n")

    print("=" * 100)

    print(
        "TABELA RESUMO - EXPERIMENTO DE INICIALIZADORES"
    )

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


    # ========================================================
    # GRID SEARCH DEFINITIVO
    # ========================================================
    #
    # 3 arquiteturas
    # x
    # 3 funções de ativação
    # x
    # 1 inicializador
    # x
    # 2 otimizadores
    # x
    # 2 learning rates
    # x
    # 2 quantidades de épocas
    #
    # TOTAL = 72 CONFIGURAÇÕES
    #
    # ========================================================

    param_grid = {

        # Varia arquitetura e profundidade
        "arquitetura": [

            [32],

            [32, 16],

            [32, 16, 8]
        ],


        # ReLU e Tanh são obrigatórias.
        # Mish foi escolhida como terceira função.
        "activation": [

            ReLU,

            Tanh,

            Mish
        ],


        # Mantido fixo para controlar
        # o tamanho da grade.
        "initializer": [

            HeInitializer
        ],


        # Dois otimizadores
        "optimizer": [

            Adam,

            RMSProp
        ],


        # Dois learning rates
        "learning_rate": [

            0.001,

            0.01
        ],


        # Duas quantidades de épocas
        "epochs": [

            50,

            100
        ],


        # Loss mantida constante
        "loss": [

            MSE
        ]
    }


    # ========================================================
    # EXECUÇÃO DO GRID SEARCH
    # ========================================================

    search = GridSearch(
        network_builder=build_network,
        param_grid=param_grid,
        random_seed=42
    )


    print("\n")
    print("=" * 60)

    print(
        "INICIANDO GRID SEARCH DEFINITIVO"
    )

    print("=" * 60)


    search.fit(
        X_train,
        y_train,
        X_val,
        y_val
    )


    search.summary()


    # ========================================================
    # AVALIAÇÃO FINAL NO TESTE
    # ========================================================

    print("\n")

    print("=" * 60)

    print(
        "AVALIAÇÃO DO MELHOR MODELO NO CONJUNTO DE TESTE"
    )

    print("=" * 60)


    best_network = (
        search.best_model
    )


    predictions = (
        best_network.forward(
            X_test
        )
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


    print(
        "\nMétricas finais no conjunto de teste:"
    )


    print(
        f"Accuracy  = {test_accuracy:.4f}"
    )

    print(
        f"Precision = {test_precision:.4f}"
    )

    print(
        f"Recall    = {test_recall:.4f}"
    )

    print(
        f"F1-score  = {test_f1:.4f}"
    )

    print(
        f"Loss      = {test_loss:.6f}"
    )


    # ========================================================
    # MATRIZ DE CONFUSÃO
    # ========================================================

    classes = (
        predictions > 0.5
    ).astype(int)


    tn = np.sum(
        (y_test == 0)
        &
        (classes == 0)
    )


    fp = np.sum(
        (y_test == 0)
        &
        (classes == 1)
    )


    fn = np.sum(
        (y_test == 1)
        &
        (classes == 0)
    )


    tp = np.sum(
        (y_test == 1)
        &
        (classes == 1)
    )


    print("\n")
    print("=" * 60)
    print("MATRIZ DE CONFUSÃO")
    print("=" * 60)


    print(
        "\n                 Predito"
    )

    print(
        "              0          1"
    )


    print(
        f"Real  0     {tn:6d}     {fp:6d}"
    )

    print(
        f"      1     {fn:6d}     {tp:6d}"
    )


    print("\nComponentes:")


    print(
        f"Verdadeiros Negativos (TN): {tn}"
    )

    print(
        f"Falsos Positivos      (FP): {fp}"
    )

    print(
        f"Falsos Negativos      (FN): {fn}"
    )

    print(
        f"Verdadeiros Positivos (TP): {tp}"
    )


if __name__ == "__main__":
    main()