import itertools
import numpy as np

from mlp.metrics import (
    Accuracy,
    Precision,
    Recall,
    F1Score
)


class ExperimentResult:

    def __init__(
        self,
        params,
        score,
        precision,
        recall,
        f1,
        loss
    ):

        self.params = params
        self.score = score
        self.precision = precision
        self.recall = recall
        self.f1 = f1
        self.loss = loss


class GridSearch:
    """
    Grid Search para Redes Neurais.

    Testa todas as combinações possíveis
    dos hiperparâmetros informados.
    """

    def __init__(
        self,
        network_builder,
        param_grid,
        metric=Accuracy,
        random_seed=42
    ):

        self.network_builder = network_builder
        self.param_grid = param_grid
        self.metric = metric

        # Seed usada para garantir
        # reprodutibilidade entre as configurações.
        self.random_seed = random_seed

        self.results = []

        # Melhor configuração
        self.best_score = -1
        self.best_loss = float("inf")

        self.best_precision = 0
        self.best_recall = 0
        self.best_f1 = 0

        self.best_params = None
        self.best_model = None

        # Pior configuração
        self.worst_score = float("inf")
        self.worst_loss = -1

        self.worst_precision = 0
        self.worst_recall = 0
        self.worst_f1 = 0

        self.worst_params = None


    # ============================================================
    # TOTAL DE COMBINAÇÕES
    # ============================================================

    def total_combinations(self):

        total = 1

        for values in self.param_grid.values():

            total *= len(values)

        return total


    # ============================================================
    # GERAÇÃO DAS COMBINAÇÕES
    # ============================================================

    def generate_combinations(self):

        keys = list(
            self.param_grid.keys()
        )

        values = list(
            self.param_grid.values()
        )

        combinations = itertools.product(
            *values
        )

        for combo in combinations:

            yield dict(
                zip(
                    keys,
                    combo
                )
            )


    # ============================================================
    # EXECUÇÃO DO GRID SEARCH
    # ============================================================

    def fit(
        self,
        X_train,
        y_train,
        X_val,
        y_val
    ):

        total = self.total_combinations()

        print("\n")
        print("=" * 60)
        print("GRID SEARCH")
        print("=" * 60)

        print(
            f"Total de combinações a serem avaliadas: {total}"
        )

        configuration_number = 1


        for params in self.generate_combinations():

            print("\n")
            print("=" * 60)

            print(
                f"Configuração "
                f"{configuration_number}/{total}"
            )

            print("=" * 60)

            print(
                f"Arquitetura: "
                f"{params['arquitetura']}"
            )

            print(
                f"Ativação: "
                f"{params['activation'].__name__}"
            )

            print(
                f"Inicializador: "
                f"{params['initializer'].__name__}"
            )

            print(
                f"Otimizador: "
                f"{params['optimizer'].__name__}"
            )

            print(
                f"Learning Rate: "
                f"{params['learning_rate']}"
            )

            print(
                f"Épocas: "
                f"{params['epochs']}"
            )


            current_params = (
                params.copy()
            )


            # ====================================================
            # REPRODUTIBILIDADE
            # ====================================================
            #
            # Reinicia a seed antes de construir cada rede.
            #
            # Dessa forma, cada configuração parte de uma
            # sequência aleatória controlada.
            #
            # Arquiteturas diferentes naturalmente possuem
            # quantidades diferentes de pesos, mas a execução
            # continua sendo reproduzível.
            #

            np.random.seed(
                self.random_seed
            )


            # ==========================================
            # Construção da rede
            # ==========================================

            network = (
                self.network_builder(
                    current_params
                )
            )


            # ==========================================
            # Criação do otimizador
            # ==========================================

            optimizer = (
                current_params[
                    "optimizer"
                ](
                    lr=current_params[
                        "learning_rate"
                    ]
                )
            )


            # ==========================================
            # Função de perda
            # ==========================================

            loss_function = (
                current_params[
                    "loss"
                ]()
            )


            # ==========================================
            # Treinamento
            # ==========================================

            network.fit(
                X_train,
                y_train,
                epochs=current_params[
                    "epochs"
                ],
                loss_function=loss_function,
                optimizer=optimizer
            )


            # ==========================================
            # Avaliação no conjunto de validação
            # ==========================================

            predictions = (
                network.forward(
                    X_val
                )
            )


            score = (
                self.metric.calculate(
                    y_val,
                    predictions
                )
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


            loss = (
                loss_function.forward(
                    y_val,
                    predictions
                )
            )


            # ==========================================
            # Armazena os resultados
            # ==========================================

            self.results.append(
                ExperimentResult(
                    self.pretty_params(
                        current_params
                    ),
                    score,
                    precision,
                    recall,
                    f1,
                    loss
                )
            )


            # ==========================================
            # Resultado da configuração
            # ==========================================

            print("\nResultado:")

            print(
                f"Accuracy  = {score:.4f}"
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


            # ====================================================
            # MELHOR CONFIGURAÇÃO
            # ====================================================
            #
            # 1) Maior Accuracy
            # 2) Menor Loss em caso de empate
            #

            if (
                score > self.best_score
                or
                (
                    score == self.best_score
                    and loss < self.best_loss
                )
            ):

                self.best_score = score

                self.best_precision = precision
                self.best_recall = recall
                self.best_f1 = f1
                self.best_loss = loss

                self.best_params = (
                    current_params.copy()
                )

                self.best_model = network


            # ====================================================
            # PIOR CONFIGURAÇÃO
            # ====================================================
            #
            # 1) Menor Accuracy
            # 2) Maior Loss em caso de empate
            #

            if (
                score < self.worst_score
                or
                (
                    score == self.worst_score
                    and loss > self.worst_loss
                )
            ):

                self.worst_score = score

                self.worst_precision = precision
                self.worst_recall = recall
                self.worst_f1 = f1
                self.worst_loss = loss

                self.worst_params = (
                    current_params.copy()
                )


            configuration_number += 1


        return self


    # ============================================================
    # RESUMO
    # ============================================================

    def summary(self):

        print("\n")
        print("=" * 150)

        print(
            "RESULTADOS DO GRID SEARCH"
        )

        print("=" * 150)


        print(
            f"{'Config.':<9}"
            f"{'Arquitetura':<20}"
            f"{'Ativação':<12}"
            f"{'Inicializador':<18}"
            f"{'Otimizador':<14}"
            f"{'LR':<12}"
            f"{'Épocas':<10}"
            f"{'Accuracy':<12}"
            f"{'Precision':<12}"
            f"{'Recall':<12}"
            f"{'F1-score':<12}"
            f"{'Loss':<12}"
        )


        print("-" * 150)


        for i, result in enumerate(
            self.results,
            start=1
        ):

            arquitetura = (
                result.params[
                    "arquitetura"
                ]
            )

            activation = (
                result.params[
                    "activation"
                ]
            )

            initializer = (
                result.params[
                    "initializer"
                ]
            )

            optimizer = (
                result.params[
                    "optimizer"
                ]
            )

            learning_rate = (
                result.params[
                    "learning_rate"
                ]
            )

            epochs = (
                result.params[
                    "epochs"
                ]
            )


            print(
                f"{i:<9}"
                f"{str(arquitetura):<20}"
                f"{activation:<12}"
                f"{initializer:<18}"
                f"{optimizer:<14}"
                f"{learning_rate:<12}"
                f"{epochs:<10}"
                f"{result.score:<12.4f}"
                f"{result.precision:<12.4f}"
                f"{result.recall:<12.4f}"
                f"{result.f1:<12.4f}"
                f"{result.loss:<12.6f}"
            )


        print("=" * 150)


        print(
            f"\nTotal de configurações avaliadas: "
            f"{len(self.results)}"
        )


        # ========================================================
        # MELHOR CONFIGURAÇÃO
        # ========================================================

        print("\n")
        print("=" * 60)
        print("MELHOR CONFIGURAÇÃO")
        print("=" * 60)


        print(
            f"Parâmetros: "
            f"{self.pretty_params(self.best_params)}"
        )

        print(
            f"Accuracy: "
            f"{self.best_score:.4f}"
        )

        print(
            f"Precision: "
            f"{self.best_precision:.4f}"
        )

        print(
            f"Recall: "
            f"{self.best_recall:.4f}"
        )

        print(
            f"F1-score: "
            f"{self.best_f1:.4f}"
        )

        print(
            f"Loss: "
            f"{self.best_loss:.6f}"
        )


        # ========================================================
        # PIOR CONFIGURAÇÃO
        # ========================================================

        print("\n")
        print("=" * 60)
        print("PIOR CONFIGURAÇÃO")
        print("=" * 60)


        print(
            f"Parâmetros: "
            f"{self.pretty_params(self.worst_params)}"
        )

        print(
            f"Accuracy: "
            f"{self.worst_score:.4f}"
        )

        print(
            f"Precision: "
            f"{self.worst_precision:.4f}"
        )

        print(
            f"Recall: "
            f"{self.worst_recall:.4f}"
        )

        print(
            f"F1-score: "
            f"{self.worst_f1:.4f}"
        )

        print(
            f"Loss: "
            f"{self.worst_loss:.6f}"
        )


    # ============================================================
    # FORMATAÇÃO DOS PARÂMETROS
    # ============================================================

    def pretty_params(
        self,
        params
    ):

        pretty = {}

        for key, value in params.items():

            if key == "activation":

                pretty[key] = (
                    value.__name__
                )

            elif key == "optimizer":

                pretty[key] = (
                    value.__name__
                )

            elif key == "loss":

                pretty[key] = (
                    value.__name__
                )

            elif key == "initializer":

                pretty[key] = (
                    value.__name__
                )

            else:

                pretty[key] = value

        return pretty