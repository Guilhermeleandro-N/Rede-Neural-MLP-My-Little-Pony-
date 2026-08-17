import itertools

from mlp.metrics import (
    Accuracy,
    Precision,
    Recall,
    F1Score

)

from mlp.optimizers import Adam

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
        metric=Accuracy
    ):

        self.network_builder = (
            network_builder
        )

        self.param_grid = param_grid

        self.metric = metric

        self.results = []

        self.best_score = -1

        self.best_loss = float("inf")

        self.best_precision = 0
        self.best_recall = 0
        self.best_f1 = 0

        self.best_params = None

        self.best_model = None

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

    def fit(
        self,
        X,
        y
    ):

#de "experiment" para "configuration" para nao nos confundirmos com os experimentos do documento
        configuration_number = 1

        for params in self.generate_combinations():

            print(
                f"\nConfiguração {configuration_number}"
            )

            print(
                f"Arquitetura: {params['arquitetura']}"
            )

            print(
                f"Ativação: {params['activation'].__name__}"
            )

            print(
                f"Inicializador: {params['initializer'].__name__}"
            )

            print(
                f"Otimizador: {params['optimizer']}"
            )

            print(
                f"Learning rate: {params['learning_rate']}"
            )

            print(
                f"Épocas: {params['epochs']}"
            )

            configuration_number += 1

            current_params = (
                params.copy()
            )

            network = (
                self.network_builder(
                    current_params
                )
            )

            if current_params["optimizer"] == "Adam":

                optimizer = Adam(
                lr=current_params["learning_rate"]
    )

            loss_function = (
                current_params[
                    "loss"
                ]()
            )

            network.fit(
                X,
                y,
                epochs=current_params[
                    "epochs"
                ],
                loss_function=loss_function,
                optimizer=optimizer
            )

            predictions = (
                network.forward(X)
            )

            score = (
                self.metric.calculate(
                    y,
                    predictions
                )
            )

            precision = Precision.calculate(
                y,
                predictions
            )

            recall = Recall.calculate(
                y,
                predictions
            )

            f1 = F1Score.calculate(
                y,
                predictions
            )

            loss = (
                loss_function.forward(
                    y,
                    predictions
                )
            )

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

            print(
                f"Accuracy = {score:.4f}"
            )

            print(
                f"Precision = {precision:.4f}"
            )

            print(
                f"Recall = {recall:.4f}"
            )

            print(
                f"F1-score = {f1:.4f}"
            )

            print(
                f"Loss Final = {loss:.6f}"
            )

            #
            # Critério de desempate
            #
            # 1) maior accuracy
            # 2) menor loss
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

                #
                # Guarda a rede treinada
                #
                self.best_model = network

        return self

    def summary(self):

        print("\n")
        print("=" * 100)
        print("RESULTADOS")
        print(f"Total de configurações avaliadas: {len(self.results)}")
        print("=" * 100)

        print(
            f"{'Config.':<10}"
            f"{'Arquitetura':<20}"
            f"{'Ativação':<12}"
            f"{'Inicializador':<16}"
            f"{'Otimizador':<12}"
            f"{'LR':<10}"
            f"{'Épocas':<10}"
            f"{'Accuracy':<12}"
            f"{'Precision':<12}"
            f"{'Recall':<12}"
            f"{'F1-score':<12}"
            f"{'Loss':<12}"
        )

        print("-" * 100)

        for i, result in enumerate(self.results, start=1):

            arquitetura = result.params["arquitetura"]
            activation = result.params["activation"]
            initializer = result.params["initializer"]
            optimizer = result.params["optimizer"]
            learning_rate = result.params["learning_rate"]
            epochs = result.params["epochs"]

            print(
                f"{i:<10}"
                f"{str(arquitetura):<20}"
                f"{activation:<12}"
                f"{initializer:<16}"
                f"{optimizer:<12}"
                f"{learning_rate:<10}"
                f"{epochs:<10}"
                f"{result.score:<12.4f}"
                f"{result.precision:<12.4f}"
                f"{result.recall:<12.4f}"
                f"{result.f1:<12.4f}"
                f"{result.loss:<12.6f}"
            )

        print("=" * 100)

        print(
            f"Parâmetros: "
            f"{self.pretty_params(self.best_params)}"
        )


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

                pretty[key] = value

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
