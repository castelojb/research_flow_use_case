from gloe import transformer
from keras import Input, Model, Sequential
from keras.engine.base_layer import Layer
from keras.layers import Bidirectional, Dense, LSTM
from keras.optimizers import Adam
from keras.regularizers import L1L2
from pydantic import BaseModel, ConfigDict, PositiveInt, PositiveFloat


class KerasBiLSTMConfig(BaseModel):
    """
    Configuration class for a Keras Bidirectional LSTM (BiLSTM) model.

    Attributes:
        model_config (ConfigDict): Arbitrary configuration for the class.
        n_neurons (PositiveInt): Number of neurons in the LSTM layer.
        n_features (PositiveInt): Number of input features.
        n_lstm_layers (PositiveInt): Number of LSTM layers. Defaults to 1.
        l1 (PositiveFloat): L1 regularization strength. Defaults to 0.0001.
        l2 (PositiveFloat): L2 regularization strength. Defaults to 0.0001.
        dropout (PositiveFloat): Dropout rate. Defaults to 0.0.
        learning_rate (PositiveFloat): Learning rate. Defaults to 0.001.
        loss_func_name (str): Name of the loss function. Defaults to "mean_squared_error".
        with_compile (bool): Whether to compile the model. Defaults to True.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    n_neurons: PositiveInt
    n_features: PositiveInt
    n_lstm_layers: PositiveInt = 1
    l1: PositiveFloat = 0.0001
    l2: PositiveFloat = 0.0001
    dropout: PositiveFloat = 0.0
    learning_rate: PositiveFloat = 0.001
    loss_func_name: str = "mean_squared_error"
    with_compile: bool = True


@transformer
def keras_bilstm(model_config: KerasBiLSTMConfig) -> Model:
    """
    Returns a compiled Keras model with the specified number of neurons
    and number of LSTM layers. The model is a bidirectional LSTM model
    with a Dense output layer.

    Parameters
    ----------
    model_config : KerasBiLSTMConfig
        The configuration for the model. This includes the number of neurons,
        number of features, number of LSTM layers, L1 and L2 regularization
        strengths, dropout rate, learning rate, loss function name, and a
        boolean indicating whether or not to compile the model.

    Returns
    -------
    model : Model
        The compiled Keras model.
    """

    def build_lstm_block(n_lstm_layers: int) -> Layer:
        """
        Builds a sequence of bidirectional LSTM layers.

        Parameters
        ----------
        n_lstm_layers : int
            The number of LSTM layers to build.

        Returns
        -------
        layer : Layer
            A Sequential layer containing the built LSTM layers.
        """
        lstm_layers = [
            Bidirectional(
                LSTM(
                    model_config.n_neurons,
                    return_sequences=True,
                    kernel_regularizer=regularizer,
                    dropout=model_config.dropout,
                )
            )
            for _ in range(n_lstm_layers)
        ]

        return Sequential(lstm_layers)

    regularizer = L1L2(l1=model_config.l1, l2=model_config.l2)

    input_layer = Input(shape=(model_config.n_features, 1))

    x = build_lstm_block(model_config.n_lstm_layers)(input_layer)

    output_layer = Dense(1)(x)

    model = Model(inputs=[input_layer], outputs=output_layer)

    if model_config.with_compile:
        optimizer = Adam(learning_rate=model_config.learning_rate)
        model.compile(loss=model_config.loss_func_name, optimizer=optimizer)
    return model
