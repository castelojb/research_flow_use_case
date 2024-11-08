from gloe import transformer
from keras import Input, Model, Sequential
from keras.engine.base_layer import Layer
from keras.layers import Bidirectional, Dense, LSTM
from keras.optimizers import Adam
from keras.regularizers import L1L2
from pydantic import BaseModel, ConfigDict, PositiveInt, PositiveFloat


class KerasBiLSTMConfig(BaseModel):
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
    def build_lstm_block(n_lstm_layers: int) -> Layer:
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
