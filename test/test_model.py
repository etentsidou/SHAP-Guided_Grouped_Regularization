import time, os, sys
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Embedding, Attention, Dense, Conv2D, Conv1D, Bidirectional, LSTM, Flatten, Input, Activation, Reshape, Dropout, Concatenate, AveragePooling1D, MaxPooling1D, BatchNormalization, Attention, GlobalAveragePooling1D, GlobalMaxPool1D, GRU, AdditiveAttention, AlphaDropout, LeakyReLU, concatenate, AveragePooling2D, MaxPooling2D, SeparableConv2D, MultiHeadAttention, Lambda
from tensorflow.keras.initializers import VarianceScaling
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.regularizers import L1

sys.path.append("../../../codes")
from positional_encoding import PositionalEncoding
from transformer_utils import add_encoder_layer, add_decoder_layer

def m81212_n13_with_epigenetic(VOCABULARY_SIZE=30, MAX_STEPS=24, EMBED_SIZE=7):
    embedding = Embedding(input_dim=VOCABULARY_SIZE, output_dim=EMBED_SIZE)
    position_encoding = PositionalEncoding(max_steps=MAX_STEPS, max_dims=EMBED_SIZE)

    inputs_1 = Input(shape=(MAX_STEPS,), name="input_1")
    embeddings_1 = embedding(inputs_1)
    positional_encoding_1 = position_encoding(embeddings_1)
    inputs_2 = Input(shape=(MAX_STEPS,), name="input_2")
    embeddings_2 = embedding(inputs_2)
    positional_encoding_2 = position_encoding(embeddings_2)
    inputs_3 = Input(shape=(MAX_STEPS,), name="input_3")
    embeddings_3 = embedding(inputs_3)
    positional_encoding_3 = position_encoding(embeddings_3)
    inputs_4 = Input(shape=(24, 4, ), name="input_4")
    inputs_5 = Input(shape=(24, 4, ), name="input_5")

    attention_1 = MultiHeadAttention(num_heads=8, key_dim=6)(positional_encoding_1, positional_encoding_1)
    attention_2 = MultiHeadAttention(num_heads=8, key_dim=6)(positional_encoding_2, positional_encoding_2)
    attention_3 = MultiHeadAttention(num_heads=8, key_dim=6)(positional_encoding_3, positional_encoding_3)
    # attention_out = MultiHeadAttention(num_heads=8, key_dim=6)(attention_2, attention_3, attention_1)
    # attention_out = Reshape(tuple([1, 24, EMBED_SIZE]))(attention_out)

    branch_1 = Reshape(tuple([24, EMBED_SIZE, 1]))(attention_1)
    conv_1 = Dropout(0.2)(branch_1)
    conv_1 = Conv2D(32, (1,4), padding='valid')(conv_1)
    conv_1 = BatchNormalization()(conv_1)
    conv_1 = Dropout(0.2)(conv_1)
    conv_1 = Conv2D(64, (1,4), padding='valid')(conv_1)
    conv_1 = BatchNormalization()(conv_1)
    conv_1 = Reshape(tuple([x for x in conv_1.shape.as_list() if x != 1 and x is not None]))(conv_1)
    conv_1_average = AveragePooling1D(data_format='channels_first')(conv_1)
    conv_1_max = MaxPooling1D(data_format='channels_first')(conv_1)
    conv_1 = Concatenate(axis=-1)([conv_1_average, conv_1_max])
    bidirectional_1_output = Bidirectional(LSTM(32, return_sequences=True, dropout=0.2))(conv_1)
    flatten_1 = Flatten()(bidirectional_1_output)

    branch_2 = Reshape(tuple([24, EMBED_SIZE, 1]))(attention_1)
    conv_2 = Dropout(0.2)(branch_2)
    conv_2 = Conv2D(64, (1,7), padding='valid')(conv_2)
    conv_2 = BatchNormalization()(conv_2)
    conv_2 = Reshape(tuple([x for x in conv_2.shape.as_list() if x != 1 and x is not None]))(conv_2)
    conv_2_average = AveragePooling1D(data_format='channels_first')(conv_2)
    conv_2_max = MaxPooling1D(data_format='channels_first')(conv_2)
    conv_2 = Concatenate(axis=-1)([conv_2_average, conv_2_max])
    bidirectional_2_output = Bidirectional(LSTM(32, return_sequences=True, dropout=0.2))(conv_2)
    flatten_2 = Flatten()(bidirectional_2_output)

    branch_3 = Reshape(tuple([24, EMBED_SIZE, 1]))(attention_2)
    conv_3 = Dropout(0.2)(branch_3)
    conv_3 = Conv2D(64, (1,7), padding='valid')(conv_3)
    conv_3 = BatchNormalization()(conv_3)
    conv_3 = Reshape(tuple([x for x in conv_3.shape.as_list() if x != 1 and x is not None]))(conv_3)
    bidirectional_3_output = Bidirectional(LSTM(32, return_sequences=True, dropout=0.2))(conv_3)
    flatten_3 = Flatten()(bidirectional_3_output)

    branch_4 = Reshape(tuple([24, EMBED_SIZE, 1]))(attention_3)
    conv_4 = Dropout(0.2)(branch_4)
    conv_4 = Conv2D(64, (1,7), padding='valid')(conv_4)
    conv_4 = BatchNormalization()(conv_4)
    conv_4 = Reshape(tuple([x for x in conv_4.shape.as_list() if x != 1 and x is not None]))(conv_4)
    bidirectional_4_output = Bidirectional(LSTM(32, return_sequences=True, dropout=0.2))(conv_4)
    flatten_4 = Flatten()(bidirectional_4_output)

    # con = Concatenate(axis=-1)([bidirectional_1_output, bidirectional_2_output, bidirectional_3_output, bidirectional_4_output])
    con = Concatenate(axis=-1)([flatten_1, flatten_2, flatten_3, flatten_4, Flatten()(inputs_4), Flatten()(inputs_5)])
    main = Flatten()(con)
    main = Dropout(0.2)(main)
    main = Dense(256)(main)
    main = BatchNormalization()(main)
    main = Dropout(0.2)(main)
    main = Dense(64)(main)
    main = BatchNormalization()(main)
    main = Dropout(0.8)(main)
    outputs = Dense(1, activation='sigmoid', name='output')(main)

    model = Model(inputs=[inputs_1, inputs_2, inputs_3, inputs_4, inputs_5], outputs=outputs)
    model.summary()
    return model

def m81212_n13_with_epigenetic_group_regularization(VOCABULARY_SIZE=30, MAX_STEPS=24, EMBED_SIZE=7):
    # Penalties per factor
    ctfc=0.07999968
    dnase=0.02666663
    h3k4me3=0.02898968
    rrbs=0.03045185

    embedding = Embedding(input_dim=VOCABULARY_SIZE, output_dim=EMBED_SIZE)
    position_encoding = PositionalEncoding(max_steps=MAX_STEPS, max_dims=EMBED_SIZE)

    inputs_1 = Input(shape=(MAX_STEPS,), name="input_1")
    embeddings_1 = embedding(inputs_1)
    positional_encoding_1 = position_encoding(embeddings_1)
    inputs_2 = Input(shape=(MAX_STEPS,), name="input_2")
    embeddings_2 = embedding(inputs_2)
    positional_encoding_2 = position_encoding(embeddings_2)
    inputs_3 = Input(shape=(MAX_STEPS,), name="input_3")
    embeddings_3 = embedding(inputs_3)
    positional_encoding_3 = position_encoding(embeddings_3)
    on_target_ctcf = Input(shape=(24,), name="on_target_ctcf")  # CTCF epigetic factor's input
    off_target_ctcf = Input(shape=(24,), name="off_target_ctcf")  # CTCF epigetic factor's input
    on_target_dnase = Input(shape=(24,), name="on_target_dnase")  # DNase epigetic factor's input
    off_target_dnase = Input(shape=(24,), name="off_target_dnase")  # DNase epigetic factor's input
    on_target_h3k4me3 = Input(shape=(24,), name="on_target_h3k4me3")  # H3K4me3 epigetic factor's input
    off_target_h3k4me3 = Input(shape=(24,), name="off_target_h3k4me3")  # H3K4me3 epigetic factor's input
    on_target_rrbs = Input(shape=(24,), name="on_target_rrbs")  # RRBS epigetic factor's input
    off_target_rrbs = Input(shape=(24,), name="off_target_rrbs")  # RRBS epigetic factor's input
    # inputs_4 = Input(shape=(24, 3, ), name="input_4")  
    # inputs_5 = Input(shape=(24, 3, ), name="input_5") 

    # Apply L1 Regularization to CTCF
    ctcf_on_dense = Dense(64, activation='relu', activity_regularizer=L1(ctfc))(on_target_ctcf)
    ctcf_off_dense = Dense(64, activation='relu', activity_regularizer=L1(ctfc))(off_target_ctcf)
    ctcf_on_bn = BatchNormalization()(ctcf_on_dense)
    ctcf_off_bn = BatchNormalization()(ctcf_off_dense)
    ctcf_on_dropout = Dropout(0.2)(ctcf_on_bn) 
    ctcf_off_dropout = Dropout(0.2)(ctcf_off_bn) 

    # Apply L1 Regularization to DNase
    dnase_on_dense = Dense(64, activation='relu', activity_regularizer=L1(dnase))(on_target_dnase)
    dnase_off_dense = Dense(64, activation='relu', activity_regularizer=L1(dnase))(off_target_dnase)
    dnase_on_bn = BatchNormalization()(dnase_on_dense)
    dnase_off_bn = BatchNormalization()(dnase_off_dense)
    dnase_on_dropout = Dropout(0.2)(dnase_on_bn) 
    dnase_off_dropout = Dropout(0.2)(dnase_off_bn) 

    # Apply L1 Regularization to H3K4me3
    h3k4me3_on_dense = Dense(64, activation='relu', activity_regularizer=L1(h3k4me3))(on_target_h3k4me3)
    h3k4me3_off_dense = Dense(64, activation='relu', activity_regularizer=L1(h3k4me3))(off_target_h3k4me3)
    h3k4me3_on_bn = BatchNormalization()(h3k4me3_on_dense)
    h3k4me3_off_bn = BatchNormalization()(h3k4me3_off_dense)
    h3k4me3_on_dropout = Dropout(0.2)(h3k4me3_on_bn) 
    h3k4me3_off_dropout = Dropout(0.2)(h3k4me3_off_bn) 

    # Apply L1 Regularization to RRBS
    rrbs_on_dense = Dense(64, activation='relu', activity_regularizer=L1(rrbs))(on_target_rrbs)
    rrbs_off_dense = Dense(64, activation='relu', activity_regularizer=L1(rrbs))(off_target_rrbs)
    rrbs_on_bn = BatchNormalization()(rrbs_on_dense)
    rrbs_off_bn = BatchNormalization()(rrbs_off_dense)
    rrbs_on_dropout = Dropout(0.2)(rrbs_on_bn) 
    rrbs_off_dropout = Dropout(0.2)(rrbs_off_bn) 

    attention_1 = MultiHeadAttention(num_heads=8, key_dim=6)(positional_encoding_1, positional_encoding_1)
    attention_2 = MultiHeadAttention(num_heads=8, key_dim=6)(positional_encoding_2, positional_encoding_2)
    attention_3 = MultiHeadAttention(num_heads=8, key_dim=6)(positional_encoding_3, positional_encoding_3)
    # attention_out = MultiHeadAttention(num_heads=8, key_dim=6)(attention_2, attention_3, attention_1)
    # attention_out = Reshape(tuple([1, 24, EMBED_SIZE]))(attention_out)

    branch_1 = Reshape(tuple([24, EMBED_SIZE, 1]))(attention_1)
    conv_1 = Dropout(0.2)(branch_1)
    conv_1 = Conv2D(32, (1,4), padding='valid')(conv_1)
    conv_1 = BatchNormalization()(conv_1)
    conv_1 = Dropout(0.2)(conv_1)
    conv_1 = Conv2D(64, (1,4), padding='valid')(conv_1)
    conv_1 = BatchNormalization()(conv_1)
    conv_1 = Reshape(tuple([x for x in conv_1.shape.as_list() if x != 1 and x is not None]))(conv_1)
    conv_1_average = AveragePooling1D(data_format='channels_first')(conv_1)
    conv_1_max = MaxPooling1D(data_format='channels_first')(conv_1)
    conv_1 = Concatenate(axis=-1)([conv_1_average, conv_1_max])
    bidirectional_1_output = Bidirectional(LSTM(32, return_sequences=True, dropout=0.2))(conv_1)
    flatten_1 = Flatten()(bidirectional_1_output)

    branch_2 = Reshape(tuple([24, EMBED_SIZE, 1]))(attention_1)
    conv_2 = Dropout(0.2)(branch_2)
    conv_2 = Conv2D(64, (1,7), padding='valid')(conv_2)
    conv_2 = BatchNormalization()(conv_2)
    conv_2 = Reshape(tuple([x for x in conv_2.shape.as_list() if x != 1 and x is not None]))(conv_2)
    conv_2_average = AveragePooling1D(data_format='channels_first')(conv_2)
    conv_2_max = MaxPooling1D(data_format='channels_first')(conv_2)
    conv_2 = Concatenate(axis=-1)([conv_2_average, conv_2_max])
    bidirectional_2_output = Bidirectional(LSTM(32, return_sequences=True, dropout=0.2))(conv_2)
    flatten_2 = Flatten()(bidirectional_2_output)

    branch_3 = Reshape(tuple([24, EMBED_SIZE, 1]))(attention_2)
    conv_3 = Dropout(0.2)(branch_3)
    conv_3 = Conv2D(64, (1,7), padding='valid')(conv_3)
    conv_3 = BatchNormalization()(conv_3)
    conv_3 = Reshape(tuple([x for x in conv_3.shape.as_list() if x != 1 and x is not None]))(conv_3)
    bidirectional_3_output = Bidirectional(LSTM(32, return_sequences=True, dropout=0.2))(conv_3)
    flatten_3 = Flatten()(bidirectional_3_output)

    branch_4 = Reshape(tuple([24, EMBED_SIZE, 1]))(attention_3)
    conv_4 = Dropout(0.2)(branch_4)
    conv_4 = Conv2D(64, (1,7), padding='valid')(conv_4)
    conv_4 = BatchNormalization()(conv_4)
    conv_4 = Reshape(tuple([x for x in conv_4.shape.as_list() if x != 1 and x is not None]))(conv_4)
    bidirectional_4_output = Bidirectional(LSTM(32, return_sequences=True, dropout=0.2))(conv_4)
    flatten_4 = Flatten()(bidirectional_4_output)

    # con = Concatenate(axis=-1)([bidirectional_1_output, bidirectional_2_output, bidirectional_3_output, bidirectional_4_output])
    # con = Concatenate(axis=-1)([flatten_1, flatten_2, flatten_3, flatten_4, Flatten()(inputs_4), Flatten()(inputs_5)])
    # Used for the model with L1 Regularization applied to the least important epigenetic factor
    con = Concatenate(axis=-1)([flatten_1, flatten_2, flatten_3, flatten_4, ctcf_on_dropout, ctcf_off_dropout, dnase_on_dropout, dnase_off_dropout, h3k4me3_on_dropout, h3k4me3_off_dropout, rrbs_on_dropout, rrbs_off_dropout])
    main = Flatten()(con)
    main = Dropout(0.2)(main)
    main = Dense(256)(main)
    main = BatchNormalization()(main)
    main = Dropout(0.2)(main)
    main = Dense(64)(main)
    main = BatchNormalization()(main)
    main = Dropout(0.8)(main)
    outputs = Dense(1, activation='sigmoid', name='output')(main)

    model = Model(inputs=[inputs_1, inputs_2, inputs_3, on_target_ctcf, off_target_ctcf, on_target_dnase, off_target_dnase, on_target_h3k4me3, off_target_h3k4me3, on_target_rrbs, off_target_rrbs], outputs=outputs)
    model.summary()
    return model

def m81212_n13_with_epigenetic_group_regularization_no_shap_information(VOCABULARY_SIZE=30, MAX_STEPS=24, EMBED_SIZE=7):
    # Uniform Penalty for Epigenetic Data
    reg_strength = 0.01

    embedding = Embedding(input_dim=VOCABULARY_SIZE, output_dim=EMBED_SIZE)
    position_encoding = PositionalEncoding(max_steps=MAX_STEPS, max_dims=EMBED_SIZE)

    inputs_1 = Input(shape=(MAX_STEPS,), name="input_1")
    embeddings_1 = embedding(inputs_1)
    positional_encoding_1 = position_encoding(embeddings_1)
    inputs_2 = Input(shape=(MAX_STEPS,), name="input_2")
    embeddings_2 = embedding(inputs_2)
    positional_encoding_2 = position_encoding(embeddings_2)
    inputs_3 = Input(shape=(MAX_STEPS,), name="input_3")
    embeddings_3 = embedding(inputs_3)
    positional_encoding_3 = position_encoding(embeddings_3)
    on_target_ctcf = Input(shape=(24,), name="on_target_ctcf")  # CTCF epigetic factor's input
    off_target_ctcf = Input(shape=(24,), name="off_target_ctcf")  # CTCF epigetic factor's input
    on_target_dnase = Input(shape=(24,), name="on_target_dnase")  # DNase epigetic factor's input
    off_target_dnase = Input(shape=(24,), name="off_target_dnase")  # DNase epigetic factor's input
    on_target_h3k4me3 = Input(shape=(24,), name="on_target_h3k4me3")  # H3K4me3 epigetic factor's input
    off_target_h3k4me3 = Input(shape=(24,), name="off_target_h3k4me3")  # H3K4me3 epigetic factor's input
    on_target_rrbs = Input(shape=(24,), name="on_target_rrbs")  # RRBS epigetic factor's input
    off_target_rrbs = Input(shape=(24,), name="off_target_rrbs")  # RRBS epigetic factor's input
    # inputs_4 = Input(shape=(24, 3, ), name="input_4")  
    # inputs_5 = Input(shape=(24, 3, ), name="input_5") 

    # Apply L1 Regularization to CTCF
    ctcf_on_dense = Dense(64, activation='relu', activity_regularizer=L1(reg_strength))(on_target_ctcf)
    ctcf_off_dense = Dense(64, activation='relu', activity_regularizer=L1(reg_strength))(off_target_ctcf)
    ctcf_on_bn = BatchNormalization()(ctcf_on_dense)
    ctcf_off_bn = BatchNormalization()(ctcf_off_dense)
    ctcf_on_dropout = Dropout(0.2)(ctcf_on_bn) 
    ctcf_off_dropout = Dropout(0.2)(ctcf_off_bn) 

    # Apply L1 Regularization to DNase
    dnase_on_dense = Dense(64, activation='relu', activity_regularizer=L1(reg_strength))(on_target_dnase)
    dnase_off_dense = Dense(64, activation='relu', activity_regularizer=L1(reg_strength))(off_target_dnase)
    dnase_on_bn = BatchNormalization()(dnase_on_dense)
    dnase_off_bn = BatchNormalization()(dnase_off_dense)
    dnase_on_dropout = Dropout(0.2)(dnase_on_bn) 
    dnase_off_dropout = Dropout(0.2)(dnase_off_bn) 

    # Apply L1 Regularization to H3K4me3
    h3k4me3_on_dense = Dense(64, activation='relu', activity_regularizer=L1(reg_strength))(on_target_h3k4me3)
    h3k4me3_off_dense = Dense(64, activation='relu', activity_regularizer=L1(reg_strength))(off_target_h3k4me3)
    h3k4me3_on_bn = BatchNormalization()(h3k4me3_on_dense)
    h3k4me3_off_bn = BatchNormalization()(h3k4me3_off_dense)
    h3k4me3_on_dropout = Dropout(0.2)(h3k4me3_on_bn) 
    h3k4me3_off_dropout = Dropout(0.2)(h3k4me3_off_bn) 

    # Apply L1 Regularization to RRBS
    rrbs_on_dense = Dense(64, activation='relu', activity_regularizer=L1(reg_strength))(on_target_rrbs)
    rrbs_off_dense = Dense(64, activation='relu', activity_regularizer=L1(reg_strength))(off_target_rrbs)
    rrbs_on_bn = BatchNormalization()(rrbs_on_dense)
    rrbs_off_bn = BatchNormalization()(rrbs_off_dense)
    rrbs_on_dropout = Dropout(0.2)(rrbs_on_bn) 
    rrbs_off_dropout = Dropout(0.2)(rrbs_off_bn) 

    attention_1 = MultiHeadAttention(num_heads=8, key_dim=6)(positional_encoding_1, positional_encoding_1)
    attention_2 = MultiHeadAttention(num_heads=8, key_dim=6)(positional_encoding_2, positional_encoding_2)
    attention_3 = MultiHeadAttention(num_heads=8, key_dim=6)(positional_encoding_3, positional_encoding_3)
    # attention_out = MultiHeadAttention(num_heads=8, key_dim=6)(attention_2, attention_3, attention_1)
    # attention_out = Reshape(tuple([1, 24, EMBED_SIZE]))(attention_out)

    branch_1 = Reshape(tuple([24, EMBED_SIZE, 1]))(attention_1)
    conv_1 = Dropout(0.2)(branch_1)
    conv_1 = Conv2D(32, (1,4), padding='valid')(conv_1)
    conv_1 = BatchNormalization()(conv_1)
    conv_1 = Dropout(0.2)(conv_1)
    conv_1 = Conv2D(64, (1,4), padding='valid')(conv_1)
    conv_1 = BatchNormalization()(conv_1)
    conv_1 = Reshape(tuple([x for x in conv_1.shape.as_list() if x != 1 and x is not None]))(conv_1)
    conv_1_average = AveragePooling1D(data_format='channels_first')(conv_1)
    conv_1_max = MaxPooling1D(data_format='channels_first')(conv_1)
    conv_1 = Concatenate(axis=-1)([conv_1_average, conv_1_max])
    bidirectional_1_output = Bidirectional(LSTM(32, return_sequences=True, dropout=0.2))(conv_1)
    flatten_1 = Flatten()(bidirectional_1_output)

    branch_2 = Reshape(tuple([24, EMBED_SIZE, 1]))(attention_1)
    conv_2 = Dropout(0.2)(branch_2)
    conv_2 = Conv2D(64, (1,7), padding='valid')(conv_2)
    conv_2 = BatchNormalization()(conv_2)
    conv_2 = Reshape(tuple([x for x in conv_2.shape.as_list() if x != 1 and x is not None]))(conv_2)
    conv_2_average = AveragePooling1D(data_format='channels_first')(conv_2)
    conv_2_max = MaxPooling1D(data_format='channels_first')(conv_2)
    conv_2 = Concatenate(axis=-1)([conv_2_average, conv_2_max])
    bidirectional_2_output = Bidirectional(LSTM(32, return_sequences=True, dropout=0.2))(conv_2)
    flatten_2 = Flatten()(bidirectional_2_output)

    branch_3 = Reshape(tuple([24, EMBED_SIZE, 1]))(attention_2)
    conv_3 = Dropout(0.2)(branch_3)
    conv_3 = Conv2D(64, (1,7), padding='valid')(conv_3)
    conv_3 = BatchNormalization()(conv_3)
    conv_3 = Reshape(tuple([x for x in conv_3.shape.as_list() if x != 1 and x is not None]))(conv_3)
    bidirectional_3_output = Bidirectional(LSTM(32, return_sequences=True, dropout=0.2))(conv_3)
    flatten_3 = Flatten()(bidirectional_3_output)

    branch_4 = Reshape(tuple([24, EMBED_SIZE, 1]))(attention_3)
    conv_4 = Dropout(0.2)(branch_4)
    conv_4 = Conv2D(64, (1,7), padding='valid')(conv_4)
    conv_4 = BatchNormalization()(conv_4)
    conv_4 = Reshape(tuple([x for x in conv_4.shape.as_list() if x != 1 and x is not None]))(conv_4)
    bidirectional_4_output = Bidirectional(LSTM(32, return_sequences=True, dropout=0.2))(conv_4)
    flatten_4 = Flatten()(bidirectional_4_output)

    # con = Concatenate(axis=-1)([bidirectional_1_output, bidirectional_2_output, bidirectional_3_output, bidirectional_4_output])
    # con = Concatenate(axis=-1)([flatten_1, flatten_2, flatten_3, flatten_4, Flatten()(inputs_4), Flatten()(inputs_5)])
    # Used for the model with L1 Regularization applied to the least important epigenetic factor
    con = Concatenate(axis=-1)([flatten_1, flatten_2, flatten_3, flatten_4, ctcf_on_dropout, ctcf_off_dropout, dnase_on_dropout, dnase_off_dropout, h3k4me3_on_dropout, h3k4me3_off_dropout, rrbs_on_dropout, rrbs_off_dropout])
    main = Flatten()(con)
    main = Dropout(0.2)(main)
    main = Dense(256)(main)
    main = BatchNormalization()(main)
    main = Dropout(0.2)(main)
    main = Dense(64)(main)
    main = BatchNormalization()(main)
    main = Dropout(0.8)(main)
    outputs = Dense(1, activation='sigmoid', name='output')(main)

    model = Model(inputs=[inputs_1, inputs_2, inputs_3, on_target_ctcf, off_target_ctcf, on_target_dnase, off_target_dnase, on_target_h3k4me3, off_target_h3k4me3, on_target_rrbs, off_target_rrbs], outputs=outputs)
    model.summary()
    return model

def m81212_n13_with_group_regularization(VOCABULARY_SIZE=30, MAX_STEPS=24, EMBED_SIZE=7):
    # Penalties for Each Data Group
    features=0.001
    feature_ont=0.00829255
    feature_offt=0.00803434
    on_ctcf=0.01
    off_ctcf=0.00921985
    on_dnase=0.00977774
    off_dnase=0.00968546
    on_h3k4me3=0.00907265
    off_h3k4me3=0.00867075
    on_rrbs=0.00850309
    off_rrbs=0.00844418

    embedding = Embedding(input_dim=VOCABULARY_SIZE, output_dim=EMBED_SIZE)
    position_encoding = PositionalEncoding(max_steps=MAX_STEPS, max_dims=EMBED_SIZE)

    inputs_1 = Input(shape=(MAX_STEPS,), name="input_1")
    embeddings_1 = embedding(inputs_1)
    positional_encoding_1 = position_encoding(embeddings_1)
    positional_encoding_1 = Dense(EMBED_SIZE, activity_regularizer=L1(features))(positional_encoding_1)

    inputs_2 = Input(shape=(MAX_STEPS,), name="input_2")
    embeddings_2 = embedding(inputs_2)
    positional_encoding_2 = position_encoding(embeddings_2)
    positional_encoding_2 = Dense(EMBED_SIZE, activity_regularizer=L1(feature_ont))(positional_encoding_2)

    inputs_3 = Input(shape=(MAX_STEPS,), name="input_3")
    embeddings_3 = embedding(inputs_3)
    positional_encoding_3 = position_encoding(embeddings_3)
    positional_encoding_3 = Dense(EMBED_SIZE, activity_regularizer=L1(feature_offt))(positional_encoding_3)

    on_target_ctcf = Input(shape=(24,), name="on_target_ctcf")  # CTCF epigetic factor's input
    off_target_ctcf = Input(shape=(24,), name="off_target_ctcf")  # CTCF epigetic factor's input
    on_target_dnase = Input(shape=(24,), name="on_target_dnase")  # DNase epigetic factor's input
    off_target_dnase = Input(shape=(24,), name="off_target_dnase")  # DNase epigetic factor's input
    on_target_h3k4me3 = Input(shape=(24,), name="on_target_h3k4me3")  # H3K4me3 epigetic factor's input
    off_target_h3k4me3 = Input(shape=(24,), name="off_target_h3k4me3")  # H3K4me3 epigetic factor's input
    on_target_rrbs = Input(shape=(24,), name="on_target_rrbs")  # RRBS epigetic factor's input
    off_target_rrbs = Input(shape=(24,), name="off_target_rrbs")  # RRBS epigetic factor's input
    # inputs_4 = Input(shape=(24, 3, ), name="input_4")  
    # inputs_5 = Input(shape=(24, 3, ), name="input_5") 

    ctcf_on_dense = Dense(64, activation='relu', activity_regularizer=L1(on_ctcf))(on_target_ctcf)
    ctcf_off_dense = Dense(64, activation='relu', activity_regularizer=L1(off_ctcf))(off_target_ctcf)
    # ctcf_on_dense = Dense(64, activation='relu')(on_target_ctcf)
    # ctcf_off_dense = Dense(64, activation='relu')(off_target_ctcf)
    # ctcf_on_bn = BatchNormalization()(ctcf_on_dense)
    # ctcf_off_bn = BatchNormalization()(ctcf_off_dense)
    # ctcf_on_dropout = Dropout(0.2)(ctcf_on_bn) 
    # ctcf_off_dropout = Dropout(0.2)(ctcf_off_bn) 

    dnase_on_dense = Dense(64, activation='relu', activity_regularizer=L1(on_dnase))(on_target_dnase)
    dnase_off_dense = Dense(64, activation='relu', activity_regularizer=L1(off_dnase))(off_target_dnase)
    # dnase_on_dense = Dense(64, activation='relu')(on_target_dnase)
    # dnase_off_dense = Dense(64, activation='relu')(off_target_dnase)
    # dnase_on_bn = BatchNormalization()(dnase_on_dense)
    # dnase_off_bn = BatchNormalization()(dnase_off_dense)
    # dnase_on_dropout = Dropout(0.2)(dnase_on_bn) 
    # dnase_off_dropout = Dropout(0.2)(dnase_off_bn) 

    h3k4me3_on_dense = Dense(64, activation='relu', activity_regularizer=L1(on_h3k4me3))(on_target_h3k4me3)
    h3k4me3_off_dense = Dense(64, activation='relu', activity_regularizer=L1(off_h3k4me3))(off_target_h3k4me3)
    # h3k4me3_on_dense = Dense(64, activation='relu')(on_target_h3k4me3)
    # h3k4me3_off_dense = Dense(64, activation='relu')(off_target_h3k4me3)
    # h3k4me3_on_bn = BatchNormalization()(h3k4me3_on_dense)
    # h3k4me3_off_bn = BatchNormalization()(h3k4me3_off_dense)
    # h3k4me3_on_dropout = Dropout(0.2)(h3k4me3_on_bn) 
    # h3k4me3_off_dropout = Dropout(0.2)(h3k4me3_off_bn) 

    rrbs_on_dense = Dense(64, activation='relu', activity_regularizer=L1(on_rrbs))(on_target_rrbs)
    rrbs_off_dense = Dense(64, activation='relu', activity_regularizer=L1(off_rrbs))(off_target_rrbs)
    # rrbs_on_dense = Dense(64, activation='relu')(on_target_rrbs)
    # rrbs_off_dense = Dense(64, activation='relu')(off_target_rrbs)
    # rrbs_on_bn = BatchNormalization()(rrbs_on_dense)
    # rrbs_off_bn = BatchNormalization()(rrbs_off_dense)
    # rrbs_on_dropout = Dropout(0.2)(rrbs_on_bn) 
    # rrbs_off_dropout = Dropout(0.2)(rrbs_off_bn) 

    attention_1 = MultiHeadAttention(num_heads=8, key_dim=6)(positional_encoding_1, positional_encoding_1)
    attention_2 = MultiHeadAttention(num_heads=8, key_dim=6)(positional_encoding_2, positional_encoding_2)
    attention_3 = MultiHeadAttention(num_heads=8, key_dim=6)(positional_encoding_3, positional_encoding_3)
    # attention_out = MultiHeadAttention(num_heads=8, key_dim=6)(attention_2, attention_3, attention_1)
    # attention_out = Reshape(tuple([1, 24, EMBED_SIZE]))(attention_out)

    branch_1 = Reshape(tuple([24, EMBED_SIZE, 1]))(attention_1)
    # conv_1 = Dropout(0.2)(branch_1)
    # conv_1 = Conv2D(32, (1,4), padding='valid')(conv_1)
    conv_1 = Conv2D(32, (1,4), padding='valid')(branch_1)
    # conv_1 = BatchNormalization()(conv_1)
    # conv_1 = Dropout(0.2)(conv_1)
    conv_1 = Conv2D(64, (1,4), padding='valid')(conv_1)
    # conv_1 = BatchNormalization()(conv_1)
    conv_1 = Reshape(tuple([x for x in conv_1.shape.as_list() if x != 1 and x is not None]))(conv_1)
    conv_1_average = AveragePooling1D(data_format='channels_first')(conv_1)
    conv_1_max = MaxPooling1D(data_format='channels_first')(conv_1)
    conv_1 = Concatenate(axis=-1)([conv_1_average, conv_1_max])
    bidirectional_1_output = Bidirectional(LSTM(32, return_sequences=True))(conv_1)
    # bidirectional_1_output = Bidirectional(LSTM(32, return_sequences=True, dropout=0.2))(conv_1)
    flatten_1 = Flatten()(bidirectional_1_output)

    branch_2 = Reshape(tuple([24, EMBED_SIZE, 1]))(attention_1)
    # conv_2 = Dropout(0.2)(branch_2)
    conv_2 = Conv2D(64, (1,7), padding='valid')(branch_2)
    # conv_2 = Conv2D(64, (1,7), padding='valid')(conv_2)
    # conv_2 = BatchNormalization()(conv_2)
    conv_2 = Reshape(tuple([x for x in conv_2.shape.as_list() if x != 1 and x is not None]))(conv_2)
    conv_2_average = AveragePooling1D(data_format='channels_first')(conv_2)
    conv_2_max = MaxPooling1D(data_format='channels_first')(conv_2)
    conv_2 = Concatenate(axis=-1)([conv_2_average, conv_2_max])
    # bidirectional_2_output = Bidirectional(LSTM(32, return_sequences=True, dropout=0.2))(conv_2)
    bidirectional_2_output = Bidirectional(LSTM(32, return_sequences=True))(conv_2)
    flatten_2 = Flatten()(bidirectional_2_output)

    branch_3 = Reshape(tuple([24, EMBED_SIZE, 1]))(attention_2)
    # conv_3 = Dropout(0.2)(branch_3)
    # conv_3 = Conv2D(64, (1,7), padding='valid')(conv_3)
    conv_3 = Conv2D(64, (1,7), padding='valid')(branch_3)
    # conv_3 = BatchNormalization()(conv_3)
    conv_3 = Reshape(tuple([x for x in conv_3.shape.as_list() if x != 1 and x is not None]))(conv_3)
    # bidirectional_3_output = Bidirectional(LSTM(32, return_sequences=True, dropout=0.2))(conv_3)
    bidirectional_3_output = Bidirectional(LSTM(32, return_sequences=True))(conv_3)
    flatten_3 = Flatten()(bidirectional_3_output)

    branch_4 = Reshape(tuple([24, EMBED_SIZE, 1]))(attention_3)
    # conv_4 = Dropout(0.2)(branch_4)
    # conv_4 = Conv2D(64, (1,7), padding='valid')(conv_4)
    conv_4 = Conv2D(64, (1,7), padding='valid')(branch_4)
    # conv_4 = BatchNormalization()(conv_4)
    conv_4 = Reshape(tuple([x for x in conv_4.shape.as_list() if x != 1 and x is not None]))(conv_4)
    # bidirectional_4_output = Bidirectional(LSTM(32, return_sequences=True, dropout=0.2))(conv_4)
    bidirectional_4_output = Bidirectional(LSTM(32, return_sequences=True))(conv_4)
    flatten_4 = Flatten()(bidirectional_4_output)

    # con = Concatenate(axis=-1)([bidirectional_1_output, bidirectional_2_output, bidirectional_3_output, bidirectional_4_output])
    # con = Concatenate(axis=-1)([flatten_1, flatten_2, flatten_3, flatten_4, Flatten()(inputs_4), Flatten()(inputs_5)])
    # Used for the model with L1 Regularization applied to the least important epigenetic factor
    # con = Concatenate(axis=-1)([flatten_1, flatten_2, flatten_3, flatten_4, ctcf_on_dense, ctcf_off_dense, dnase_on_dense, dnase_off_dense, h3k4me3_on_dense, h3k4me3_off_dense, rrbs_on_dense, rrbs_off_dense])
  
    
    con = Concatenate(axis=-1)([flatten_1, flatten_2, flatten_3, flatten_4, ctcf_on_dense, ctcf_off_dense, dnase_on_dense, dnase_off_dense, h3k4me3_on_dense, h3k4me3_off_dense, rrbs_on_dense, rrbs_off_dense])
    main = Flatten()(con)
    # main = Dropout(0.2)(main)
    main = Dense(256)(main)
    # main = BatchNormalization()(main)
    # main = Dropout(0.2)(main)
    main = Dense(64)(main)
    # main = BatchNormalization()(main)
    # main = Dropout(0.8)(main)
    outputs = Dense(1, activation='sigmoid', name='output')(main)

    model = Model(inputs=[inputs_1, inputs_2, inputs_3, on_target_ctcf, off_target_ctcf, on_target_dnase, off_target_dnase, on_target_h3k4me3, off_target_h3k4me3, on_target_rrbs, off_target_rrbs], outputs=outputs)
    model.summary()
    return model