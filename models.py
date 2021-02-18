# TCN+LSTM+CRF  
from keras.models import Sequential
from tcn import TCN
import keras.backend as K
from tensorflow.python.keras import backend as k
from keras.layers.normalization import BatchNormalization
from keras import losses
from keras import metrics
from keras_contrib.metrics import crf_marginal_accuracy
from keras_contrib.losses import crf_loss
from keras.layers import LocallyConnected1D
from keras import regularizers
#from keras_self_attention import SeqSelfAttention,ScaledDotProductAttention,SeqWeightedAttention
def model_TCNN_CRF():
    model1 = Sequential()
    model1.add(Convolution1D(64, kernel_size=200, activation=activations.selu,strides=20, padding="valid", name='layer_11',input_shape=(3000, 1)))  
    model1.add(Convolution1D(64, kernel_size=6, activation=activations.selu, padding="valid", name='layer_12'))
    model1.add(MaxPool1D(pool_size=3,strides=3))
    #model1.add(SpatialDropout1D(rate=0.3))
    model1.add(Convolution1D(32, kernel_size=6, activation=activations.selu, padding="valid", name='layer_13'))
    model1.add(Convolution1D(32, kernel_size=6, activation=activations.selu, padding="valid", name='layer_14'))
    model1.add(MaxPool1D(pool_size=2,strides=2))
    model1.add(Convolution1D(32, kernel_size=3, activation=activations.selu, padding="valid", name='layer_15',kernel_regularizer= keras.regularizers.l1_l2(l1=0.0001, l2=0.001)))
    model1.add(Convolution1D(32, kernel_size=3, activation=activations.selu, padding="valid", name='layer_16'))
    model1.add(MaxPool1D(pool_size=2,strides=2))
    #model1.add(BatchNormalization())
    model1.add(SpatialDropout1D(rate=0.01))
    model1.add(LocallyConnected1D(128, kernel_size=3,activation=activations.selu, padding="valid", name='layer_17', kernel_regularizer=keras.regularizers.l1_l2(l1=0.0001, l2=0.0001)))
    #model1.add(Convolution1D(256, kernel_size=3, activation=activations.relu, padding="valid", name='layer_18', kernel_regularizer=regularizers.l2(0.0001)))
    #model1.add(GlobalMaxPool1D())
    convout2 = GlobalMaxPool1D()
    model1.add(convout2)
    model1.add(Dropout(rate=0.01))
    #model1.add((Dense(64, activation=activations.relu, name='layer_19')))
    #model1.compile(loss='categorical_crossentropy', optimizer=optimizer1, metrics=['accuracy'])
    #model1.compile(loss='sparse_categorical_crossentropy', optimizer=adam_with_lr_multipliers1, metrics=['accuracy'])

    model1.compile(optimizers.Adam(0.001), losses.sparse_categorical_crossentropy, metrics=['acc'])
    #for layer in model1.layers[:-5]:
     #    layer.trainable = False
    #model.summary()
    model3 = Sequential()
    model3.add(Convolution1D(64, kernel_size=25,strides=3, activation=activations.selu, padding="valid", name='layer_31', input_shape=(3000, 1)))  
    model3.add(Convolution1D(64, kernel_size=8, activation=activations.selu, padding="valid", name='layer_32'))
    #model3.add(BatchNormalization())
    model3.add(MaxPool1D(pool_size=4,strides=4))
    #model3.add(SpatialDropout1D(rate=0.5))
    model3.add(Convolution1D(32, kernel_size=8, activation=activations.selu, padding="valid", name='layer_33'))
    model3.add(Convolution1D(32, kernel_size=8, activation=activations.selu, padding="valid", name='layer_34'))
    model3.add(MaxPool1D(pool_size=2,strides=2))
    model3.add(Convolution1D(32, kernel_size=5, activation=activations.selu, padding="valid", name='layer_25',kernel_regularizer= keras.regularizers.l1_l2(l1=0.0001, l2=0.001)))
    model3.add(Convolution1D(32, kernel_size=5, activation=activations.selu, padding="valid", name='layer_26'))
    model3.add(MaxPool1D(pool_size=2,strides=2))
    model3.add(SpatialDropout1D(rate=0.01))
    model3.add(LocallyConnected1D(128, kernel_size=3, activation=activations.selu, padding="valid", name='layer_37',kernel_regularizer= keras.regularizers.l1_l2(l1=0.0001, l2=0.0001)))
    model3.add(GlobalMaxPool1D())
    model3.add(Dropout(rate=0.01))
    #model3.add((Dense(64, activation=activations.relu, name='layer_39')))
    model3.compile(optimizers.Adam(0.001), losses.sparse_categorical_crossentropy, metrics=['acc'])

    nclass = 5

    seq_input1 = Input(shape=(None, 3000, 1))
    
    seq_input3 = Input(shape=(None, 3000, 1))

    base_model1 = model1
 
    base_model3 = model3

    #for layer in base_model1.layers:
     # layer.trainable = False
    #This wrapper applies a layer to every temporal slice of an input.
    #for layer in base_model2.layers:
     #    layer.trainable = False
        
    #for layer in base_model3.layers:
     #    layer.trainable = False
    encoded_sequence1 = TimeDistributed(base_model1)(seq_input1)
    encoded_sequence3 = TimeDistributed(base_model3)(seq_input3)

    #encoded_sequence3=BatchNormalization()(encoded_sequence3)    
    encoded_sequence = keras.layers.Concatenate()([encoded_sequence1, encoded_sequence3])
  
    #encoded_sequence=BatchNormalization()(encoded_sequence)
    #encoded_sequence = Dropout(rate=0.5)(encoded_sequence)
    #encoded_sequence=SeqWeightedAttention(encoded_sequence)
    #lstm_enc, fh, fc, bh, bc= Bidirectional(LSTM(120, return_sequences=True, return_state=True),
     #                                     name='bidirectional_enc')(encoded_sequence)
    # load forward LSTM with reverse states following Liu, Lane 2016 (and do reverse)
   
    #lstm_dec = Bidirectional(LSTM(120, return_sequences=True),
     #                    name='bidirectional_dec')(lstm_enc, initial_state=[bh, bc, fh, fc])
    

    #lyr_crf   = CRF(120, sparse_target=True, learn_mode='marginal', test_mode='marginal')
    #out_slot  = lyr_crf(lstm_dec)
   # combine lstm with CRF for attention (see Liu & Lane)
    #seq_concat = keras.layers.Concatenate()([lstm_dec , out_slot])
    #att_int = AttentionWithContext(name='intent_attention',W_regularizer=keras.regularizers.l2(1e-4),
     #                  b_regularizer=keras.regularizers.l1(1e-4),u_regularizer=keras.regularizers.l1(1e-4))(seq_concat)
    #seq_concat = Dropout(.3, name='bidirectional_dropout_3')(lstm_dec)

# layer: intent attention w/context (Liu & Lane)
   
    #att_int = SeqSelfAttention(attention_type=SeqSelfAttention.ATTENTION_TYPE_MUL,
                       #kernel_regularizer=keras.regularizers.l2(1e-4),
                       #bias_regularizer=keras.regularizers.l1(1e-4),
                       #attention_regularizer_weight=1e-4,
                       #name='Attention')(seq_concat)
    #out_int = Dense(K.int_shape(seq_concat)[-1],
                
     #           name='intent_dense_1',activation="relu")(seq_concat)

    out_int=TCN(return_sequences=True,name='flaten1')(encoded_sequence)
    #lyr_crf   = CRF(64, sparse_target=True, learn_mode='marginal', test_mode='marginal',name='flaten2')
    #out_slot  = lyr_crf(out_int)
    #out_int = keras.layers.Concatenate()([out_int , out_slot])
    #out_int = Dense(K.int_shape(out_int)[-1],
                
     #           name='flaten',activation="selu")(out_int)
    #crf = CRF(nclass, sparse_target=True, learn_mode='marginal')
    #out1 = crf(out_int)
    
    out1 = TimeDistributed(Dense(nclass, activation="softmax"))(out_int)
    #out = Convolution1D(nclass, kernel_size=3, activation="softmax", padding="same", name='layer_4')(encoded_sequence)
    #out = crf(att_int)
    from keras.models import Model

    model = Model([seq_input1,seq_input3], [out1])
    
    #model.compile(loss=crf.loss, optimizer='sgd')

    #model.compile(loss='losses.sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    #model.compile(optimizers.optimizer, crf.loss_function, metrics=[crf.accuracy])
    
    model.compile(optimizers.Adam(0.001,amsgrad=True), sample_weight_mode="temporal",loss='sparse_categorical_crossentropy', metrics=[crf_marginal_accuracy])

    #crf.loss_function
    #model.summary()
    
    return model

def model_without_locallyconected():
    model1 = Sequential()
    model1.add(Convolution1D(64, kernel_size=200, activation=activations.selu,strides=20, padding="valid", name='layer_11',input_shape=(3000, 1)))  
    model1.add(Convolution1D(64, kernel_size=6, activation=activations.selu, padding="valid", name='layer_12'))
    model1.add(MaxPool1D(pool_size=3,strides=3))
    #model1.add(SpatialDropout1D(rate=0.3))
    model1.add(Convolution1D(32, kernel_size=6, activation=activations.selu, padding="valid", name='layer_13'))
    model1.add(Convolution1D(32, kernel_size=6, activation=activations.selu, padding="valid", name='layer_14'))
    model1.add(MaxPool1D(pool_size=2,strides=2))
    model1.add(Convolution1D(32, kernel_size=3, activation=activations.selu, padding="valid", name='layer_15',kernel_regularizer= keras.regularizers.l1_l2(l1=0.0001, l2=0.001)))
    model1.add(Convolution1D(32, kernel_size=3, activation=activations.selu, padding="valid", name='layer_16'))
    model1.add(MaxPool1D(pool_size=2,strides=2))
    #model1.add(BatchNormalization())
    model1.add(SpatialDropout1D(rate=0.01))
    model1.add(Convolution1D(128, kernel_size=3,activation=activations.selu, padding="valid", name='layer_17', kernel_regularizer=keras.regularizers.l1_l2(l1=0.0001, l2=0.0001)))
    #model1.add(Convolution1D(256, kernel_size=3, activation=activations.relu, padding="valid", name='layer_18', kernel_regularizer=regularizers.l2(0.0001)))
    #model1.add(GlobalMaxPool1D())
    convout2 = GlobalMaxPool1D()
    model1.add(convout2)
    model1.add(Dropout(rate=0.01))

    model1.compile(optimizers.Adam(0.001), losses.sparse_categorical_crossentropy, metrics=['acc'])

    model3 = Sequential()
    model3.add(Convolution1D(64, kernel_size=25,strides=3, activation=activations.selu, padding="valid", name='layer_31', input_shape=(3000, 1)))  
    model3.add(Convolution1D(64, kernel_size=8, activation=activations.selu, padding="valid", name='layer_32'))


    #model3.add(BatchNormalization())
    model3.add(MaxPool1D(pool_size=4,strides=4))
    #model3.add(SpatialDropout1D(rate=0.5))
    model3.add(Convolution1D(32, kernel_size=8, activation=activations.selu, padding="valid", name='layer_33'))
    model3.add(Convolution1D(32, kernel_size=8, activation=activations.selu, padding="valid", name='layer_34'))
    model3.add(MaxPool1D(pool_size=2,strides=2))
    model3.add(Convolution1D(32, kernel_size=5, activation=activations.selu, padding="valid", name='layer_25',kernel_regularizer= keras.regularizers.l1_l2(l1=0.0001, l2=0.001)))
    model3.add(Convolution1D(32, kernel_size=5, activation=activations.selu, padding="valid", name='layer_26'))
    model3.add(MaxPool1D(pool_size=2,strides=2))
    
    
    model3.add(SpatialDropout1D(rate=0.01))
    model3.add(Convolution1D(128, kernel_size=3, activation=activations.selu, padding="valid", name='layer_37',kernel_regularizer= keras.regularizers.l1_l2(l1=0.0001, l2=0.0001)))
    model3.add(GlobalMaxPool1D())
    model3.add(Dropout(rate=0.01))
    #model3.add((Dense(64, activation=activations.relu, name='layer_39')))
    model3.compile(optimizers.Adam(0.001), losses.sparse_categorical_crossentropy, metrics=['acc'])


    
    
    nclass = 5

    seq_input1 = Input(shape=(None, 3000, 1))
    
    seq_input3 = Input(shape=(None, 3000, 1))

    base_model1 = model1
 
    base_model3 = model3

    #for layer in base_model1.layers:
     # layer.trainable = False
    #This wrapper applies a layer to every temporal slice of an input.
    #for layer in base_model2.layers:
     #    layer.trainable = False
        
    #for layer in base_model3.layers:
     #    layer.trainable = False
    encoded_sequence1 = TimeDistributed(base_model1)(seq_input1)
    encoded_sequence3 = TimeDistributed(base_model3)(seq_input3)

    #encoded_sequence3=BatchNormalization()(encoded_sequence3)
    
    
    encoded_sequence = keras.layers.Concatenate()([encoded_sequence1, encoded_sequence3])
  
    #encoded_sequence=BatchNormalization()(encoded_sequence)
  
  
    
    #encoded_sequence = Dropout(rate=0.5)(encoded_sequence)
    #encoded_sequence=SeqWeightedAttention(encoded_sequence)
    lstm_enc, fh, fc, bh, bc= Bidirectional(LSTM(120, return_sequences=True, return_state=True),
                                          name='bidirectional_enc')(encoded_sequence)
    # load forward LSTM with reverse states following Liu, Lane 2016 (and do reverse)
   
    lstm_dec = Bidirectional(LSTM(120, return_sequences=True),
                         name='bidirectional_dec')(lstm_enc, initial_state=[bh, bc, fh, fc])
    

    lyr_crf   = CRF(120, sparse_target=True, learn_mode='marginal', test_mode='marginal')
    out_slot  = lyr_crf(lstm_dec)
    
   # combine lstm with CRF for attention (see Liu & Lane)
    seq_concat = keras.layers.Concatenate()([lstm_dec , out_slot])
    att_int = AttentionWithContext(name='intent_attention',W_regularizer=keras.regularizers.l2(1e-4),
                       b_regularizer=keras.regularizers.l1(1e-4),u_regularizer=keras.regularizers.l1(1e-4))(seq_concat)
    #seq_concat = Dropout(.3, name='bidirectional_dropout_3')(lstm_dec)

# layer: intent attention w/context (Liu & Lane)
   
    #att_int = SeqSelfAttention(attention_type=SeqSelfAttention.ATTENTION_TYPE_MUL,
                       #kernel_regularizer=keras.regularizers.l2(1e-4),
                       #bias_regularizer=keras.regularizers.l1(1e-4),
                       #attention_regularizer_weight=1e-4,
                       #name='Attention')(seq_concat)
    out_int = Dense(K.int_shape(seq_concat)[-1],
                
                name='intent_dense_1',activation="relu")(seq_concat)

    out_int=TCN(return_sequences=True,name='flaten1')(encoded_sequence)
    lyr_crf   = CRF(64, sparse_target=True, learn_mode='marginal', test_mode='marginal',name='flaten2')
    out_slot  = lyr_crf(out_int)
    out_int = keras.layers.Concatenate()([out_int , out_slot])
    out_int = Dense(K.int_shape(out_int)[-1],
                
                name='flaten',activation="selu")(out_int)
    crf = CRF(nclass, sparse_target=True, learn_mode='marginal')
    out1 = crf(out_int)
    
    #out = TimeDistributed(Dense(nclass, activation="softmax"))(att_int)
    #out = Convolution1D(nclass, kernel_size=3, activation="softmax", padding="same", name='layer_4')(encoded_sequence)
    

    #out = crf(att_int)


    from keras.models import Model

    model = Model([seq_input1,seq_input3], [out1])
    
    #model.compile(loss=crf.loss, optimizer='sgd')

    #model.compile(loss='losses.sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    #model.compile(optimizers.optimizer, crf.loss_function, metrics=[crf.accuracy])
    
    model.compile(optimizers.Adam(0.001,amsgrad=True),loss='sparse_categorical_crossentropy', metrics=[crf_marginal_accuracy])
    #crf.loss_function
    #model.summary()
    
    return model
# lstm model

def lstm_model():
    model1 = Sequential()
    model1.add(Convolution1D(64, kernel_size=200, activation=activations.selu,strides=20, padding="valid", name='layer_11',input_shape=(3000, 1)))  
    model1.add(Convolution1D(64, kernel_size=6, activation=activations.selu, padding="valid", name='layer_12'))
    model1.add(MaxPool1D(pool_size=3,strides=3))
    #model1.add(SpatialDropout1D(rate=0.3))
    model1.add(Convolution1D(32, kernel_size=6, activation=activations.selu, padding="valid", name='layer_13'))
    model1.add(Convolution1D(32, kernel_size=6, activation=activations.selu, padding="valid", name='layer_14'))
    model1.add(MaxPool1D(pool_size=2,strides=2))
    model1.add(Convolution1D(32, kernel_size=3, activation=activations.selu, padding="valid", name='layer_15',kernel_regularizer= keras.regularizers.l1_l2(l1=0.0001, l2=0.001)))
    model1.add(Convolution1D(32, kernel_size=3, activation=activations.selu, padding="valid", name='layer_16'))
    model1.add(MaxPool1D(pool_size=2,strides=2))
    #model1.add(BatchNormalization())
    model1.add(SpatialDropout1D(rate=0.01))
    model1.add(LocallyConnected1D(128, kernel_size=3,activation=activations.selu, padding="valid", name='layer_17', kernel_regularizer=keras.regularizers.l1_l2(l1=0.0001, l2=0.0001)))
    #model1.add(Convolution1D(256, kernel_size=3, activation=activations.relu, padding="valid", name='layer_18', kernel_regularizer=regularizers.l2(0.0001)))
    #model1.add(GlobalMaxPool1D())
    convout2 = GlobalMaxPool1D()
    model1.add(convout2)
    model1.add(Dropout(rate=0.01))
    #model1.add((Dense(64, activation=activations.relu, name='layer_19')))
    #model1.compile(loss='categorical_crossentropy', optimizer=optimizer1, metrics=['accuracy'])
    #model1.compile(loss='sparse_categorical_crossentropy', optimizer=adam_with_lr_multipliers1, metrics=['accuracy'])

    model1.compile(optimizers.Adam(0.001), losses.sparse_categorical_crossentropy, metrics=['acc'])
    #for layer in model1.layers[:-5]:
     #    layer.trainable = False
    
   
    
    #model.summary()
    model3 = Sequential()
    model3.add(Convolution1D(64, kernel_size=25,strides=3, activation=activations.selu, padding="valid", name='layer_31', input_shape=(3000, 1)))  
    model3.add(Convolution1D(64, kernel_size=8, activation=activations.selu, padding="valid", name='layer_32'))


    #model3.add(BatchNormalization())
    model3.add(MaxPool1D(pool_size=4,strides=4))
    #model3.add(SpatialDropout1D(rate=0.5))
    model3.add(Convolution1D(32, kernel_size=8, activation=activations.selu, padding="valid", name='layer_33'))
    model3.add(Convolution1D(32, kernel_size=8, activation=activations.selu, padding="valid", name='layer_34'))
    model3.add(MaxPool1D(pool_size=2,strides=2))
    model3.add(Convolution1D(32, kernel_size=5, activation=activations.selu, padding="valid", name='layer_25',kernel_regularizer= keras.regularizers.l1_l2(l1=0.0001, l2=0.001)))
    model3.add(Convolution1D(32, kernel_size=5, activation=activations.selu, padding="valid", name='layer_26'))
    model3.add(MaxPool1D(pool_size=2,strides=2))
    
    model3.add(SpatialDropout1D(rate=0.01))
    model3.add(LocallyConnected1D(128, kernel_size=3, activation=activations.selu, padding="valid", name='layer_37',kernel_regularizer= keras.regularizers.l1_l2(l1=0.0001, l2=0.0001)))
    model3.add(GlobalMaxPool1D())
    model3.add(Dropout(rate=0.01))
    #model3.add((Dense(64, activation=activations.relu, name='layer_39')))
    model3.compile(optimizers.Adam(0.001), losses.sparse_categorical_crossentropy, metrics=['acc'])

    nclass = 5

    seq_input1 = Input(shape=(None, 3000, 1))
    
    seq_input3 = Input(shape=(None, 3000, 1))

    base_model1 = model1
 
    base_model3 = model3

    #for layer in base_model1.layers:
     # layer.trainable = False
    #This wrapper applies a layer to every temporal slice of an input.
    #for layer in base_model2.layers:
     #    layer.trainable = False
        
    #for layer in base_model3.layers:
     #    layer.trainable = False
    encoded_sequence1 = TimeDistributed(base_model1)(seq_input1)
    encoded_sequence3 = TimeDistributed(base_model3)(seq_input3)

    #encoded_sequence3=BatchNormalization()(encoded_sequence3)
    
    
    encoded_sequence = keras.layers.Concatenate()([encoded_sequence1, encoded_sequence3])
  
    #encoded_sequence=BatchNormalization()(encoded_sequence)
  
  
    
    #encoded_sequence = Dropout(rate=0.5)(encoded_sequence)
    #encoded_sequence=SeqWeightedAttention(encoded_sequence)
    lstm_enc, fh, fc, bh, bc= Bidirectional(LSTM(120, return_sequences=True, return_state=True),
                                          name='bidirectional_enc')(encoded_sequence)
    # load forward LSTM with reverse states following Liu, Lane 2016 (and do reverse)
   
    lstm_dec = Bidirectional(LSTM(120, return_sequences=True),
                         name='bidirectional_dec')(lstm_enc, initial_state=[bh, bc, fh, fc])
    

    lyr_crf   = CRF(120, sparse_target=True, learn_mode='marginal', test_mode='marginal')
    out_slot  = lyr_crf(lstm_dec)
    
   # combine lstm with CRF for attention (see Liu & Lane)
    seq_concat = keras.layers.Concatenate()([lstm_dec , out_slot])
    #att_int = AttentionWithContext(name='intent_attention',W_regularizer=keras.regularizers.l2(1e-4),
     #                  b_regularizer=keras.regularizers.l1(1e-4),u_regularizer=keras.regularizers.l1(1e-4))(seq_concat)
    #seq_concat = Dropout(.3, name='bidirectional_dropout_3')(lstm_dec)

# layer: intent attention w/context (Liu & Lane)
   
    #att_int = SeqSelfAttention(attention_type=SeqSelfAttention.ATTENTION_TYPE_MUL,
                       #kernel_regularizer=keras.regularizers.l2(1e-4),
                       #bias_regularizer=keras.regularizers.l1(1e-4),
                       #attention_regularizer_weight=1e-4,
                       #name='Attention')(seq_concat)
    out_int = Dense(K.int_shape(seq_concat)[-1],
                
                name='intent_dense_1',activation="relu")(seq_concat)

    out_int = Bidirectional(LSTM(120, return_sequences=True),
                         name='bidirectional_dec1')(encoded_sequence)
    out_int = Bidirectional(LSTM(120, return_sequences=True),
                         name='bidirectional_dec2')(out_int)
    #lyr_crf   = CRF(64, sparse_target=True, learn_mode='marginal', test_mode='marginal',name='flaten2')
    #out_slot  = lyr_crf(out_int)
    #out_int = keras.layers.Concatenate()([out_int , out_slot])
    out_int = Dense(K.int_shape(out_int)[-1],
                
                name='flaten',activation="selu")(out_int)
    crf = CRF(nclass, sparse_target=True, learn_mode='marginal')
    #out1 = crf(out_int)
    
    #out = TimeDistributed(Dense(nclass, activation="softmax"))(out_int)
    #out = Convolution1D(nclass, kernel_size=3, activation="softmax", padding="same", name='layer_4')(encoded_sequence)
    

    out = crf(out_int)


    from keras.models import Model

    model = Model([seq_input1,seq_input3], [out])
    
    #model.compile(loss=crf.loss, optimizer='sgd')

    #model.compile(optimizers.optimizer, crf.loss_function, metrics=[crf.accuracy])
    
    model.compile(optimizers.Adam(0.001,amsgrad=True), loss='sparse_categorical_crossentropy', metrics=[crf_marginal_accuracy])
    #model.compile(optimizers.Adam(0.001,amsgrad=True), loss=categorical_focal_loss(alpha=[[3, 1, 3,1.5,1.5]], gamma=3), metrics=[crf_marginal_accuracy])
    #model.compile(optimizers.Adam(0.001,amsgrad=True), loss=KerasFocalLoss, metrics=[crf_marginal_accuracy])

    
    
    return model
