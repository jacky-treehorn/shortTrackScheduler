# -*- coding: utf-8 -*-
"""
Created on Sat Aug  6 20:51:53 2022

@author: rasta
"""
from tensorflow import keras, GradientTape, convert_to_tensor, float32 as tffloat32
from tensorflow.keras import layers
from tensorflow.experimental import numpy as tfnp
import numpy as np
from random import shuffle
from copy import copy
import keras.backend as K


def dataCreator(sampleInput: np.array, func, n_samples: int = 1000, funcArgs: dict = {}) -> tuple:
    inpVec = copy(sampleInput)
    samplesInput = []
    samplesTarget = []
    for n_sample in range(n_samples):
        shuffle(inpVec)
        target = func(inpVec, **funcArgs)
        _temp = copy(inpVec)
        samplesInput.append(_temp)
        samplesTarget.append(target)
    return samplesInput, samplesTarget


def kerasGradientsHessianOutputInput(model, sample):
    x_tensor = convert_to_tensor(sample, dtype=tffloat32)
    with GradientTape() as t:
        t.watch(x_tensor)
        with GradientTape() as tt:
            tt.watch(x_tensor)
            output = model(x_tensor)
        jac = tt.gradient(output, x_tensor)
    hess = t.jacobian(jac, x_tensor)
    return jac, hess


def kerasMLP_withPoolingForSkateScheduler(n_inputs: int,
                                          n_outputs: int,
                                          pool_size: int = 4,
                                          mlp_nodes: list = [50, 50],
                                          model_name: str = 'kerasMLP_withPoolingForSkateScheduler',
                                          dropoutFrac: float = 0.333,
                                          activation: str = 'elu',
                                          initOutputBias=None):
    inputLayer = keras.Input(shape=(n_inputs,), name='inputLayer')
    reshapeInputLayer = layers.Reshape(
        (n_inputs, 1), name='reshapeForPooling')(inputLayer)
    maxPoolLayer = layers.MaxPooling1D(
        pool_size=pool_size, name='maxPoolLayer')(reshapeInputLayer)
    valueInverterLayerPreMinPooling = layers.Lambda(
        lambda x: -x, name='valueInverterLayerPreMinPooling')(reshapeInputLayer)
    minPoolLayer = layers.MaxPooling1D(
        pool_size=pool_size, name='minPoolLayer')(valueInverterLayerPreMinPooling)
    valueInverterLayerPostMinPooling = layers.Lambda(
        lambda x: -x, name='valueInverterLayerPostMinPooling')(minPoolLayer)
    heavisideLayer = layers.Lambda(lambda x: tfnp.heaviside(
        x, 1.0)*tfnp.heaviside(-(x - n_inputs), 0.0), name='heavisideLayer')(reshapeInputLayer)
    # heavisideLayer = layers.Lambda(lambda x: 0.25*(1.0+tfnp.tanh(
    #    50.0*x))*(1.0+tfnp.tanh(-50.0*(x - n_inputs))), name='heavisideLayer')(reshapeInputLayer)
    avePoolOnHeaviside = layers.AveragePooling1D(
        pool_size=pool_size, name='avePoolOnHeaviside')(heavisideLayer)
    countingPoolLayer = layers.Lambda(
        lambda x: x*pool_size, name='countingPoolLayer')(avePoolOnHeaviside)
    concatPoolLayer = layers.Concatenate(name='concatPoolLayer')(
        [maxPoolLayer, valueInverterLayerPostMinPooling, countingPoolLayer])

    prevLayer = concatPoolLayer
    for i, mlp_n in enumerate(mlp_nodes):
        _tempLayer = layers.Dense(
            mlp_n, name='mlp'+str(i), activation=activation)(prevLayer)
        _dropoutLayer = layers.Dropout(
            dropoutFrac, name='mlpDropout'+str(i))(_tempLayer)
        prevLayer = _dropoutLayer

    preOutputLayer = layers.Dense(
        n_outputs, name='preOutputLayer')(prevLayer)
    newSize = 1
    for extent in preOutputLayer.shape:
        if extent is None:
            continue
        newSize *= extent
    preOutputReshapeLayer = layers.Reshape(
        (newSize,), name='preOutputReshapeLayer')(preOutputLayer)
    outputLayer = layers.Dense(
        n_outputs, name='outputLayer', activation=activation)(preOutputReshapeLayer)

    model = keras.Model(inputs=inputLayer,
                        outputs=outputLayer, name=model_name)
    if initOutputBias is not None:
        K.set_value(model.layers[-1].weights[1], initOutputBias)
    return model


if __name__ == '__main__':
    inpVec = np.arange(12).astype(float)
    model = kerasMLP_withPoolingForSkateScheduler(inpVec.size, 1)
    y = model(inpVec.reshape(1, inpVec.size, 1))
    print(y)
    print(model.summary())
    jac, hess = kerasGradientsHessianOutputInput(
        model, inpVec.reshape(1, inpVec.size, 1))
    print(jac)
