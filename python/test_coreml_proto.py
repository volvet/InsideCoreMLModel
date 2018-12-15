#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  8 12:48:34 2018

@author: volvetzhang
"""

import Model_pb2


def parseWeightParam(weightParam):
    print('\tweight param')
    pass

def parseConvolution(convolution):
    print('\tkernal size: ', convolution.kernelSize)
    print('\tinput channels: ', convolution.kernelChannels)
    print('\toutput channels: ', convolution.outputChannels)
    print('\tkernel channles: ', convolution.kernelChannels)
    print('\tgroups: ', convolution.nGroups)
    print('\tstride: ', convolution.stride)
    print('\tdilation factor: ', convolution.dilationFactor)
    if convolution.HasField('valid'):
        print('\tvalid: ', convolution.valid)
    if convolution.HasField('same'):
        print('\tsame: ', convolution.same)
    print('\tis deconvolution: ', convolution.isDeconvolution)
    print('\thas bias: ', convolution.hasBias)
    if convolution.hasBias:
        parseWeightParam(convolution.bias)
    parseWeightParam(convolution.weights)
    print('\toutput shape: ', convolution.outputShape)

    pass

def parsePooling(pooling):
    print('\ttype: ', pooling.type)
    print('\tkernel size: ', pooling.kernelSize)
    print('\tstride: ', pooling.stride)
    if pooling.HasField('valid'):
        print('\tvalid: ', pooling.valid)
    if pooling.HasField('same'):
        print('\tsame: ', pooling.same)
    if pooling.HasField('includeLastPixel'):
        print('\tincludeLastPixel: ', pooling.includeLastPixel)
    print('\tavgPoolExcludePadding: ', pooling.avgPoolExcludePadding)
    print('\tglobalPooling: ', pooling.globalPooling)
    pass


# f(x) = max(0, x)
def parseReLU(ReLU):
    print('\trelu')
    pass

def parseActivation(activation):
    if activation.HasField('ReLU'):
        parseReLU(activation.ReLU)
    pass

def parseLRNLayer(lrn):
    print('\talpha: ', lrn.alpha)
    print('\tbeta: ', lrn.beta)
    print('\tlocal size: ', lrn.localSize)
    print('\tk: ', lrn.k)
    pass

def parseFlattenLayer(flatten):
    print('\tmode: ', flatten.mode)
    pass

def parseInnerProductLayer(innerProduct):
    print('\tinput channels: ', innerProduct.inputChannels)
    print('\toutput channels: ', innerProduct.outputChannels)
    print('\thas bias: ', innerProduct.hasBias)
    if innerProduct.hasBias:
        parseWeightParam(innerProduct.bias)
    parseWeightParam(innerProduct.weights)
    pass

def parseSoftmax(softmax):
    print('\tsoftmax')
    pass

def parseNeuralNetworkClassifier(neuralNetworkClassifier):
    for preprocess in neuralNetworkClassifier.preprocessing:
        print(preprocess.featureName)
        if preprocess.HasField('scaler'):
            print (preprocess.scaler)
        if preprocess.HasField('meanImage'):
            print (preprocess.meanImage)
    for layer in neuralNetworkClassifier.layers:
        print(layer.name, layer.input, layer.output)
        if layer.HasField('convolution'):
            parseConvolution(layer.convolution)
        elif layer.HasField('pooling'):
            parsePooling(layer.pooling)
        elif layer.HasField('activation'):
            parseActivation(layer.activation)
        elif layer.HasField('lrn'):
            parseLRNLayer(layer.lrn)
        elif layer.HasField('flatten'):
            parseFlattenLayer(layer.flatten)
        elif layer.HasField('innerProduct'):
            parseInnerProductLayer(layer.innerProduct)
        elif layer.HasField('softmax'):
            parseSoftmax(layer.softmax)
    pass

if __name__ == '__main__':
    model = Model_pb2.Model()
    with open('AgeNet.mlmodel', 'rb') as f:
        model.ParseFromString(f.read())
    
    print(model.description)
    print(model.specificationVersion)
    if model.HasField('neuralNetworkClassifier'):
        parseNeuralNetworkClassifier(model.neuralNetworkClassifier)
    
    
    
    