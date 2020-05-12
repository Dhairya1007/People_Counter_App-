#!/usr/bin/env python3
"""Inference Pyhton file used in main.py"""
import os
import sys
import logging as log
from openvino.inference_engine import IENetwork, IECore
import numpy as np


class Network:
    """
    Load and configure inference plugins for the specified target devices 
    and performs synchronous and asynchronous modes for the specified infer requests.
    """

    def __init__(self):
        ### TODO: Initialize any class variables desired ###
        self.plugin = None
        self.network = None
        self.input_blob = None
        self.output_blob = None
        self. exec_network = None
        self.infer_request = None
        

    def load_model(self, model, device="CPU", cpu_extension=None):
        
        model_xml = model
        model_bin = os.path.splitext(model_xml)[0] + ".bin"
        
        self.plugin = IECore()
        
        if cpu_extension and "CPU" in device:
            self.plugin.add_extension(cpu_extension, device)
        
        self.network = IENetwork(model=model_xml, weights=model_bin)       
        self.exec_network = self.plugin.load_network(self.network, device)
        self.input_blob = next(iter(self.network.inputs))
        self.output_blob = next(iter(self.network.outputs))
        return

    def get_input_shape(self):
        
        return self.network.inputs[self.input_blob].shape

    def exec_net(self, image):
        
        self.exec_network.start_async(request_id=0, 
            inputs={self.input_blob: image})
        
        return

    def wait(self):
        status = self.exec_network.requests[0].wait(-1)
        return status

    def get_output(self):
        
        return self.exec_network.requests[0].outputs[self.output_blob]
