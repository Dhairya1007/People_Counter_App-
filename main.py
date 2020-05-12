"""People Counter."""

import os
import sys
import time
import socket
import json
import cv2
import numpy as np

import logging as log
import paho.mqtt.client as mqtt

from argparse import ArgumentParser
from inference import Network

# MQTT server environment variables
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60

classes = ["bicycle", "person", "car", "motorcycle", "airplane", "bus",
    "train", "truck", "boat", "traffic light", "fire hydrant", 
   "stop sign", "parking meter", "bench", "bird", "cat", "dog", 
   "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", 
   "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", 
   "skis", "snowboard", "sports ball", "kite", "baseball bat", 
   "baseball glove", "skateboard", "surfboard", "tennis racket", 
   "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", 
   "banana", "apple", "sandwich", "orange", "broccoli", "carrot", 
   "hot dog", "pizza", "donut", "cake", "chair", "couch", 
   "potted plant", "bed", "dining table", "toilet", "tv", "laptop", 
   "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", 
   "toaster", "sink", "refrigerator", "book", "clock", "vase", 
   "scissors", "teddy bear", "hair drier", "toothbrush"]


def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    c_desc = "The color of the bounding boxes to draw; RED, GREEN or BLUE"
    ct_desc = "The confidence threshold to use with the bounding boxes"
    
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=True, type=str,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")
    parser.add_argument("-c", help=c_desc, default='BLUE')
    parser.add_argument("-ct", help=ct_desc, default=0.05)
    
    return parser


def connect_mqtt():
    ### TODO: Connect to the MQTT client ###
    client = mqtt.Client()
    client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)
    
    return client

def convert_color(color_string):
    '''
    Get the BGR value of the desired bounding box color.
    Defaults to Blue if an invalid color is given.
    '''
    colors = {"BLUE": (255,0,0), "GREEN": (0,255,0), "RED": (0,0,255)}
    out_color = colors.get(color_string)
    if out_color:
        return out_color
    else:
        return colors['BLUE']

def draw_boxes(frame, result, args, width, height, t_count, inf_time):
    '''
    Draw bounding boxes onto the frame.
    '''
    c_count = 0
    inf_time_message = "Inference time: {:.3f} ms".format(inf_time * 1000)
    for box in result[0][0]:
        conf = box[2]
        if conf >= args.ct:
            c_count = c_count+1
            
            xmin = int(box[3] * width)
            ymin = int(box[4] * height)
            xmax = int(box[5] * width)
            ymax = int(box[6] * height)
            class_id = int(box[1])
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), args.c, 1)
            det_label = classes[class_id] if classes else str(class_id)
            cv2.putText(frame, det_label + ' ' + str(round(box[2] * 100, 1)) + ' %', (xmin, ymin - 7),cv2.FONT_HERSHEY_COMPLEX, 0.5, (255,0,0), 1)
            cv2.putText(frame, inf_time_message, (15, 15), cv2.FONT_HERSHEY_COMPLEX, 0.5, (200, 10, 10), 1)

    return frame, c_count

def get_class_names(class_nums):
    class_names= []
    for i in class_nums:
        class_names.append(CLASSES[int(i)])
    return class_names

def infer_on_stream(args, client):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.

    :param args: Command line arguments parsed by `build_argparser()`
    :param client: MQTT client
    :return: None
    """
    current_count = 0
    total_count = 0
    last_count = 0
    duration = 0

    args.c = convert_color(args.c)
    args.ct = float(args.ct)
    
    # Initialise the class
    infer_network = Network()
    # Set Probability threshold for detections
    prob_threshold = args.prob_threshold

    ### TODO: Load the model through `infer_network` ###
    infer_network.load_model(args.model, args.device, args.cpu_extension)
    net_ip_shape = infer_network.get_input_shape()
    ### TODO: Handle the input stream ###
    cap = cv2.VideoCapture(args.input)
    cap.open(args.input)
    width = int(cap.get(3))
    height = int(cap.get(4))
    
    ### TODO: Loop until stream is over ###
    while cap.isOpened():
        # Read the next frame
        flag, frame = cap.read()
        if not flag:
            break
        key_pressed = cv2.waitKey(60)

        ### TODO: Pre-process the image as needed ###
        p_frame = cv2.resize(frame, (net_ip_shape[3], net_ip_shape[2]))
        p_frame = p_frame.transpose((2,0,1))
        p_frame = p_frame.reshape(1, *p_frame.shape)
        
        ### TODO: Start asynchronous inference for specified request ###
        start = time.time()
        infer_network.exec_net(p_frame)
        ### TODO: Wait for the result ###
        if infer_network.wait() == 0:
            
            inf_time = time.time() - start
            result = infer_network.get_output()
            #print(result)
                
            # Draw the output mask onto the input
            frame, current_count = draw_boxes(frame, result, args, width, height, total_count, inf_time)
            
            # When new person enters the video
            if current_count > last_count:
                 #take start time
                # increment total_count with current_count and decrement last_count.
                start_time = time.time()
                total_count = total_count + current_count
                last_count = last_count - 1
           
            #publish total_count
            client.publish("person", json.dumps({"total": total_count}))
            
            # Person duration in the video is calculated
            if current_count < last_count:
                # calculate duration by current_time - start_time
                # Publish messages to the MQTT server
                
                duration = int(time.time() - start_time)
                
                if duration > 2:
                    client.publish("person/duration", json.dumps({"duration": duration}))
            
            client.publish("person", json.dumps({
                "count": current_count
            }))
            
            last_count = current_count
            
             
        ### TODO: Send the frame to the FFMPEG server ###
        sys.stdout.buffer.write(frame)
        sys.stdout.flush()

        ### TODO: Write an output image if `single_image_mode` ###
       
    
def main():
    """
    Load the network and parse the output.

    :return: None
    """
    # Grab command line args
    args = build_argparser().parse_args()
    # Connect to the MQTT server
    client = connect_mqtt()
    # Perform inference on the input stream
    infer_on_stream(args, client)


if __name__ == '__main__':
    main()
