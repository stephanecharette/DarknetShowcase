#!/bin/bash

# get the MSCOCO pre-trained weights from AlexeyAB's github repo


if [ ! -f coco.names ]
then
	wget https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/coco.names
fi

if [ ! -f yolov4-tiny.cfg ]
then
	wget https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg
fi

if [ ! -f yolov4-tiny.weights ]
then
	wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights
fi

if [ ! -f yolov4.cfg ]
then
	wget https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4.cfg
fi

if [ ! -f yolov4.weights ]
then
	wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4.weights
fi

