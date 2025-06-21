# meter-reader
meter (water, gas, electricity) reader from image to digits

![alt text](https://github.com/nliaudat/meter-reader/blob/main/result.jpg? "digit recognition result")


## Goals : 

* Read meter (water, gas, electricity)
* Hardware independant (works with any camera that can produce a fixed image)
* Keep simple

## Process : 

* Get image from any camera (local or http)
* Setup a python compatible apps to process the image (docker, flask or other python compatible apps)
* [onetime] set regions to process
* Produce digits from image
  

## Solutions 1 - python only : (tested on Python 3.12.3)
`pip install -r requirements.txt`

`python draw_regions.py`

  *input image source and draw digits regions from left to right*
  
`python meter_reading.py`


`optional arguments : --image_source http://192.168.1.113/img_tmp/alg.jpg --model model.tflite --regions regions.txt --no-gui --no-output-image`

For testing all models : 
`--test-all-models --expected_result 177663`

## Solutions 2 - docker/webapp - flask_meter_reader :
`pip install -r requirements.txt`

`python app.py`

*- http://127.0.0.1:5000
- [onetime]Go to  Draw Regions> input you image (http or local) and set regions
   
- Return to index and process you image
(api is available)
  
## Todos: 
* make a hassio component (not actually possible cause nor tflite-runtime nor tensorflow supports python 3.13). Perhaps can hassio use venv ? The actual solution use the docker image
* make a scene text recognition model to recognise all digits in one shot. (actually investigating paddleOcr, OpenOcr, Parsec)
  

## Sources and inspiration: 
* The idea : https://github.com/jomjol/AI-on-the-edge-device/
* The AI model : https://github.com/haverland/Tenth-of-step-of-a-meter-digit/tree/master/output

## Licence: 
* Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC-BY-NC-SA)
* No commercial use
* The AI model from haverland is under Apache Licence
