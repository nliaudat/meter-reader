# meter-reader
meter (water, gas, electricity) reader from image to digits

![alt text](https://github.com/nliaudat/meter-reader/blob/main/result.jpg "digit recognition result")

Final Meter Reading:

176882

## Goals : 

* Make a independant from hardware meter reading solution
* Keep simple

## Process : 

* Get image from any camera (local or http)
* [onetime] set regions to process
* Produce digits from image

## Solutions 1 - python only : 
`pip install -r requirements.txt`

`python draw_regions.py`

  *input image source and draw digits regions from left to right*
  
`python meter_reading.py`


## Solutions 2 - webapp - flask_meter_reader : 
`pip install -r requirements.txt`

`python app.py`

*- http://127.0.0.1:5000
- [onetime]Go to  Draw Regions> input you image (http or local) and set regions
   
- Return to index and process you image*
  

## Sources and inspiration: 
* The idea : https://github.com/jomjol/AI-on-the-edge-device/
* The AI model : https://github.com/haverland/Tenth-of-step-of-a-meter-digit/tree/master/output

## Licence: 
* Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC-BY-NC-SA)
* No commercial use
