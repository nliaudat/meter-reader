copy in you homeassistant/custom_components folder

make sur the rights is 755 (read and execute)

reboot hassio

add integration "meter_collector"


config : 

-instance : name of your sensor

-value_url : http://{IP]/value?all=true&type=raw

-image_url : http://{IP}/img_tmp/alg.jpg

-scan_interval : (in seconds)
