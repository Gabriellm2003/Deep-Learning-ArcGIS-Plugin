# Deep-Learning-ArcGIS-Plugin

# The framework pipeline

This framework was proposed to train and infer CNNs for the tasks of semantic segmentation and object detection using large scale images and annotations made using shapefiles. 

All steps of the framework are implemented to work in ArcGIS via a plugin, and there is also a web service implementation of it in this repository. 

# Plugin instalation

The plugin installation is quite easy. The only step needed is to copy this repository to the folder that ArcGis is installed. Typically it can be found in C:\Users\Usename\Documents\ArcGIS DocumentosnArcGIS).
After that, open ArcGIS software and you'll be able to visualize the plugin on the right bar, as can be seen in the figure below.
![alt text](images/./arcgis_plugin1.png)

# Web Service

There is a web service implementation of the framework located in scr/webservice. To use the server, run the following command:
```diff
python3 server.py <port_number>
```

# Using your own network in the framework

To use your network and integrate it into the framework, you will need to edit some files.
First, for image segmentation, you will need to edit the file segmentationTools.py. The only alteration you need to perform is to change the cmd variable to handle the command used to run your network. For object detection, the only difference is that you have to edit a different file, called detectionTools.py.


The same process can be done for the web service editing the file /src/webservice/server.py with your network commands.
In this implementation, there isn`t an example code for object detection, but for image segmentation there is one, using Mask-R-CNN. 


# Plugin Use

The use of the plugin is quite simple since it has an intuitive interface. The only thing you have to do is to select the desired option and fill all the options with the desired data. The figures bellow illustrates the plugin interface.
![alt text](images/./arcgis_usetrain.png)
![alt text](images/./arcgis_use2.png)
