RFWhole: Radial Filtration as topological filter method, implemented to entire image.

Create three folders named: excels, graphs, weights.

Then change data folders in the code.

I did not delete unused images; instead, I created chosen_data files and used those files while data loading.

I cut images on the fly by using the numbers in the json files. You may want to create your json files or delete the part of the code that uses these files.

json files like: {"file_name.dcm": [131, 387, 62, 446], ...}

You'll need to create diagrams and other things depending on gold images in advance and use them in training. CreateHelperData.ipynb contains everything you'll need.
