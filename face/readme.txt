1. Please first download Equalizedface.tar.gz and test.tar.gz from website http://www.whdeng.cn/RFW/index.html and put the files in this folder

2. Run the commands
tar zxvf Equalizedface.tar.gz
python preprocess_equal.py
# resize, transpose, and split images in equalizedface dataset into train, val set, and store images in h5 file

# Note that the images in folder Caucasian and Africa, which are used in our experiment, have already been aligned by the owner of the Equalizedface dataset. If you would like to use the images in Asian or Indian folder, you would need to run the code in mtcnn folder to detect facial landmarks and perform similarity transformation (None of the images in Asian folder has been aligned, and 53651 images in Indian folder has been aligned, you can use file verify.py to see the statistics).


tar zxvf test.tar.gz
python preprocess_rfw.py
# process the image pairs stated in all races' pairs.txt file, which are used to do testing. Images are resized, flipped, transposed, and stored in h5 file
# Note that all images have been aligned by the owner of the RFW dataset.


3. Run experiments
# use the .sh files to reproduce each of the experiment result