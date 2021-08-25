# MURA-abnormality-detection
## Detection of abnormalities on X-ray scans.

This project is a good example of application of AI in Medical Science. Being a doctor it was a fascinating project to work on. We regularly diagnose abnormalities in bone X-ray 
but as doctors we acquire such skills after rigrous studies for a minimum of 5 years. Artifical Intelligence with its detection and recognition capabilities is a game changer 
and with cutting edge algorithms and increasing data availibility, it is competing with acumen of a doctor in several medical imaging problems.

Dataset:
MURA (musculoskeletal radiographs) is a large dataset of bone X-rays. It consists many folder of various xray of different bony parts such as eLbow, wrist, humerus, forearm etc.
Each folder contains patient entries with negative or positive in the name denoting absence or presence of abnormalities respectivies. Each patient folder has X-ray image with 
different views.

## Repository:

     main.py - Runs the model for training. Inludes making dataloader object, simple loss function, using ReduceLRonPLeatue to lower the 
     learning rate in later phase of training, and saving the model. We use Densenet 169 for training and prediction purposes.

     pipeline.py - Includes  :
                   1. Function to fetch data from each patient entry folder along with label in its name.
                   2. Dataset function to create a dataset - It collects image and labels from pandas dataframe created in the first step.
                   3. Dataloader function with transforms mainly resize, random horizontal flip and random rotation.

     train.py -    Includes :
                   1. Training loop 
                   2. Metric function to calculate accuracy - we collect only those prediction with threshold > 0.5
              
     utils.py -    Includes plotting function 

     visualize.py(refer demo folder) - Mainly includes plotting function to output test images with their prediction and truth values.

     appstream.py -  Streamlit implementation of the project. Here we create an app to allow users to upload X-ray scans following
     which the model detects whether there is any abnormality in the scan. We use pillow to load the given image and various streamlit 
     functions to upload image, print predictions etc.




 ![ Alt text](https://github.com/parth-mango/MURA-abnormality-detection/blob/main/demo/mura.gif) 


      Ending Notes:
      This is a very simple implementation of the problem. I personally found it a nice usecase and it can be further improved and implemented
      on a larger scale in peripheries where specialists are not available to diagnose such cases. 
      
      For further discussions on Medical AI and its applications, you can reach out to me @ parth15237@gmail.com
