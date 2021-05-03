# Self-supervised-learning Final Competition
In this project we had the task of classifiying images of size 96x96x3. There were around 800 classes with a validation set of size 25600 and 25600 labeled images. Apart from that there were 512000 imgaes without labels which were used for the unsupervised learning task

File Description
-----------------
dataloader.py -> Contains the logic for loading images and creating a dataloader
demo.sbatch -> The sbatch file used for submitting job to the HPC (GCP) cluster
eval.py -> Evalutaion script file
label_finder.py -> Code for finding out the additional 12800 labels
lr_schedule.py -> Code for implementing the Cosine decay learning rate schedule
simsiam.py -> code for Training the Simsiamese model for pretect task
simsiam_linear_eval.py -> Code for training and finding out the best validation accuracy for the supervised task
submission.py -> Model architecture for our project
transform.py -> Image augmentations utility file

Running the project
-------------------
Step - 1 : Train the unsupervised model
On Line number 27, replace the path location with the location where the unlabeled images are stored
On Line number 88, replace the path location with the location where the final trained model weights will be stored
Then run python simsiam.py

Step-2 : Train and evaluate the supervised model
Create a folder checkpointfinal96
On Line number 38, provide the model weight location you had stored in the above step
On Line number 39, replace the path location with the location where the labeled images are stored
On Line number 55 and 58, replace the path location with the location where the labeled images and validation images are stored
run python simsiam_linear_eval.py


