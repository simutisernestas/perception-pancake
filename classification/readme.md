SGDnoPCA uses SGD classifier without PCA. Grayscale or not doesn't seem to affect the performance.

taker_example is a file that shows how to call the classifier

Torch folder contains a classifier with torch

pic folder contains train and validation pictures, alongside four validation pictures used for testing. It also contains some files usable to prepare the data (rename, resize and grayscale). Not used anymore as the modifications are no longer required (no renaming, grayscale and resize made within the classifier code)

noPCA is a test folder

backup is a...backup folder