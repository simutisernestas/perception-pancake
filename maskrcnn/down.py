# import necessary libraries
import glob
import warnings
import cv2
import numpy as np
from PIL import Image
warnings.filterwarnings('ignore')

def main():
    # images from video without occlusions!
    rims = glob.glob("stereo/*Right*")
    rims.sort()
    lims = glob.glob("stereo/*Left*")
    lims.sort()
    count = len(rims)
    for i in range(count):
        imfile = rims[i]
        imgorg = Image.open(imfile)
        width, height = imgorg.size
        imgorg = imgorg.resize((int(width/10), int(height/10)))
        # Setting the points for cropped image
        image = cv2.cvtColor(np.array(imgorg), cv2.COLOR_RGB2BGR)
        cv2.imwrite(f"data/right-{i}.png", image)

if __name__ == '__main__':
    main()
    cv2.destroyAllWindows()
