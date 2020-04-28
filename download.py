'''
The official training data (use this to train your model):
 - Images and annotations (GTSRB_Final_Training_Images.zip)
 - Three sets of different HOG features (GTSRB_Final_Training_HOG.zip)
 - Haar-like features (GTSRB_Final_Training_Haar.zip)
 - Hue histograms (GTSRB_Final_Training_HueHist.zip)
'''

import os
import requests

# in order
URLS = {
	'Train_Images.zip': 'https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Training_Images.zip',
	'Train_HOG.zip': 'https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Training_HOG.zip',
	'Train_Haar.zip': 'https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Training_Haar.zip',
	'Train_HeuHist.zip': 'https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Training_HueHist.zip'
}

# download data in .zip
def download(path='.'):
	for key, val in URLS.items():
		dest = os.path.join(path, key)

		if not os.path.exists(dest):
			r = requests.get(val, stream=True)
			with open(dest, 'wb') as f:
				for chunk in r.iter_content():
					f.write(chunk)

if __name__ == '__main__':
	download()