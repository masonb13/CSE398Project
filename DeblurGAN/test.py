import time
import os
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer
from pdb import set_trace as st
from util import html
from util.metrics import PSNR
#from ssim import SSIM
from PIL import Image
import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim
import torch
torch.cuda.empty_cache()

opt = TestOptions().parse()
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip
opt.how_many=11  #----------------------
opt.resize_or_crop = 0

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
model = create_model(opt)
#visualizer = Visualizer(opt)
# create website
web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))
# test
avgPSNR_real = []
avgPSNR_pred = []
avgSSIM_real = []
avgSSIM_pred = []
counter = 0

for i, data in enumerate(dataset):
	if (i<opt.how_many):
		counter = i
		print("-------------------------------------\n-\n-\n-\n",i)
		model.set_input(data)
		model.test()
		visuals = model.get_current_visuals()
  
		real_B_path = data['A_paths'][0]
		print(real_B_path)  
  	# Convert the image data to a numpy array
		real_A = np.array(visuals["real_A"])
		fake_B = np.array(visuals["fake_B"])
		real_B = np.array(cv2.cvtColor(cv2.imread(real_B_path),cv2.COLOR_BGR2RGB))
		print(real_B.size)
		print(real_A.size)
		print(real_A.shape)
		print(fake_B.shape)
		print(real_B.shape)
  	# Convert the image array to PIL Image
		real_A1 = Image.fromarray(real_A)
		fake_B1 = Image.fromarray(fake_B)
		real_B1 = Image.fromarray(real_B)
		model_name_1 = "updated15S"
  	# Save the PIL Image as PNG
		real_A1.save(f'./new/{model_name_1}real_A_image{i}.png')
		fake_B1.save(f'./new/{model_name_1}fake_B_image{i}.png')
		real_B1.save(f'./new/{model_name_1}real_B_image{i}.png')
  
		avgPSNR_pred.append(PSNR(real_B,fake_B))
		avgPSNR_real.append(PSNR(real_B,real_A))
		print(avgPSNR_pred[-1], avgPSNR_real[-1])
   
		avgSSIM_pred.append(ssim(real_B,fake_B,full=True,channel_axis=2)[0])
		avgSSIM_real.append(ssim(real_B,real_A,full=True,channel_axis=2)[0])
		print(avgSSIM_pred[-1], avgSSIM_real[-1])
  	#img_path = model.get_image_paths()
  	#print('process image... %s' % img_path)
  
  	#visualizer.save_images(webpage, visuals, img_path)
	
# avgPSNR /= len(dataset)
#avgSSIM /= counter
print('PSNR_pred = %f' % np.mean(avgPSNR_pred))
print('PSNR_real = %f' % np.mean(avgPSNR_real))

print('SSIM_pred = %f' % np.mean(avgSSIM_pred))
print('SSIM_real = %f' % np.mean(avgSSIM_real))

webpage.save()
