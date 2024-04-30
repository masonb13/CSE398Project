import time
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer
from util.metrics import PSNR, SSIM
from multiprocessing import freeze_support
from skimage.metrics import structural_similarity as ssim

def train(opt, data_loader, model):
	dataset = data_loader.load_data()
	dataset_size = len(data_loader)
	print('#training images = %d' % dataset_size)
	print(opt.save_latest_freq)
	print(opt.save_epoch_freq)
	total_steps = 0
#	for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
	for epoch in range(opt.epoch_count):
		model_name_1 = "updated15S-"
		print("Epoch",epoch)
		epoch_start_time = time.time()
		epoch_iter = 0
		for i, data in enumerate(dataset):
			if i==300:
				break   
			iter_start_time = time.time()
			total_steps += opt.batchSize
			epoch_iter += opt.batchSize
			model.set_input(data)
			model.optimize_parameters()
#			if i==500:
#				break

			if total_steps % opt.display_freq == 0:
				results = model.get_current_visuals()
				psnrMetric = PSNR(results['Restored_Train'], results['Sharp_Train'])
				print('PSNR on Train = %f' % psnrMetric)
				ssimMetric = ssim(results['Restored_Train'], results['Sharp_Train'],full=True,channel_axis=2)[0]
				print('PSNR on Train = %f' % ssimMetric)
				#visualizer.display_current_results(results, epoch)

			if total_steps % opt.print_freq == 0:
				#errors = model.get_current_errors()
				t = (time.time() - iter_start_time) / opt.batchSize
				#visualizer.print_current_errors(epoch, epoch_iter, errors, t)
				#if opt.display_id > 0:
					#visualizer.plot_current_errors(epoch, float(epoch_iter)/dataset_size, opt, errors)

			if total_steps % opt.save_latest_freq == 0:
				print('saving the latest model (epoch %d, total_steps %d)' % (epoch, total_steps))
				model.save('latest'+model_name_1+str(epoch))

		if epoch % opt.save_epoch_freq == 0:
			print('saving the model at the end of epoch %d, iters %d' % (epoch, total_steps))
			model.save('latest'+model_name_1+str(epoch))
			model.save(model_name_1+str(epoch))

		print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

		if epoch > opt.niter:
			model.update_learning_rate()
	model.save('latest'+model_name_1)


if __name__ == '__main__':
	freeze_support()

	# python train.py --dataroot /.path_to_your_data --learn_residual --resize_or_crop crop --fineSize CROP_SIZE (we used 256)

	opt = TrainOptions().parse()
	opt.dataroot = '../data'
	opt.learn_residual = True
	opt.resize_or_crop = "crop"
	opt.fineSize = 256
	opt.gan_type = "gan"
	opt.epoch_count = 300
	opt.save_epoch_freq = 25
	opt.continue_train = False #---------------------------------
	#opt.which_model_netG = "resnet_6blocks"

	# default = 100
	opt.print_freq = 20

	data_loader = CreateDataLoader(opt)
	model = create_model(opt)
	#visualizer = Visualizer(opt)
	#print("1---------------------------------\n-\n-\n-")
	train(opt, data_loader, model)
