"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
from __future__ import print_function
from utils import get_config, pytorch03_to_pytorch04
from trainer import MUNIT_Trainer, UNIT_Trainer
import argparse
from torch.autograd import Variable
from subprocess import call
import torchvision.utils as vutils
import sys
import torch
import os
import glob
from torchvision import transforms
from PIL import Image

def usage():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help="net configuration", required=True)
    parser.add_argument('--input_image', type=str, help="input image path", required=True)
    parser.add_argument('--output_folder', type=str, help="output image path", required=True)
    parser.add_argument('--checkpoint', type=str, help="checkpoint of autoencoders", required=True)
    parser.add_argument('--input_folder', type=str, help="input folder path", required=False)#added this line
    parser.add_argument('--start_num', type=int,  default=0, help="number of image to start from for the conversion of a folder")#parameter to use if colab interrupts so that the next time we can start from the image number start_num
    parser.add_argument('--style', type=str, default='', help="style image path")
    parser.add_argument('--a2b', type=int, default=1, help="1 for a2b and others for b2a")
    parser.add_argument('--seed', type=int, default=10, help="random seed")
    parser.add_argument('--num_style',type=int, default=10, help="number of styles to sample")
    parser.add_argument('--synchronized', action='store_true', help="whether use synchronized style code or not")
    parser.add_argument('--output_only', action='store_true', help="whether use synchronized style code or not")
    parser.add_argument('--output_path', type=str, default='.', help="path for logs, checkpoints, and VGG model weight")
    parser.add_argument('--trainer', type=str, default='UNIT', help="MUNIT|UNIT")
    parser.add_argument('--device', metavar='GPU', nargs='+', help='GPU List', default=["0"])
    return parser.parse_args()
	
opts = usage()

# Choose GPU device to run
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"]=",".join(str(x) for x in opts.device)

torch.manual_seed(opts.seed)
torch.cuda.manual_seed(opts.seed)
if not os.path.exists(opts.output_folder):
    os.makedirs(opts.output_folder)

# Print System Info 
#print('CUDA Devices') #commented
#call(["nvidia-smi", "--format=csv", "--query-gpu=index,name,driver_version,memory.total,memory.used,memory.free"]) #commented
#print(f'Available devices {torch.cuda.device_count()}: {", ".join(str(x) for x in opts.device)}') #commented
#print('Active CUDA Device: GPU', torch.cuda.current_device())  #commented

#print('Load experiment setting')   #commented
config = get_config(opts.config)
opts.num_style = 1 if opts.style != '' else opts.num_style

#print('Setup model and data loader')   #commented
config['vgg_model_path'] = opts.output_path
config['resnet_model_path'] = opts.output_path
if opts.trainer == 'MUNIT':
    style_dim = config['gen']['style_dim']
    trainer = MUNIT_Trainer(config)
elif opts.trainer == 'UNIT':
    trainer = UNIT_Trainer(config)
else:
    sys.exit("Only support MUNIT|UNIT")

try:
    state_dict = torch.load(opts.checkpoint)
    trainer.gen_a.load_state_dict(state_dict['a'])
    trainer.gen_b.load_state_dict(state_dict['b'])
except:
    state_dict = pytorch03_to_pytorch04(torch.load(opts.checkpoint))
    trainer.gen_a.load_state_dict(state_dict['a'])
    trainer.gen_b.load_state_dict(state_dict['b'])

trainer.cuda()
trainer.eval()
encode = trainer.gen_a.encode if opts.a2b else trainer.gen_b.encode # encode function
style_encode = trainer.gen_b.encode if opts.a2b else trainer.gen_a.encode # encode function
decode = trainer.gen_b.decode if opts.a2b else trainer.gen_a.decode # decode function

if 'new_size' in config:
    new_size = config['new_size']
else:
    if opts.a2b==1:
        new_size = config['new_size_a']
    else:
        new_size = config['new_size_b']

with torch.no_grad():
    if opts.input_image == 'no':			#if i put "no" as a string to input_image it means I have an input folder
        image_path = []	#list that contains all the images paths that are in the folder
        image_name = []	#list that contains all the names of the images
        for filepath in glob.glob(opts.input_folder+'*.jpg'):	#save in a list of paths the paths of every image in the folder
            image_path.append(filepath)
        image_name = [os.path.basename(el)[:-4] for el in image_path]   #save just the name of the image (without the .jpg) in the folder (use this to name the input and output files)
        cc=0
		#from here we have the FOR LOOP that converts every image in the folder
        for count, el in enumerate(image_path, start=opts.start_num):
            transform = transforms.Compose([transforms.Resize(new_size),transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            image = Variable(transform(Image.open(image_path[count]).convert('RGB')).unsqueeze(0).cuda())
            style_image = Variable(transform(Image.open(opts.style).convert('RGB')).unsqueeze(0).cuda()) if opts.style != '' else None
			
            content, _ = encode(image)

            if opts.trainer == 'MUNIT':
                style_rand = Variable(torch.randn(opts.num_style, style_dim, 1, 1).cuda())
                if opts.style != '':
                    _, style = style_encode(style_image)
                else:
                    style = style_rand
                for j in range(opts.num_style):
                    s = style[j].unsqueeze(0)
                    outputs = decode(content, s)
                    outputs = (outputs + 1) / 2.
                    output_name = image_name[count] + 'n.jpg'	#I want the output name to be like the input name but with an n at the end to indicate that it's a night image
                    path = os.path.join(opts.output_folder, output_name.format(j))		#adding the name I want for the output
                    vutils.save_image(outputs.data, path, padding=0, normalize=True)
            elif opts.trainer == 'UNIT':
                outputs = decode(content)
                outputs = (outputs + 1) / 2.
                output_name = image_name[count] + 'n.jpg'
                path = os.path.join(opts.output_folder, output_name)		#here I use my output name as well
                vutils.save_image(outputs.data, path, padding=0, normalize=True)
            else:
                pass
	
            if not opts.output_only:
				# also save input images
                vutils.save_image(image.data, os.path.join(opts.output_folder, image_path[count]), padding=0, normalize=True)
            cc=count
            if count%20==0:
                print(' Picture number ' + str(count))
        print('Conversion complete!'+str(cc))
	
	
    else:
	    #from here is the case in which we have only the input image and not the input folder
        transform = transforms.Compose([transforms.Resize(new_size), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        image = Variable(transform(Image.open(opts.input).convert('RGB')).unsqueeze(0).cuda())
        style_image = Variable(transform(Image.open(opts.style).convert('RGB')).unsqueeze(0).cuda()) if opts.style != '' else None
        content, _ = encode(image)

        if opts.trainer == 'MUNIT':
            style_rand = Variable(torch.randn(opts.num_style, style_dim, 1, 1).cuda())
            if opts.style != '':
                _, style = style_encode(style_image)
            else:
                style = style_rand
            for j in range(opts.num_style):
                s = style[j].unsqueeze(0)
                outputs = decode(content, s)
                outputs = (outputs + 1) / 2.
                path = os.path.join(opts.output_folder, 'output{:03d}.jpg'.format(j))
                vutils.save_image(outputs.data, path, padding=0, normalize=True)
        elif opts.trainer == 'UNIT':
            outputs = decode(content)
            outputs = (outputs + 1) / 2.
            path = os.path.join(opts.output_folder, 'output.jpg')
            vutils.save_image(outputs.data, path, padding=0, normalize=True)
        else:
            pass

        if not opts.output_only:
	    # also save input images
            vutils.save_image(image.data, os.path.join(opts.output_folder, 'input.jpg'), padding=0, normalize=True)
        print('Testing complete!') 
