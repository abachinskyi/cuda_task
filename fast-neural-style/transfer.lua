require 'nngraph'
require 'imgtools'
require 'nn'
require 'cudnn'
require 'cunn'
require 'cutorch'



local cmd = torch.CmdLine()

cmd:option('-content_img','image_sets/tubingen.jpg','The image to apply style transfer on')
cmd:option('-transfer_model','models/newlytrainedmodel.t7','The model of transferring')
cmd:option('-output','output.jpg','The path to output')
cmd:option('-output_size',false,'Maximum edge of the output image')

function main(params)
	cutorch.setDevice(1)
	cutorch.setDevice(2)
	timer = torch.Timer()
	transfer_model = torch.load(params.transfer_model)
	transfer_model = transfer_model:cuda()
	local image = loadimg(params.content_img,params.output_size):float()
	image = image:cuda()
	local org_size = #image
	local content_batch_size = torch.LongStorage(4)
	content_batch_size[1] = 1
	for i=1,3 do
		content_batch_size[i+1] = (#image)[i]
		print(content_batch_size[i+1])
	end
	image = torch.reshape(image,content_batch_size):cuda()
	local newimg = transfer_model:forward(image)
	newimg = newimg:float()
	newimg = torch.reshape(newimg,org_size)

	saveimg(newimg, params.output)
	print(cutorch.getDevice())
	print('Transfer complete in '.. timer:time().real ..' seconds!')
end

local params = cmd:parse(arg)
main(params)