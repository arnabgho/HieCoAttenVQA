------------------------------------------------------------------------------
--  Hierarchical Question-Image Co-Attention for Visual Question Answering
--  J. Lu, J. Yang, D. Batra, and D. Parikh
--  https://arxiv.org/abs/1606.00061, 2016
--  if you have any question about the code, please contact jiasenlu@vt.edu
-----------------------------------------------------------------------------
--
-- Check whether the dependency files need to be changed to the ones with the image fact 
require 'nn'
require 'torch'
require 'optim'
require 'misc.DataLoaderDiskFact'
require 'misc.word_level'
require 'misc.phrase_level'
require 'misc.ques_level'
require 'misc.recursive_atten_fact'
require 'misc.img_fact_encoding'
require 'misc.optim_updates'
cjson=require 'cjson'
local utils = require 'misc.utils'
require 'xlua'
require 'os'

-------------------------------------------------------------------------------
-- Input arguments and options
-------------------------------------------------------------------------------

cmd = torch.CmdLine()
cmd:text()
cmd:text('Train a Visual Question Answering model')
cmd:text()
cmd:text('Options')

-- Check that the data input files are correct for the Visual Genome Dataset
-- Data input settings
cmd:option('-input_img_train_h5','data/vqa_data_img_vgg_train.h5','path to the h5file containing the image feature')
cmd:option('-input_img_test_h5','data/vqa_data_img_vgg_test.h5','path to the h5file containing the image feature')
cmd:option('-input_ques_h5','data/vqa_fact_data_prepro.h5','path to the h5file containing the preprocessed dataset')
cmd:option('-input_json','data/vqa_fact_data_prepro.json','path to the json file containing additional info and vocab')

cmd:option('-start_from', '', 'path to a model checkpoint to initialize model weights from. Empty = don\'t')
cmd:option('-co_atten_type', 'Alternating', 'co_attention type. Parallel or Alternating, alternating trains more faster than parallel.')
cmd:option('-feature_type', 'VGG', 'VGG or Residual')


cmd:option('-hidden_size',512,'the hidden layer size of the model.')
cmd:option('-rnn_size',512,'size of the rnn in number of hidden nodes in each layer')
cmd:option('-batch_size',20,'what is theutils batch size in number of images per batch? (there will be x seq_per_img sentences)')
cmd:option('-output_size', 1000, 'number of output answers')
cmd:option('-rnn_layers',2,'number of the rnn layer')


-- Optimization
cmd:option('-optim','rmsprop','what update to use? rmsprop|sgd|sgdmom|adagrad|adam')
cmd:option('-learning_rate',4e-4,'learning rate')
cmd:option('-learning_rate_decay_start', 0, 'at what iteration to start decaying learning rate? (-1 = dont)')
cmd:option('-learning_rate_decay_every', 300, 'every how many epoch thereafter to drop LR by 0.1?')
cmd:option('-optim_alpha',0.99,'alpha for adagrad/rmsprop/momentum/adam')
cmd:option('-optim_beta',0.995,'beta used for adam')
cmd:option('-optim_epsilon',1e-8,'epsilon that goes into denominator in rmsprop')
cmd:option('-max_iters', -1, 'max number of iterations to run for (-1 = run forever)')
cmd:option('-iterPerEpoch', 1200)
cmd:option('-max_relations',20)
-- Evaluation/Checkpointing
cmd:option('-save_checkpoint_every', 6000, 'how often to save a model checkpoint?')
cmd:option('-checkpoint_path', 'save/train_vgg_fact', 'folder to save checkpoints into (empty = this folder)')

-- Visualization
cmd:option('-losses_log_every', 600, 'How often do we save losses, for inclusion in the progress dump? (0 = disable)')


-- Add the parameters for the image_fact features such as image_fact_size etc and the maximum number of relations


-- misc
cmd:option('-id', '0', 'an id identifying this run/job. used in cross-val and appended when writing progress files')
cmd:option('-backend', 'cudnn', 'nn|cudnn')
cmd:option('-gpuid', 6, 'which gpu to use. -1 = use CPU')
cmd:option('-seed', 123, 'random number generator seed to use')

cmd:text()

-------------------------------------------------------------------------------
-- Basic Torch initializations
-------------------------------------------------------------------------------
local opt = cmd:parse(arg)
torch.manualSeed(opt.seed)
print(opt)
torch.setdefaulttensortype('torch.FloatTensor') -- for CPU

if opt.gpuid >= 0 then
  require 'cutorch'
  require 'cunn'
  if opt.backend == 'cudnn' then 
  require 'cudnn' 
  end
  --cutorch.manualSeed(opt.seed)
  --cutorch.setDevice(opt.gpuid+1) -- note +1 because lua is 1-indexed
end

opt = cmd:parse(arg)

-------------------------------------------------------------------------------
-- Create the Data Loader instance
-------------------------------------------------------------------------------
--Add the data loader for the image fact representation 
local loader = DataLoader{h5_img_file_train = opt.input_img_train_h5, h5_img_file_test = opt.input_img_test_h5, h5_ques_file = opt.input_ques_h5, json_file = opt.input_json, feature_type = opt.feature_type}
------------------------------------------------------------------------
--Design Parameters and Network Definitions
------------------------------------------------------------------------
local protos = {}
print('Building the model...')
-- intialize language model
local iter = 0
local loaded_checkpoint
local lmOpt
local learning_rate = opt.learning_rate
if string.len(opt.start_from) > 0 then
  local start_path = path.join(opt.checkpoint_path .. '_' .. opt.co_atten_type ,  opt.start_from)
  loaded_checkpoint = torch.load(start_path)
  lmOpt = loaded_checkpoint.lmOpt
  str_iter= opt.start_from:match("model_id0_iter([^.]+).t7")
  iter=tonumber(str_iter)
else
  lmOpt = {}
  lmOpt.vocab_size = loader:getVocabSize()
  lmOpt.hidden_size = opt.hidden_size
  lmOpt.rnn_size = opt.rnn_size
  lmOpt.num_layers = opt.rnn_layers
  lmOpt.dropout = 0.5
  lmOpt.seq_length = loader:getSeqLength()
  lmOpt.batch_size = opt.batch_size
  lmOpt.output_size = opt.rnn_size
  lmOpt.atten_type = opt.co_atten_type
  lmOpt.feature_type = opt.feature_type
  lmOpt.max_relations=opt.max_relations
  lmOpt.gpu= (opt.gpuid>=0)
end

protos.word = nn.word_level(lmOpt)
protos.phrase = nn.phrase_level(lmOpt)
protos.ques = nn.ques_level(lmOpt)
protos.img_fact_encoding=nn.img_fact_encoding(lmOpt)

protos.atten = nn.recursive_atten(lmOpt)
protos.crit = nn.CrossEntropyCriterion()

-- ship everything to GPU, maybe

if opt.gpuid >= 0 then
  for k,v in pairs(protos) do v:cuda() end
end

local wparams, grad_wparams = protos.word:getParameters()
local pparams, grad_pparams = protos.phrase:getParameters()
local qparams, grad_qparams = protos.ques:getParameters()
local aparams, grad_aparams = protos.atten:getParameters()
local iparams, grad_iparams = protos.img_fact_encoding:getParameters()

if string.len(opt.start_from) > 0 then
  print('Load the weight...')
  wparams:copy(loaded_checkpoint.wparams)
  pparams:copy(loaded_checkpoint.pparams)
  qparams:copy(loaded_checkpoint.qparams)
  aparams:copy(loaded_checkpoint.aparams)
  iparams:copy(loaded_checkpoint.iparams)
  learning_rate=loaded_checkpoint.learning_rate
end

print('total number of parameters in word_level: ', wparams:nElement())
assert(wparams:nElement() == grad_wparams:nElement())

print('total number of parameters in phrase_level: ', pparams:nElement())
assert(pparams:nElement() == grad_pparams:nElement())

print('total number of parameters in ques_level: ', qparams:nElement())
assert(qparams:nElement() == grad_qparams:nElement())
protos.ques:shareClones()

print('total number of parameters in recursive_attention: ', aparams:nElement())
assert(aparams:nElement() == grad_aparams:nElement())

print('total number of parameters in img_fact_encoding: ', iparams:nElement())
assert(iparams:nElement() == grad_iparams:nElement())


collectgarbage() 

-------------------------------------------------------------------------------
-- Validation evaluation
-------------------------------------------------------------------------------

-------------------------------------------------
-- See how to include the image fact encoding
-------------------------------------------------
local function eval_split(split)

  protos.word:evaluate()
  protos.phrase:evaluate()
  protos.ques:evaluate()
  protos.atten:evaluate()
  protos.img_fact_encoding:evaluate()
  loader:resetIterator(split)

  local n = 0
  local loss_sum = 0
  local loss_evals = 0
  local right_sum = 0
  local predictions = {}
  local total_num = loader:getDataNum(split)
  local total_num=1000
  while true do
    local data = loader:getBatch{batch_size = opt.batch_size, split = split}
    local precision=1e-5
   -- local jac = nn.Jacobian
   -- local err=jac.testJacobian(protos.img_fact_encoding,data.captions)
   -- print("====> error: "..err)
   -- if err<precision then
   --     print("============> module OK")
   -- else
   --     print("============> error too large incorrect implementation")
   --   --  os.execute("sleep 1")
   -- end
   -- ship the data to cuda
    if opt.gpuid >= 0 then
      data.answer = data.answer:cuda()
      data.images = data.images:cuda()
      data.questions = data.questions:cuda()
      data.ques_len = data.ques_len:cuda()
      --data.img_fact=data.img_fact:cuda()
      data.captions=data.captions:cuda()
    end
  n = n + data.images:size(1)
  xlua.progress(n, total_num)
  

  --local img_fact_feat=protos.img_fact_encoding:forward( { data.captions}  )
  -- Check Gradient Computation with Jacobian
--  local precision=1e-5
--  local jac = nn.Jacobian
--  local err=jac.testJacobian(protos.img_fact_encoding,data.captions)
--  print("====> error: "..err)
--  if err<precision then
--      print("============> module OK")
--  else
--      print("============> error too large incorrect implementation")
--      os.execute("sleep 1")
--  end
  --------------------------------------------


  local img_fact_feat=protos.img_fact_encoding:forward( { data.captions }  )
  -- Check what all needs to be added here to accommodate the img_fact_feat
  local word_feat, img_feat, w_ques, w_img, mask = unpack(protos.word:forward({data.questions, data.images}))

  local conv_feat, p_ques, p_img = unpack(protos.phrase:forward({word_feat, data.ques_len, img_feat, mask}))

  local q_ques, q_img = unpack(protos.ques:forward({conv_feat, data.ques_len, img_feat, mask}))

  local feature_ensemble = {w_ques, w_img, p_ques, p_img, q_ques, q_img, img_fact_feat}
  local out_feat = protos.atten:forward(feature_ensemble)

  -- forward the language model criterion
  local loss = protos.crit:forward(out_feat, data.answer)

    local tmp,pred=torch.max(out_feat,2)

    for i = 1, pred:size()[1] do

      if pred[i][1] == data.answer[i] then
        right_sum = right_sum + 1
      end
    end

    loss_sum = loss_sum + loss
    loss_evals = loss_evals + 1
    if n >= total_num then break end
  end

  return loss_sum/loss_evals, right_sum / total_num
end


-------------------------------------------------------------------------------
-- Loss function
-------------------------------------------------------------------------------
local function lossFun()        -- The MVP , see what the individual functions are doing 
  protos.word:training()
  grad_wparams:zero()  

  protos.phrase:training()
  grad_pparams:zero()

  protos.ques:training()
  grad_qparams:zero()

  protos.atten:training()
  grad_aparams:zero()

  protos.img_fact_encoding:training()
  grad_iparams:zero()
  ----------------------------------------------------------------------------
  -- Forward pass
  -----------------------------------------------------------------------------
  -- get batch of data  
  local data = loader:getBatch{batch_size = opt.batch_size, split = 0}
  if opt.gpuid >= 0 then
    data.answer = data.answer:cuda()
    data.questions = data.questions:cuda()
    data.ques_len = data.ques_len:cuda()
    data.images = data.images:cuda()
    --data.img_fact=data.img_fact:cuda()
    data.captions=data.captions:cuda()
  end

  --local img_fact_feat=protos.img_fact_encoding:forward( { data.captions  }  )

  local img_fact_feat=protos.img_fact_encoding:forward( {data.captions} )

  -- Check what all needs to be modified in these files to accommodate img_fact_feat
  local word_feat, img_feat, w_ques, w_img, mask = unpack(protos.word:forward({data.questions, data.images}))

  local conv_feat, p_ques, p_img = unpack(protos.phrase:forward({word_feat, data.ques_len, img_feat, mask}))

  local q_ques, q_img = unpack(protos.ques:forward({conv_feat, data.ques_len, img_feat, mask}))

  local feature_ensemble = {w_ques, w_img, p_ques, p_img, q_ques, q_img,img_fact_feat}
  local out_feat = protos.atten:forward(feature_ensemble)
  
  collectgarbage()
  -- forward the language model criterion
  local loss = protos.crit:forward(out_feat, data.answer)
  -----------------------------------------------------------------------------
  -- Backward pass
  -----------------------------------------------------------------------------
  -- backprop criterion
  local dlogprobs = protos.crit:backward(out_feat, data.answer)
  
  local d_w_ques, d_w_img, d_p_ques, d_p_img, d_q_ques, d_q_img , d_img_fact_feat = unpack(protos.atten:backward(feature_ensemble, dlogprobs))

  local d_ques_feat, d_ques_img = unpack(protos.ques:backward({conv_feat, data.ques_len, img_feat}, {d_q_ques, d_q_img}))
    
  --local d_ques1 = protos.bl1:backward({ques_feat_0, data.ques_len}, d_ques2)
  local d_conv_feat, d_conv_img = unpack(protos.phrase:backward({word_feat, data.ques_len, img_feat}, {d_ques_feat, d_p_ques, d_p_img}))
  
  local dummy = protos.word:backward({data.questions, data.images}, {d_conv_feat, d_w_ques, d_w_img, d_conv_img, d_ques_img})

  local dummy2=protos.img_fact_encoding:backward( {data.captions} , { d_img_fact_feat   } )
  collectgarbage() 
--  print("sum of grad_iparams")
--  print(torch.sum(grad_iparams))
--  print("sum of grad_aparams")
--  print(torch.sum(grad_aparams))
 --print(d_img_fact_feat.modules[1].gradInput)
  -- Check the modification that needs to be done to allow to backpropagate through image_fact_encoding
  --
  --
  -----------------------------------------------------------------------------
  -- and lets get out!
  local stats = {}
  stats.dt = dt
  local losses = {}
  losses.total_loss = loss
  return losses, stats
end

-------------------------------------------------------------------------------
-- Main loop
-------------------------------------------------------------------------------

local loss0
local w_optim_state = {}
local p_optim_state = {}
local q_optim_state = {}
local a_optim_state = {}
local i_optim_state = {}
local loss_history = {}
local accuracy_history = {}
local learning_rate_history = {}
local best_val_loss = 10000
local ave_loss = 0
local timer = torch.Timer()
local decay_factor = math.exp(math.log(0.1)/opt.learning_rate_decay_every/opt.iterPerEpoch)
-- create the path to save the model.
paths.mkdir(opt.checkpoint_path .. '_' .. opt.co_atten_type)
--os.execute('mkdir -p '..sys.dirname(opt.checkpoint_path .. '_' .. opt.co_atten_type))
print("Directory to make")
print( opt.checkpoint_path .. '_' .. opt.co_atten_type)
while true do
  -- eval loss/gradient
  local losses, stats = lossFun()
  ave_loss = ave_loss + losses.total_loss
  -- decay the learning rate
  if iter > opt.learning_rate_decay_start and opt.learning_rate_decay_start >= 0 then
    learning_rate = learning_rate * decay_factor -- set the decayed rate
  end

  if iter % opt.losses_log_every == 0 then
    ave_loss = ave_loss / opt.losses_log_every
    loss_history[iter] = losses.total_loss 
    accuracy_history[iter] = ave_loss
    learning_rate_history[iter] = learning_rate

    print(string.format('iter %d: %f, %f, %f, %f', iter, losses.total_loss, ave_loss, learning_rate, timer:time().real))

    ave_loss = 0
  end

  -- save checkpoint once in a while (or on final iteration)
  if (iter % opt.save_checkpoint_every == 0 or iter == opt.max_iters) then
      local val_loss, val_accu = eval_split(2)
      print('validation loss: ', val_loss, 'accuracy ', val_accu)

      local checkpoint_path = path.join(opt.checkpoint_path .. '_' .. opt.co_atten_type, 'model_id' .. opt.id .. '_iter'.. iter)
      torch.save(checkpoint_path..'.t7', {learning_rate=learning_rate,wparams=wparams, pparams = pparams, qparams=qparams, aparams=aparams, iparams=iparams , lmOpt=lmOpt}) 

      local checkpoint = {}
      checkpoint.opt = opt
      checkpoint.iter = iter
      checkpoint.loss_history = loss_history
      checkpoint.accuracy_history = accuracy_history
      checkpoint.learning_rate_history = learning_rate_history


      local checkpoint_path = path.join(opt.checkpoint_path .. '_' .. opt.co_atten_type, 'checkpoint' .. '.json')
      --utils.write_json(checkpoint_path, checkpoint)
      --print('wrote json checkpoint to ' .. checkpoint_path .. '.json')

  end

  -- perform a parameter update
  if opt.optim == 'rmsprop' then
    rmsprop(wparams, grad_wparams, learning_rate, opt.optim_alpha, opt.optim_epsilon, w_optim_state)
    rmsprop(pparams, grad_pparams, learning_rate, opt.optim_alpha, opt.optim_epsilon, p_optim_state)
    rmsprop(qparams, grad_qparams, learning_rate, opt.optim_alpha, opt.optim_epsilon, q_optim_state)
    rmsprop(aparams, grad_aparams, learning_rate, opt.optim_alpha, opt.optim_epsilon, a_optim_state)
    rmsprop(iparams, grad_iparams, learning_rate, opt.optim_alpha, opt.optim_epsilon, i_optim_state)


  else
    error('bad option opt.optim')
  end

  iter = iter + 1
  xlua.progress(iter, (tonumber(math.floor(iter/opt.save_checkpoint_every) )+1)*opt.save_checkpoint_every )
  if opt.max_iters > 0 and iter >= opt.max_iters then break end -- stopping criterion
end
