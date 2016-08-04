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
require 'misc.recursive_atten_fact_att'
require 'misc.img_fact_encoding_att'
require 'misc.optim_updates'
local utils = require 'misc.utils'
require 'xlua'
require 'os'
local json=require 'cjson'
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

cmd:option('-start_from', 'model_id0_iter240000.t7', 'path to a model checkpoint to initialize model weights from. Empty = don\'t')
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
cmd:option('-checkpoint_path', 'save/train_vgg_fact_att', 'folder to save checkpoints into (empty = this folder)')
cmd:option('-load_path', 'save/train_vgg_fact_att_Alternating_w_ques', 'folder to save checkpoints into (empty = this folder)')

-- Visualization
cmd:option('-losses_log_every', 600, 'How often do we save losses, for inclusion in the progress dump? (0 = disable)')
cmd:option('-num_batches','5','Number of batches to be visualized')

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
--print(opt)
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
local loaded_checkpoint
local lmOpt
local iter=0
local learning_rate=opt.learning_rate
if string.len(opt.start_from) > 0 then
  local start_path = path.join(opt.load_path ,  opt.start_from)
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
protos.img_fact_encoding_w=nn.img_fact_encoding_att(lmOpt)

--protos.img_fact_encoding_p=nn.img_fact_encoding_att(lmOpt)


--protos.img_fact_encoding_q=nn.img_fact_encoding_att(lmOpt)
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
local iwparams, grad_iwparams = protos.img_fact_encoding_w:getParameters()
--local ipparams, grad_ipparams = protos.img_fact_encoding_p:getParameters()
--local iqparams, grad_iqparams = protos.img_fact_encoding_q:getParameters()

if string.len(opt.start_from) > 0 then
  print('Load the weight...')
  wparams:copy(loaded_checkpoint.wparams)
  pparams:copy(loaded_checkpoint.pparams)
  qparams:copy(loaded_checkpoint.qparams)
  aparams:copy(loaded_checkpoint.aparams)
  iwparams:copy(loaded_checkpoint.iparams)
--  iwparams:copy(loaded_checkpoint.iwparams)
--  ipparams:copy(loaded_checkpoint.ipparams) 
--  iqparams:copy(loaded_checkpoint.iqparams)
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

print('total number of parameters in img_fact_encoding w: ', iwparams:nElement())
assert(iwparams:nElement() == grad_iwparams:nElement())

--print('total number of parameters in img_fact_encoding p: ', ipparams:nElement())
--assert(ipparams:nElement() == grad_ipparams:nElement())
--
--print('total number of parameters in img_fact_encoding q: ', iqparams:nElement())
--assert(iqparams:nElement() == grad_iqparams:nElement())


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
  protos.img_fact_encoding_w:evaluate()
  --protos.img_fact_encoding_p:evaluate()
  --protos.img_fact_encoding_q:evaluate()
  loader:resetIterator(split)

  local n = 0
  local loss_sum = 0
  local loss_evals = 0
  local right_sum = 0
  local predictions = {}
  local total_num = loader:getDataNum(split)
  local total_num=opt.num_batches*opt.batch_size
  while true do
    local data = loader:getBatch{batch_size = opt.batch_size, split = split}
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
  --xlua.progress(n, total_num)
  

  input_json_file=io.open(opt.input_json,'r')
  input_json_file_text=input_json_file:read("*all")

  input_json_dict=json.decode(input_json_file_text)


  -- Check what all needs to be added here to accommodate the img_fact_feat
  local word_feat, img_feat, w_ques, w_img, mask = unpack(protos.word:forward({data.questions, data.images}))

  local conv_feat, p_ques, p_img = unpack(protos.phrase:forward({word_feat, data.ques_len, img_feat, mask}))

  local q_ques, q_img = unpack(protos.ques:forward({conv_feat, data.ques_len, img_feat, mask}))

  local img_fact_feat_w=protos.img_fact_encoding_w:forward( { data.captions , w_ques  }  )
  collectgarbage() 
--  local img_fact_feat_p=protos.img_fact_encoding_p:forward( { data.captions , p_ques  }  )
--  collectgarbage()
--  local img_fact_feat_q=protos.img_fact_encoding_q:forward( { data.captions , q_ques  }  )
--  collectgarbage()
--  local feature_ensemble = {w_ques, w_img, p_ques, p_img, q_ques, q_img, img_fact_feat_w  , img_fact_feat_p, img_fact_feat_q}
  local feature_ensemble = {w_ques, w_img, p_ques, p_img, q_ques, q_img, img_fact_feat_w  , img_fact_feat_w, img_fact_feat_w}
  local out_feat = protos.atten:forward(feature_ensemble)

  collectgarbage()


  for i=1,opt.batch_size do
    print("Question")
    --print(data.questions[i])
    for j=1,data.questions[i]:size(1) do
        if(data.questions[i][j]~=0) then
            io.write(input_json_dict.ix_to_word[tostring(data.questions[i][j]) ] .. " ")
        end
    end

    print("\nAnswer")
    print(input_json_dict.ix_to_ans[ tostring(data.answer[i] )  ])

   print("Captions")
   for j=1,data.captions[i]:size(1) do 
        io.write(tonumber(protos.img_fact_encoding_w.softmax_output[i][j][1]) .. " ")
        for k=1,data.captions[i]:size(2) do
            if(data.captions[i][j][k]~=0) then 
                io.write( input_json_dict.ix_to_word[tostring(data.captions[i][j][k]) ].." ")
            end
        end
        print("")
   end

  end


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
-- Main loop
-------------------------------------------------------------------------------

local loss0
local w_optim_state = {}
local p_optim_state = {}
local q_optim_state = {}
local a_optim_state = {}
local iw_optim_state = {}
local ip_optim_state = {}
local iq_optim_state = {}
local loss_history = {}
local accuracy_history = {}
local learning_rate_history = {}
local best_val_loss = 10000
local ave_loss = 0
local timer = torch.Timer()
local decay_factor = math.exp(math.log(0.1)/opt.learning_rate_decay_every/opt.iterPerEpoch)
local learning_rate = opt.learning_rate
-- create the path to save the model.
--paths.mkdir(opt.checkpoint_path .. '_' .. opt.co_atten_type)
--os.execute('mkdir -p '..sys.dirname(opt.checkpoint_path .. '_' .. opt.co_atten_type))
--print("Directory to make")
--print( opt.checkpoint_path .. '_' .. opt.co_atten_type)

local val_loss, val_accu = eval_split(2)
print('validation loss: ', val_loss, 'accuracy ', val_accu)

