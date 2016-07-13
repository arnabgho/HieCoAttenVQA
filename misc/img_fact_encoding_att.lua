require 'rnn'
require 'nn'

--local img_graph_encoding={}
--img_graph_encoding.__index=img_graph_encoding

local utils=require 'misc.utils'
local LanguageEmbedding = require 'misc.LanguageEmbedding'

local layer , parent= torch.class('nn.img_fact_encoding_att','nn.Module')

function layer:__init( opt )
    parent.__init(self)
    self.max_relations=utils.getopt(opt,'max_relations')
    self.vocab_size = utils.getopt(opt, 'vocab_size') -- required
    self.image_doc_size=utils.getopt(opt,'hidden_size') 
    self.hidden_size = utils.getopt(opt, 'hidden_size')
    local dropout = utils.getopt(opt, 'dropout', 0)
    self.seq_length = utils.getopt(opt, 'seq_length')
    self.atten_type = utils.getopt(opt, 'atten_type')
    self.feature_type = utils.getopt(opt, 'feature_type')
    self.LE = LanguageEmbedding.LE(self.vocab_size, self.hidden_size/4, self.hidden_size/4, self.seq_length)    
    --self.LE=nn.LookupTable(self.vocab_size,self.hidden_size)

    self.gpu=utils.getopt(opt,'gpu',true)
    self.fact_encoder=nn.Sequential()
    self.fact_encoder:add(self.LE)
    self.fact_encoder:add(nn.SplitTable(1,2))
    self.fact_encoder:add(nn.Sequencer(nn.LSTM( self.hidden_size/4, self.hidden_size )  ))
    --self.fact_encoder:add(nn.Sequencer( nn.LSTM( self.hidden_size, self.hidden_size )  ))
    self.fact_encoder:add(nn.SelectTable(-1) )
    
    -- doc_encoder : to convert the relationship encodings into a single encoding using an LSTM
   -- self.doc_encoder=nn.Sequential()
   -- self.doc_encoder:add(nn.SplitTable(1,2))
   -- self.doc_encoder:add(nn.Sequencer(nn.LSTM( self.hidden_size , self.image_doc_size  )))
   -- self.doc_encoder:add(nn.SelectTable(-1))
    self.doc_encoder=nn.Sequential()
    self.doc_encoder:add(nn.Linear(self.hidden_size,self.hidden_size) )
    self.MM1=nn.MM()
    self.MM2=nn.MM(true,false)
    self.softmax=nn.SoftMax()
end

function layer:getModulesList()
    return {self.LE, self.fact_encoder , self.doc_encoder}
end


function layer:parameters()
    local p1,g1=self.fact_encoder:parameters()
    local p2,g2=self.doc_encoder:parameters()
    local p3,g3=self.LE:parameters() -- Check Whether The Lookup Tables' parameters are to be included
    
    local params = {}
    for k,v in pairs(p1) do table.insert(params, v) end
    for k,v in pairs(p2) do table.insert(params, v) end
    for k,v in pairs(p3) do table.insert(params, v) end

    local grad_params = {}
    for k,v in pairs(g1) do table.insert(grad_params, v) end
    for k,v in pairs(g2) do table.insert(grad_params, v) end
    for k,v in pairs(g3) do table.insert(grad_params, v) end

    return params, grad_params
end

function layer:training()
    self.LE:training()
    self.fact_encoder:training()
    self.doc_encoder:training()
end

function layer:evaluate()
    self.LE:evaluate()
    self.fact_encoder:evaluate()
    self.doc_encoder:evaluate()
end

--function layer:updateOutput( input )
--    local img_doc_relations=input[1]
--    local question_embedding=input[2]
--    
--
--    local batch_size=img_doc_relations:size(1)
--    --self.relation_embeddings={}
--    local doc_encoder_input=torch.Tensor( batch_size, self.max_relations , self.hidden_size  ) 
--    local batch_doc_encoded=torch.Tensor(batch_size , self.hidden_size):zero()
--    local softmax_input=torch.Tensor(batch_size , self.max_relations)
--
--    if(self.gpu) then
--        doc_encoder_input=doc_encoder_input:cuda()
--        batch_doc_encoded=batch_doc_encoded:cuda()
--        softmax_input=softmax_input:cuda()
--    end
--    for i=1,batch_size do
--        local fact_encoder_input=img_doc_relations[i]
--        local encoded_facts=self.fact_encoder:forward(fact_encoder_input)  --encoded_facts:size max_relations x self.hidden_size
--        doc_encoder_input[i]=encoded_facts
--    end
--    local transform_ques=self.doc_encoder:forward( question_embedding) --transform_ques : batch_size x self.hidden_size
--    for i=1,batch_size do
--        softmax_input[i]=doc_encoder_input[i]*transform_ques[i]
--    end
--
--    softmax_input=nn.SoftMax():forward(softmax_input)
--
--    for i=1,batch_size do
--        for j=1,self.max_relations do
--            batch_doc_encoded[i]=batch_doc_encoded[i] + softmax_input[i][j]*doc_encoder_input[i][j]
--        end
--    end
--   -- for i=1,batch_size do
--   --     local attention_input=doc_encoder_input[i]     --attention_input : max_relations x self.hidden_size
--   --     local softmax_input[i]=attention_input * transform_ques[i]:t()
--   -- end
--    return batch_doc_encoded
--end


function layer:updateOutput(input)
    local batch_size=input[1]:size(1)

    input[1]=input[1]:reshape(batch_size * self.max_relations,self.seq_length)
    self.doc_encoder_input=self.fact_encoder:forward(input[1]):reshape(batch_size , self.max_relations , self.hidden_size)
    self.transform_ques=self.doc_encoder:forward(input[2]):reshape(batch_size , self.hidden_size ,1) -- transform_ques : batch_size x hidden_size
   
    self.softmax_input=self.MM1:cuda():forward({ self.doc_encoder_input,self.transform_ques }):reshape(batch_size,self.max_relations)
    self.softmax_output=self.softmax:cuda():forward(self.softmax_input):reshape(batch_size,self.max_relations,1)
    --softmax_input : batch_size x max_relations x 1

    return self.MM2:cuda():forward({self.doc_encoder_input,self.softmax_output}):reshape(batch_size,self.hidden_size)
end

-- This Function Need Not Be Computed Right Now since the computation doc will help and do it automatically 
function layer:updateGradInput( input , gradOutput  )
    local batch_size=input[1]:size(1)

    local d_MM2=self.MM2:cuda():backward( {self.doc_encoder_input , self.softmax_output} , gradOutput[1]:reshape(batch_size , self.hidden_size,1 )   )

    -- d_MM2[1] : d_self.doc_encoder_input 
    -- d_MM2[2] : d_self.softmax_output
    local d_softmax_input=self.softmax:cuda():backward( self.softmax_input , d_MM2[2]:reshape( batch_size , self.max_relations ) )
    --local d_softmax_input=torch.cmul(self.softmax_input,d_MM2[2]:reshape(batch_size,self.max_relations) )
    -- d_softmax_input : batch_size x self.max_relations
    local d_MM1=self.MM1:cuda():backward({ self.doc_encoder_input, self.transform_ques  } , d_softmax_input:reshape( batch_size , self.max_relations , 1 ) )

    --d_MM1[1] : d_self.doc_encoder_input
    --d_MM1[2] : d_self.transform_ques
    local d_input_2=self.doc_encoder:backward( input[2]  , d_MM1[2]:reshape(batch_size , self.hidden_size )  )
    local d_self_doc_encoder_input=d_MM1[1]:add(d_MM2[1])
    input[1]=input[1]:reshape(batch_size*self.max_relations , self.seq_length)
    local d_input_1=self.fact_encoder:backward(input[1] , d_self_doc_encoder_input:reshape(batch_size*self.max_relations , self.hidden_size)):reshape(batch_size,self.max_relations,self.seq_length) 
    return {d_input_1,d_input_2}
end

