require 'rnn'
require 'nn'

--local img_graph_encoding={}
--img_graph_encoding.__index=img_graph_encoding

local utils=require 'misc.utils'
local LanguageEmbedding = require 'misc.LanguageEmbedding'

local layer , parent= torch.class('nn.img_fact_encoding','nn.Module')

function layer:__init( opt )
    parent.__init(self)
    self.max_relations=utils.getopt(opt,'max_relations')
    self.vocab_size = utils.getopt(opt, 'vocab_size') -- required
    --opt.hidden_size=1       -- Just for testing
    self.image_doc_size=utils.getopt(opt,'hidden_size') 
    self.hidden_size = utils.getopt(opt, 'hidden_size')
    local dropout = utils.getopt(opt, 'dropout', 0)
    self.seq_length = utils.getopt(opt, 'seq_length')
    self.atten_type = utils.getopt(opt, 'atten_type')
    self.feature_type = utils.getopt(opt, 'feature_type')
    self.LE = LanguageEmbedding.LE(self.vocab_size, self.hidden_size, self.hidden_size, self.seq_length)    
    --self.LE=nn.LookupTable(self.vocab_size,self.hidden_size)

    self.gpu=utils.getopt(opt,'gpu',true)
    self.fact_encoder=nn.Sequential()
    self.fact_encoder:add(self.LE)
    self.fact_encoder:add(nn.SplitTable(1,2))
    self.fact_encoder:add(nn.Sequencer(nn.LSTM( self.hidden_size, self.hidden_size )  ))
    self.fact_encoder:add(nn.SelectTable(-1) )
    
    -- doc_encoder : to convert the relationship encodings into a single encoding using an LSTM
    self.doc_encoder=nn.Sequential()
    self.doc_encoder:add(nn.SplitTable(1,2))
    self.doc_encoder:add(nn.Sequencer(nn.LSTM( self.hidden_size , self.image_doc_size  )))
    self.doc_encoder:add(nn.SelectTable(-1))
end

function layer:getModulesList()
    return {self.LE, self.fact_encoder , self.doc_encoder}
    --return {self.fact_encoder , self.doc_encoder}
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

function layer:updateOutput( input )
    --local img_doc_relations=input[1]
    --local img_doc_relations=input[1]          
    local batch_size=input[1]:size(1)
    --self.relation_embeddings={}
    --self.doc_encoder_input=torch.Tensor( batch_size, self.max_relations , self.hidden_size  ) 
    --if(self.gpu) then
    --    self.doc_encoder_input=self.doc_encoder_input:cuda()
    --end
   -- for i =1,batch_size do
   --     --local fact_encoder_input=img_doc_relations[i]
   --     local encoded_facts=self.fact_encoder:forward(img_doc_relations[i])  --encoded_facts:size max_relations x self.hidden_size
   --     self.doc_encoder_input[i]=encoded_facts
   -- end
    input[1]=input[1]:reshape(batch_size * self.max_relations,self.seq_length)
    self.doc_encoder_input=self.fact_encoder:forward(input[1]):reshape(batch_size , self.max_relations , self.hidden_size)
    print("self.doc_encoder_input:sum()")
    print(self.doc_encoder_input:sum())
    return self.doc_encoder:forward(self.doc_encoder_input)
end



-- This Function Need Not Be Computed Right Now since the computation doc will help and do it automatically 
function layer:updateGradInput( input , gradOutput  )
    -- gradOutput : batch_size x hidden_size
    --local img_doc_relations=input[1]
    local batch_size=input[1]:size(1)

    local d_doc_encoder_input=self.doc_encoder:backward( self.doc_encoder_input , gradOutput )
    -- d_doc_encoder_input : batch_size x max_relations x hidden_size
    d_doc_encoder_input=d_doc_encoder_input:reshape(batch_size * self.max_relations , self.hidden_size)
    input[1]=input[1]:reshape(batch_size*self.max_relations,self.seq_length)

    --local mask=input[1]:eq(0)
    --mask=1-mask
    --d_doc_encoder_input=d_doc_encoder_input:cmul(mask)
    return self.fact_encoder:backward( input[1] , d_doc_encoder_input ):reshape(batch_size,self.max_relations,self.seq_length)
end

