require 'rnn'
require 'nn'

--local img_graph_encoding={}
--img_graph_encoding.__index=img_graph_encoding

local utils=require 'misc.utils'
local LanguageEmbedding = require 'misc.LanguageEmbedding'

local layer , parent= torch.class('nn.img_fact_encoding','nn.Module')

function img_fact_encoding:__init( opt )
    parent.__init(self)

self.max_relations=utils.getopt(opt,'max_relations')
    self.vocab_size = utils.getopt(opt, 'vocab_size') -- required
    self.image_doc_size=utils.getopt(opt,'hidden_size') 
    self.hidden_size = utils.getopt(opt, 'hidden_size')
    local dropout = utils.getopt(opt, 'dropout', 0)
    self.seq_length = utils.getopt(opt, 'seq_length')
    self.atten_type = utils.getopt(opt, 'atten_type')
    self.feature_type = utils.getopt(opt, 'feature_type')
    self.LE = LanguageEmbedding.LE(self.vocab_size, self.hidden_size, self.hidden_size, self.seq_length)    
    --self.LT=nn.LookupTable(self.vocab_size,self.hidden_size)

    self.fact_encoder=nn.Sequential()
    self.fact_encoder:add(self.LE)
    self.fact_encoder.add(nn.SplitTable(1))
    self.fact_encoder:add(nn.Sequencer( nn.LSTM( self.hidden_size, self.hidden_size )  ))
    self.fact_encoder:add(nn.SelectTable(-1) )
    
    -- doc_encoder : to convert the relationship encodings into a single encoding using an LSTM
    self.doc_encoder=nn.Sequential()
    self.doc_encoder:add(nn.SplitTable(1))
    self.doc_encoder:add(nn.Sequencer(nn.LSTM( self.hidden_size , self.image_doc_size  )))
    self.doc_encoder:add(nn.SelectTable(-1))
end

function layer:getModulesList()
    return {self.LE, self.fact_encoder , self.doc_encoder}
end


function layer:parameters()
    local p1,g1=self.fact_encoder:getParameters()
    local p2,g2=self.doc_encoder:getParameters()
    local p3,g3=self.LE:getParameters() -- Check Whether The Lookup Tables' parameters are to be included
    
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
    local img_doc_relations=input[1]

    local batch_size=img_doc_relations:size(1)
    --self.relation_embeddings={}
    self.doc_encoder_input=torch.Tensor( batch_size, max_relations , self.hidden_size  ) 
    for i =1,batch_size do
        local fact_encoder_input=input[i]
        local encoded_facts=self.fact_encoder:forward(fact_encoder_input)  --encoded_facts:size max_relations x self.hidden_size
        local doc_encoder_input[i]=encoded_facts
    end

    local batch_doc_encoded=self.doc_encoder:forward(doc_encoder_input)
    return batch_doc_encoded
end

-- This Function Need Not Be Computed Right Now since the computation doc will help and do it automatically 
--function layer:updateGradInput( input , gradOutput  )
--    local img_doc_relations=input[1]
--    local batch_size=img_doc_relations:size(1)
--
--    local d_
--    
--end

