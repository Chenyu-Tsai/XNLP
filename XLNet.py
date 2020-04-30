import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers.modeling_utils import (WEIGHTS_NAME, PretrainedConfig, PreTrainedModel,
                             SequenceSummary, PoolerAnswerClass, PoolerEndLogits, PoolerStartLogits)
from transformers import XLNetTokenizer, XLNetForSequenceClassification, XLNetPreTrainedModel, XLNetModel
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from transformers import get_linear_schedule_with_warmup

class XLNetForMultiSequenceClassification(XLNetPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = 3
        self.num_labels_3way = 3
        self.num_labels_multi = 5
        
        self.transformer = XLNetModel(config)
        self.sequence_summary = SequenceSummary(config)
        self.logits_proj_3way = nn.Linear(config.d_model, self.num_labels_3way)
        self.logits_proj_multi = nn.Linear(config.d_model, self.num_labels_multi)
        self.weights_3way = [1, 1.4, 3.3]
        self.weights_multi = [15, 10, 15, 5, 5]
        self.class_weights_3way = torch.FloatTensor(self.weights_3way).to(device)
        self.class_weights_multi = torch.FloatTensor(self.weights_multi).to(device)
        
        self.init_weights()
        

    def forward(self, input_ids, attention_mask=None, mems=None, perm_mask=None, target_mapping=None,
                token_type_ids=None, input_mask=None, head_mask=None, labels=None, inputs_embeds=None):
        transformer_outputs = self.transformer(input_ids,
                                               attention_mask=attention_mask,
                                               mems=mems,
                                               perm_mask=perm_mask,
                                               target_mapping=target_mapping,
                                               token_type_ids=token_type_ids,
                                               input_mask=input_mask, 
                                               head_mask=head_mask,
                                               inputs_embeds=inputs_embeds)
    
        output = transformer_outputs[0]
        output = self.sequence_summary(output)
        
        if labels is None:
            logits = self.logits_proj_3way(output)
            outputs = (logits,) + transformer_outputs[1:]

        if labels is not None:
            task_check = 0
        
            if labels.size() == torch.Size([1]):
                logits_3way = self.logits_proj_3way(output)
                outputs = (logits_3way,) + transformer_outputs[1:]
                task_check = 1
            else:
                logits_multi = self.logits_proj_multi(output)
                outputs = (logits_multi,) + transformer_outputs[1:]

            if task_check:
                loss_fct = CrossEntropyLoss(weight=self.class_weights_3way)
                loss = loss_fct(logits_3way.view(-1, self.num_labels_3way), labels.view(-1)).to(device)
            else:
                loss_fct = BCEWithLogitsLoss(pos_weight=self.class_weights_multi)
                loss = loss_fct(logits_multi.view(-1, self.num_labels_multi), labels).to(device)
            outputs = (loss,) + outputs
            
        return outputs