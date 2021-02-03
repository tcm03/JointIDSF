import torch.nn as nn
import torch
import numpy as np

class Attention(nn.Module):
    """ Applies attention mechanism on the `context` using the `query`.

    **Thank you** to IBM for their initial implementation of :class:`Attention`. Here is
    their `License
    <https://github.com/IBM/pytorch-seq2seq/blob/master/LICENSE>`__.

    Args:
        dimensions (int): Dimensionality of the query and context.
        attention_type (str, optional): How to compute the attention score:

            * dot: :math:`score(H_j,q) = H_j^T q`
            * general: :math:`score(H_j, q) = H_j^T W_a q`

    Example:

         >>> attention = Attention(256)
         >>> query = torch.randn(5, 1, 256)
         >>> context = torch.randn(5, 5, 256)
         >>> output, weights = attention(query, context)
         >>> output.size()
         torch.Size([5, 1, 256])
         >>> weights.size()
         torch.Size([5, 1, 5])
    """

    def __init__(self, dimensions, attention_type='general'):
        super(Attention, self).__init__()

        if attention_type not in ['dot', 'general']:
            raise ValueError('Invalid attention type selected.')
        hidden_size = 768
        self.dimensions = dimensions
        self.attention_type = attention_type
        # self.linear_query = nn.Linear(hidden_size, dimensions)
        if self.attention_type == 'general':
            self.linear_in = nn.Linear(hidden_size, dimensions, bias=False)
        self.linear_out = nn.Linear(dimensions * 2, dimensions, bias=False)
        self.softmax = nn.Softmax(dim=1)
        self.tanh = nn.Tanh()

    def forward(self, query, context, attention_mask):
        """
        Args:
            query (:class:`torch.FloatTensor` [batch size, output length, dimensions]): Sequence of
                queries to query the context.
            context (:class:`torch.FloatTensor` [batch size, query length, dimensions]): Data
                overwhich to apply the attention mechanism.

        Returns:
            :class:`tuple` with `output` and `weights`:
            * **output** (:class:`torch.LongTensor` [batch size, output length, dimensions]):
              Tensor containing the attended features.
            * **weights** (:class:`torch.FloatTensor` [batch size, output length, query length]):
              Tensor containing attention weights.
        """
        # print(query.shape)
        # print(context.shape)
        # query = self.linear_query(query)
        
        batch_size, output_len, hidden_size = query.size()
        query_len = context.size(1)
        # print('query_len', query_len)
        if self.attention_type == "general":
            # print('query 0', query.shape)
            # query = query.reshape(batch_size * output_len, hidden_size)
            # print('query 1', query.shape)
            query = self.linear_in(query)
            # print('query 2', query.shape)
            # query = query.reshape(batch_size, output_len, self.dimensions)
            # print('query 3', query.shape)

        # (batch_size, output_len, dimensions) * (batch_size, query_len, dimensions) ->
        # (batch_size, output_len, query_len)
        attention_scores = torch.bmm(query, context.transpose(1, 2).contiguous())
        # Compute weights across every context sequence
        # attention_scores = attention_scores.view(batch_size * output_len, query_len)
        
        # Create attention mask, apply attention mask before softmax
        # attention_mask = torch.unsqueeze(attention_mask,2)
        # attention_mask = attention_mask.view(batch_size * output_len, query_len)
        # attention_scores.masked_fill_(attention_mask == 0, -np.inf)
        # attention_scores = torch.squeeze(attention_scores,1)
        attention_weights = self.softmax(attention_scores)
        # from IPython import embed; embed()
        # attention_weights = attention_weights.view(batch_size, output_len, query_len)

        # (batch_size, output_len, query_len) * (batch_size, query_len, dimensions) ->
        # (batch_size, output_len, dimensions)
        mix = torch.bmm(attention_weights, context)

        # concat -> (batch_size * output_len, 2*dimensions)
        combined = torch.cat((mix, query), dim=2)
        combined = combined.view(batch_size * output_len, 2 * self.dimensions)

        # Apply linear_out on every 2nd dimension of concat
        # output -> (batch_size, output_len, dimensions)
        output = self.linear_out(combined).view(batch_size, output_len, self.dimensions)
        output = self.tanh(output)

        return output, attention_weights

class IntentClassifier(nn.Module):
    def __init__(self, input_dim, num_intent_labels, dropout_rate=0.):
        super(IntentClassifier, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(input_dim, num_intent_labels)

    def forward(self, x):
        x = self.dropout(x)
        # print('intent', x.shape)
        # print('intent', self.linear(x).shape)
        return self.linear(x)

class SlotClassifier(nn.Module):
    def __init__(self, input_dim, num_intent_labels, num_slot_labels, use_intent_context_concat = False, use_intent_context_attn = False, max_seq_len = 50, intent_embedding_size = 22, attention_embedding_size = 768, attention_type = 'general', dropout_rate=0.):
        super(SlotClassifier, self).__init__()
        self.use_intent_context_attn = use_intent_context_attn
        self.use_intent_context_concat = use_intent_context_concat
        self.max_seq_len = max_seq_len
        self.num_intent_labels = num_intent_labels
        self.num_slot_labels = num_slot_labels
        self.intent_embedding_size = intent_embedding_size
        self.attention_embedding_size = attention_embedding_size
        self.attention_type = attention_type
        # print('attention_type', self.attention_type)
        
        output_dim = input_dim #base model
        if self.use_intent_context_concat:
            output_dim = self.intent_embedding_size * 2
        elif self.use_intent_context_attn:
            output_dim = self.attention_embedding_size
            self.intent_embedding_size = self.attention_embedding_size

        self.softmax = nn.Softmax(dim = -1) #softmax layer for intent logits
        
        self.attention = Attention(attention_embedding_size, self.attention_type)
        
        #project intent vector and slot vector to have the same dimensions
        self.linear_intent_context = nn.Linear(self.num_intent_labels, self.intent_embedding_size, bias = False)
        self.linear_slot = nn.Linear(input_dim, self.intent_embedding_size, bias=False)
        #output
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(output_dim, num_slot_labels)

    def forward(self, x, intent_context, attention_mask):
        if self.use_intent_context_concat:
            intent_context = self.softmax(intent_context)
            intent_context = self.linear_intent_context(intent_context)
            intent_context = torch.unsqueeze(intent_context, 1)
            intent_context = intent_context.expand(-1, self.max_seq_len, -1)
            # print(x.shape)
            x = self.linear_slot(x)
            hidden_size = x.shape[2]
            x = nn.ConstantPad1d((0,self.intent_embedding_size), 1)(x)
            x[:,:,hidden_size:] = intent_context
        
        elif self.use_intent_context_attn:
            intent_context = self.softmax(intent_context)
            intent_context = self.linear_intent_context(intent_context)
            intent_context = torch.unsqueeze(intent_context, 1) #1: query length (each token)
            # intent_context = intent_context.expand(-1, self.max_seq_len, -1)
            output, weights = self.attention(x, intent_context, attention_mask)
            x = output
        x = self.dropout(x)
        return self.linear(x)