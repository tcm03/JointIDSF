import numpy as np
import torch
import torch.nn as nn
import logging
logger = logging.getLogger("tcm_logger")

class Attention(nn.Module):
    """Applies attention mechanism on the `context` using the `query`.
    Args:
        dimensions (int): Dimensionality of the query and context.
        attention_type (str, optional): How to compute the attention score:

            * dot: :math:`score(H_j,q) = H_j^T q`
            * general: :math:`score(H_j, q) = H_j^T W_a q`

    Example:

         >>> attention = Attention(256)
         >>> query = torch.randn(32, 50, 256)
         >>> context = torch.randn(32, 1, 256)
         >>> output, weights = attention(query, context)
         >>> output.size()
         torch.Size([32, 50, 256])
         >>> weights.size()
         torch.Size([32, 50, 1])
    """

    def __init__(self, dimensions):
        super(Attention, self).__init__()

        self.dimensions = dimensions
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
            output length: length of utterance
            query length: length of each token (1)
        Returns:
            :class:`tuple` with `output` and `weights`:
            * **output** (:class:`torch.LongTensor` [batch size, output length, dimensions]):
              Tensor containing the attended features.
            * **weights** (:class:`torch.FloatTensor` [batch size, output length, query length]):
              Tensor containing attention weights.
        """

        # @tcm
        # query: contextualized emb from pretrained LLM (ci)
        # context: w = Wp where p is probabilities vector from IntentClassifier (after softmax)

        # query = self.linear_query(query)

        batch_size, output_len, hidden_size = query.size()
        # query_len = context.size(1)

        # (batch_size, output_len, dimensions) * (batch_size, query_len, dimensions) ->
        # (batch_size, output_len, query_len)
        attention_scores = torch.bmm(query, context.transpose(1, 2).contiguous())
        # Compute weights across every context sequence
        # attention_scores = attention_scores.view(batch_size * output_len, query_len)
        if attention_mask is not None:
            # Create attention mask, apply attention mask before softmax
            attention_mask = torch.unsqueeze(attention_mask, 2)
            # attention_mask = attention_mask.view(batch_size * output_len, query_len)
            attention_scores.masked_fill_(attention_mask == 0, -np.inf)
        # attention_scores = torch.squeeze(attention_scores,1)
        attention_weights = self.softmax(attention_scores)
        # attention_weights = attention_weights.view(batch_size, output_len, query_len)

        # (batch_size, output_len, query_len) * (batch_size, query_len, dimensions) ->
        # (batch_size, output_len, dimensions)
        mix = torch.bmm(attention_weights, context)
        # from IPython import embed; embed()
        # concat -> (batch_size * output_len, 2*dimensions)
        combined = torch.cat((mix, query), dim=2)
        # combined = combined.view(batch_size * output_len, 2 * self.dimensions)

        # Apply linear_out on every 2nd dimension of concat
        # output -> (batch_size, output_len, dimensions)
        # output = self.linear_out(combined).view(batch_size, output_len, self.dimensions)
        output = self.linear_out(combined)

        output = self.tanh(output)
        # output = combined
        return output, attention_weights


class IntentClassifier(nn.Module):
    def __init__(self, input_dim, num_intent_labels, dropout_rate=0.0):
        super(IntentClassifier, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(input_dim, num_intent_labels)

    def forward(self, x):
        x = self.dropout(x)
        return self.linear(x)


class SlotClassifier(nn.Module):
    def __init__(
        self,
        input_dim,
        num_intent_labels,
        num_slot_labels,
        use_intent_context_concat=False,
        use_intent_context_attn=False,
        max_seq_len=50,
        attention_embedding_size=200,
        dropout_rate=0.0,
    ):
        super(SlotClassifier, self).__init__()
        self.use_intent_context_attn = use_intent_context_attn
        self.use_intent_context_concat = use_intent_context_concat
        self.max_seq_len = max_seq_len
        self.num_intent_labels = num_intent_labels
        self.num_slot_labels = num_slot_labels
        self.attention_embedding_size = attention_embedding_size

        output_dim = self.attention_embedding_size  # base model
        if self.use_intent_context_concat:
            output_dim = self.attention_embedding_size
            self.linear_out = nn.Linear(2 * attention_embedding_size, attention_embedding_size)

        elif self.use_intent_context_attn:
            output_dim = self.attention_embedding_size
            self.attention = Attention(attention_embedding_size)

        self.linear_slot = nn.Linear(input_dim, self.attention_embedding_size, bias=False)

        if self.use_intent_context_attn or self.use_intent_context_concat:
            # project intent vector and slot vector to have the same dimensions
            self.linear_intent_context = nn.Linear(self.num_intent_labels, self.attention_embedding_size, bias=False)
            self.softmax = nn.Softmax(dim=-1)  # softmax layer for intent logits

            # self.linear_out = nn.Linear(2 * intent_embedding_size, intent_embedding_size)
        # output
        self.dropout = nn.Dropout(dropout_rate)
        # @tcm: try replacing the ffn with LSTM+ffn for slot classification
        # self.linear = nn.Linear(output_dim, num_slot_labels)
        self.slot_lstm = nn.LSTM(
            input_size = output_dim,
            hidden_size = output_dim,
            num_layers = 1,
            bias = True,
            batch_first = True,
            dropout = 0,
            bidirectional = True
        )
        self.linear = nn.Linear(2 * output_dim, num_slot_labels)

    def forward(self, x, intent_context, attention_mask):
        # x: contextualized emb from pretrained LLM (ci)
        # intent_context: logits from IntentClassifier (pi)
        x = self.linear_slot(x)
        lstm_h0 = None
        lstm_c0 = None
        logger.info(f"@tcm: use_intent_context_concat: {self.use_intent_context_concat}")
        logger.info(f"@tcm: use_intent_context_attn: {self.use_intent_context_attn}")
        if self.use_intent_context_concat:
            intent_context = self.softmax(intent_context)
            intent_context = self.linear_intent_context(intent_context)
            logger.info(f"@tcm: intent_context: {intent_context}")
            lstm_h0 = intent_context.unsqueeze(0).expand(2, -1, -1).contiguous()
            lstm_c0 = intent_context.unsqueeze(0).expand(2, -1, -1).contiguous()
            intent_context = torch.unsqueeze(intent_context, 1)
            intent_context = intent_context.expand(-1, self.max_seq_len, -1)
            x = torch.cat((x, intent_context), dim=2)
            x = self.linear_out(x)

        elif self.use_intent_context_attn:
            intent_context = self.softmax(intent_context)
            intent_context = self.linear_intent_context(intent_context)
            logger.info(f"@tcm: intent_context: {intent_context}")
            lstm_h0 = intent_context.unsqueeze(0).expand(2, -1, -1).contiguous()
            lstm_c0 = intent_context.unsqueeze(0).expand(2, -1, -1).contiguous()
            intent_context = torch.unsqueeze(intent_context, 1)  # 1: query length (each token)
            output, weights = self.attention(x, intent_context, attention_mask)
            x = output
        x = self.dropout(x)
        # @tcm: try replacing the ffn with LSTM+ffn for slot classification
        # final_output = self.linear(x)
        logger.info(f"@tcm: lstm_h0: {lstm_h0}")
        logger.info(f"@tcm: lstm_c0: {lstm_c0}")
        final_output, _ = self.slot_lstm(x, (lstm_h0, lstm_c0))
        final_output = self.linear(final_output)
        
        return final_output
