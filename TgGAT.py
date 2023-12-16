import torch
import torch.nn as nn
import torch.nn.functional as F


# Class TgGAT ini sudah menerapkan pembentukan attention score yang diconcat dengan topic representation

class GATLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat=False):
        super(GATLayer, self).__init__()
        self.dropout = dropout
        # in_features =  banyaknya fitur setiap node misal 768
        self.in_features = in_features
        # out_features = jumlah fitur yang dihasilkan
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        # Adjust the size of a for compatibility with multiplication
        self.a = nn.Parameter(torch.zeros(size=(4 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj, topic_representation):

        # W*h, input =  node, fitur merupakan hasil dari acnhor aware graph
        h = torch.mm(input, self.W)
        # jumlah node
        N = h.size()[0]
        

        # Attention Mechanism with topic representation
        # [whi||whj||tpi||tpj]
        a_input_with_topic = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1), topic_representation.repeat(
            1, N).view(N * N, -1), topic_representation.repeat(N, 1)], dim=1).view(N, -1, 4 * self.out_features)
      
        # LeakyReLU(a[whi||whj||tpi||tpj])
        e_with_topic = self.leakyrelu(torch.matmul(
            a_input_with_topic, self.a).squeeze(2))
       
        # membuat vektor dengan seluruh nilainya -9e15 dengan ukuran yang sama dengan e_with_topic
        zero_vec_with_topic = -9e15 * torch.ones_like(e_with_topic)
        # nilai attention didapatkan dari zero_vec_with_topic dan e_with_topic, nilai nya e_with_topic jika lebih besar dari 0
        attention_with_topic = torch.where(
            adj > 0, e_with_topic, zero_vec_with_topic)

        attention_with_topic = F.softmax(attention_with_topic, dim=1)
        attention_with_topic = F.dropout(attention_with_topic, self.dropout)
        # attention x hi
        h_prime_with_topic = torch.matmul(attention_with_topic, h)
      
        if self.concat:
            return F.elu(h_prime_with_topic)
        else:
            return h_prime_with_topic


class TgGATLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, num_heads, concat=True):
        super(TgGATLayer, self).__init__()
        self.heads = nn.ModuleList([
            GATLayer(in_features, out_features, dropout, alpha, concat) for _ in range(num_heads)
        ])

    def forward(self, input, adj, topic_representation):
        head_outputs = [head(input, adj, topic_representation)
                        for head in self.heads]
        aggregated = torch.stack(head_outputs, dim=2).mean(dim=2)
        # Add the topic guidance
        aggregated += topic_representation.squeeze(1).expand_as(aggregated)
        return aggregated



