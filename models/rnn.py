import torch


class GRU(torch.nn.Module):
    def __init__(self, num_embeddings, embedding_dim, hidden_size):
        super().__init__()
        self.embed = torch.nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim)
        self.rnn = torch.nn.GRU(input_size=embedding_dim, hidden_size=hidden_size)
        self.linear = torch.nn.Linear(in_features=64, out_features=1)

    def forward(self, input_ids, seq_len):
        embeds = self.embed(input_ids)
        embeds = embeds.permute(1, 0, 2)
        all_outputs, _ = self.rnn(embeds)
        all_outputs = all_outputs.permute(1, 0, 2)
        embeds = all_outputs[torch.arange(0, end=embeds.shape[1], step=1), seq_len]
        embeds = embeds.squeeze(dim=1)
        embeds = self.linear(embeds)
        return embeds