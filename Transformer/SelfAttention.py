import numpy as np

class SelfAttention:
    def __init__(self, embed_size):
        self.embed_size = embed_size
        self.W_q = np.random.randn(embed_size, embed_size)
        self.W_k = np.random.randn(embed_size, embed_size)
        self.W_v = np.random.randn(embed_size, embed_size)
        
    def forward(self, x):
        """
        x: input có dạng (N, T, embed_size)
        N: số lượng mẫu (batch size)
        T: chiều dài chuỗi (sequence length)
        embed_size: kích thước embedding
        """
        Q = np.dot(x, self.W_q)
        K = np.dot(x, self.W_k)
        V = np.dot(x, self.W_v)
        scores = np.matmul(Q, K.transpose(0, 2, 1))
        
        dk = self.embed_size
        scores = scores / np.sqrt(dk)
        attention_weights = self.softmax(scores)
        out = np.matmul(attention_weights, V)
        
        return out, attention_weights
    
    def softmax(self, x):
        """
        Áp dụng softmax theo trục cuối cùng
        """
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    
