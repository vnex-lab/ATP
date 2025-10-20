import numpy as np
try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    cp = np
    GPU_AVAILABLE = False

class TransformerChatbot:
    def __init__(self, vocab_size, embed_dim=256, num_heads=8, num_layers=4, 
                 ff_dim=1024, max_seq_len=50, learning_rate=0.001):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.ff_dim = ff_dim
        self.max_seq_len = max_seq_len
        self.learning_rate = learning_rate
        self.gpu_available = GPU_AVAILABLE
        self.xp = cp if GPU_AVAILABLE else np
        
        if self.embed_dim % self.num_heads != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})")
        
        self.head_dim = embed_dim // num_heads
        
        # BETTER initialization for training from scratch!
        # Embeddings: larger scale for stronger signals
        self.embedding = self.xp.random.randn(vocab_size, embed_dim) * 0.1
        self.pos_encoding = self._create_positional_encoding()
        
        self.encoder_layers = []
        self.decoder_layers = []
        for _ in range(num_layers):
            self.encoder_layers.append(self._init_encoder_layer())
            self.decoder_layers.append(self._init_decoder_layer())
        
        # Xavier initialization for output layer
        scale = self.xp.sqrt(2.0 / (embed_dim + vocab_size))
        self.output_weights = self.xp.random.randn(embed_dim, vocab_size) * scale
        self.output_bias = self.xp.zeros((1, vocab_size))
        
        self.training_history = {'loss': []}
        
        print(f"🤖 Transformer Model Initialized!")
        print(f"   GPU: {'✅ Enabled (CuPy)' if GPU_AVAILABLE else '❌ Disabled (NumPy)'}")
        print(f"   Vocab size: {vocab_size:,}")
        print(f"   Embed dim: {embed_dim}")
        print(f"   Attention heads: {num_heads}")
        print(f"   Layers: {num_layers}")
        print(f"   Feed-forward dim: {ff_dim}")
        
        total_params = self._count_parameters()
        print(f"   📊 Total parameters: {total_params:,}")
        if total_params >= 1_000_000_000:
            print(f"   🎉 BILLION PARAMETER MODEL! {total_params/1e9:.2f}B params!")
        elif total_params >= 1_000_000:
            print(f"   🚀 {total_params/1e6:.1f}M parameter model")
    
    def _create_positional_encoding(self):
        position = self.xp.arange(self.max_seq_len)[:, self.xp.newaxis]
        div_term = self.xp.exp(self.xp.arange(0, self.embed_dim, 2) * 
                               -(self.xp.log(10000.0) / self.embed_dim))
        
        pos_encoding = self.xp.zeros((self.max_seq_len, self.embed_dim))
        pos_encoding[:, 0::2] = self.xp.sin(position * div_term)
        pos_encoding[:, 1::2] = self.xp.cos(position * div_term)
        
        return pos_encoding
    
    def _init_encoder_layer(self):
        # Xavier initialization for better gradient flow
        scale = self.xp.sqrt(2.0 / self.embed_dim)
        scale_ff = self.xp.sqrt(2.0 / (self.embed_dim + self.ff_dim))
        
        layer = {
            'Wq': self.xp.random.randn(self.embed_dim, self.embed_dim) * scale,
            'Wk': self.xp.random.randn(self.embed_dim, self.embed_dim) * scale,
            'Wv': self.xp.random.randn(self.embed_dim, self.embed_dim) * scale,
            'Wo': self.xp.random.randn(self.embed_dim, self.embed_dim) * scale,
            'ff_w1': self.xp.random.randn(self.embed_dim, self.ff_dim) * scale_ff,
            'ff_b1': self.xp.zeros((1, self.ff_dim)),
            'ff_w2': self.xp.random.randn(self.ff_dim, self.embed_dim) * scale_ff,
            'ff_b2': self.xp.zeros((1, self.embed_dim)),
            'ln1_gamma': self.xp.ones((1, self.embed_dim)),
            'ln1_beta': self.xp.zeros((1, self.embed_dim)),
            'ln2_gamma': self.xp.ones((1, self.embed_dim)),
            'ln2_beta': self.xp.zeros((1, self.embed_dim)),
        }
        return layer
    
    def _init_decoder_layer(self):
        # Xavier initialization for better gradient flow
        scale = self.xp.sqrt(2.0 / self.embed_dim)
        scale_ff = self.xp.sqrt(2.0 / (self.embed_dim + self.ff_dim))
        
        layer = {
            'Wq_self': self.xp.random.randn(self.embed_dim, self.embed_dim) * scale,
            'Wk_self': self.xp.random.randn(self.embed_dim, self.embed_dim) * scale,
            'Wv_self': self.xp.random.randn(self.embed_dim, self.embed_dim) * scale,
            'Wo_self': self.xp.random.randn(self.embed_dim, self.embed_dim) * scale,
            'Wq_cross': self.xp.random.randn(self.embed_dim, self.embed_dim) * scale,
            'Wk_cross': self.xp.random.randn(self.embed_dim, self.embed_dim) * scale,
            'Wv_cross': self.xp.random.randn(self.embed_dim, self.embed_dim) * scale,
            'Wo_cross': self.xp.random.randn(self.embed_dim, self.embed_dim) * scale,
            'ff_w1': self.xp.random.randn(self.embed_dim, self.ff_dim) * scale_ff,
            'ff_b1': self.xp.zeros((1, self.ff_dim)),
            'ff_w2': self.xp.random.randn(self.ff_dim, self.embed_dim) * scale_ff,
            'ff_b2': self.xp.zeros((1, self.embed_dim)),
            'ln1_gamma': self.xp.ones((1, self.embed_dim)),
            'ln1_beta': self.xp.zeros((1, self.embed_dim)),
            'ln2_gamma': self.xp.ones((1, self.embed_dim)),
            'ln2_beta': self.xp.zeros((1, self.embed_dim)),
            'ln3_gamma': self.xp.ones((1, self.embed_dim)),
            'ln3_beta': self.xp.zeros((1, self.embed_dim)),
        }
        return layer
    
    def _count_parameters(self):
        count = self.embedding.size + self.output_weights.size + self.output_bias.size
        
        for layer in self.encoder_layers:
            for param in layer.values():
                count += param.size
        
        for layer in self.decoder_layers:
            for param in layer.values():
                count += param.size
        
        return count
    
    def _layer_norm(self, x, gamma, beta, epsilon=1e-5):
        mean = self.xp.mean(x, axis=-1, keepdims=True)
        var = self.xp.var(x, axis=-1, keepdims=True)
        x_norm = (x - mean) / self.xp.sqrt(var + epsilon)
        return gamma * x_norm + beta
    
    def _multi_head_attention(self, Q, K, V, Wq, Wk, Wv, Wo, mask=None):
        batch_size, q_seq_len, _ = Q.shape
        _, k_seq_len, _ = K.shape
        _, v_seq_len, _ = V.shape
        
        Q_proj = self.xp.dot(Q, Wq)
        K_proj = self.xp.dot(K, Wk)
        V_proj = self.xp.dot(V, Wv)
        
        Q_heads = Q_proj.reshape(batch_size, q_seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        K_heads = K_proj.reshape(batch_size, k_seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        V_heads = V_proj.reshape(batch_size, v_seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        
        scores = self.xp.matmul(Q_heads, K_heads.transpose(0, 1, 3, 2)) / self.xp.sqrt(self.head_dim)
        
        if mask is not None:
            scores = scores + mask
        
        attn_weights = self._softmax(scores, axis=-1)
        
        attn_output = self.xp.matmul(attn_weights, V_heads)
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(batch_size, q_seq_len, self.embed_dim)
        
        output = self.xp.dot(attn_output, Wo)
        
        return output, attn_weights
    
    def _feed_forward(self, x, w1, b1, w2, b2):
        hidden = self.xp.maximum(0, self.xp.dot(x, w1) + b1)
        output = self.xp.dot(hidden, w2) + b2
        return output
    
    def _softmax(self, x, axis=-1):
        exp_x = self.xp.exp(x - self.xp.max(x, axis=axis, keepdims=True))
        return exp_x / self.xp.sum(exp_x, axis=axis, keepdims=True)
    
    def _create_causal_mask(self, seq_len):
        mask = self.xp.triu(self.xp.ones((seq_len, seq_len)), k=1)
        mask = mask * -1e9
        return mask
    
    def encode(self, input_seq):
        batch_size, seq_len = input_seq.shape
        
        x = self.embedding[input_seq]
        x = x + self.pos_encoding[:seq_len]
        
        for layer in self.encoder_layers:
            attn_output, _ = self._multi_head_attention(
                x, x, x, 
                layer['Wq'], layer['Wk'], layer['Wv'], layer['Wo']
            )
            x = self._layer_norm(x + attn_output, layer['ln1_gamma'], layer['ln1_beta'])
            
            ff_output = self._feed_forward(
                x, layer['ff_w1'], layer['ff_b1'], layer['ff_w2'], layer['ff_b2']
            )
            x = self._layer_norm(x + ff_output, layer['ln2_gamma'], layer['ln2_beta'])
        
        return x
    
    def decode(self, target_seq, encoder_output):
        batch_size, target_len = target_seq.shape
        
        x = self.embedding[target_seq]
        x = x + self.pos_encoding[:target_len]
        
        causal_mask = self._create_causal_mask(target_len)
        
        for layer in self.decoder_layers:
            self_attn_output, _ = self._multi_head_attention(
                x, x, x,
                layer['Wq_self'], layer['Wk_self'], layer['Wv_self'], layer['Wo_self'],
                mask=causal_mask
            )
            x = self._layer_norm(x + self_attn_output, layer['ln1_gamma'], layer['ln1_beta'])
            
            cross_attn_output, _ = self._multi_head_attention(
                x, encoder_output, encoder_output,
                layer['Wq_cross'], layer['Wk_cross'], layer['Wv_cross'], layer['Wo_cross']
            )
            x = self._layer_norm(x + cross_attn_output, layer['ln2_gamma'], layer['ln2_beta'])
            
            ff_output = self._feed_forward(
                x, layer['ff_w1'], layer['ff_b1'], layer['ff_w2'], layer['ff_b2']
            )
            x = self._layer_norm(x + ff_output, layer['ln3_gamma'], layer['ln3_beta'])
        
        return x
    
    def forward(self, input_seq, target_seq):
        encoder_output = self.encode(input_seq)
        decoder_output = self.decode(target_seq, encoder_output)
        
        logits = self.xp.dot(decoder_output, self.output_weights) + self.output_bias
        
        return logits
    
    def train_batch(self, input_batch, target_batch, learning_rate=None):
        if learning_rate is None:
            learning_rate = self.learning_rate
        
        batch_size = len(input_batch)
        total_loss = 0.0
        
        for input_seq, target_seq in zip(input_batch, target_batch):
            input_seq = self.xp.array(input_seq).reshape(1, -1)
            target_input = self.xp.array(target_seq[:-1]).reshape(1, -1)
            target_output = self.xp.array(target_seq[1:]).reshape(1, -1)
            
            logits = self.forward(input_seq, target_input)
            
            probs = self._softmax(logits, axis=-1)
            
            loss = 0.0
            for t in range(target_output.shape[1]):
                correct_token = int(target_output[0, t])
                prob = float(probs[0, t, correct_token])
                loss += -self.xp.log(prob + 1e-10)
            
            loss /= target_output.shape[1]
            total_loss += float(loss)
            
            self._backward_and_update(input_seq, target_input, target_output, probs, learning_rate)
        
        return total_loss / batch_size
    
    def _backward_and_update(self, input_seq, target_input, target_output, probs, lr):
        batch_size, target_len, _ = probs.shape
        
        # Calculate loss gradient (cross-entropy)
        d_logits = probs.copy()
        for t in range(target_len):
            correct_token = int(target_output[0, t])
            d_logits[0, t, correct_token] -= 1.0
        
        d_logits /= target_len
        
        # Forward pass to get intermediate values
        encoder_output = self.encode(input_seq)
        decoder_output = self.decode(target_input, encoder_output)
        
        # Backprop through output layer
        d_output_weights = self.xp.dot(decoder_output.reshape(-1, self.embed_dim).T, 
                                       d_logits.reshape(-1, self.vocab_size))
        d_output_bias = self.xp.sum(d_logits.reshape(-1, self.vocab_size), axis=0, keepdims=True)
        
        # Update output layer
        self.output_weights -= lr * d_output_weights
        self.output_bias -= lr * d_output_bias
        
        # Gradient for decoder output
        d_decoder_output = self.xp.dot(d_logits, self.output_weights.T)
        
        # FIXED: Combine all gradients BEFORE updating embeddings!
        # This prevents conflicting updates to the same token
        
        d_embedding = self.xp.zeros_like(self.embedding)
        
        # Accumulate gradients from target tokens (decoder input)
        for t in range(target_input.shape[1]):
            token_idx = int(target_input[0, t])
            if token_idx < self.vocab_size:
                # Target tokens get direct gradient from decoder output
                d_embedding[token_idx] += d_decoder_output[0, t, :]
        
        # Accumulate gradients from input tokens (encoder input)
        # Get gradient signal from decoder - tells us what encoder should produce
        decoder_wants = self.xp.mean(d_decoder_output, axis=1)[0, :]
        
        for t in range(input_seq.shape[1]):
            token_idx = int(input_seq[0, t])
            if token_idx < self.vocab_size:
                # Input tokens get averaged feedback signal
                # Use 0.5 weight so input doesn't dominate over target
                d_embedding[token_idx] += 0.5 * decoder_wants
        
        # Single update with combined gradients - NO CONFLICTS!
        self.embedding -= lr * d_embedding
    
    def generate(self, input_text, tokenizer, max_length=50, temperature=1.0):
        input_tokens = tokenizer.encode(input_text)
        input_seq = self.xp.array(input_tokens).reshape(1, -1)
        
        encoder_output = self.encode(input_seq)
        
        generated = [tokenizer.word2idx['<START>']]
        
        for _ in range(max_length):
            target_seq = self.xp.array(generated).reshape(1, -1)
            decoder_output = self.decode(target_seq, encoder_output)
            
            logits = self.xp.dot(decoder_output[:, -1:, :], self.output_weights) + self.output_bias
            logits = logits.reshape(-1)
            
            if self.gpu_available:
                logits = cp.asnumpy(logits)
            
            probs = np.exp(logits / temperature)
            probs = probs / np.sum(probs)
            
            next_token = np.random.choice(len(probs), p=probs)
            
            if next_token == tokenizer.word2idx['<END>']:
                break
            
            generated.append(next_token)
        
        return tokenizer.decode(generated[1:])
    
    def save(self, filepath):
        model_data = {
            'vocab_size': self.vocab_size,
            'embed_dim': self.embed_dim,
            'num_heads': self.num_heads,
            'num_layers': self.num_layers,
            'ff_dim': self.ff_dim,
            'max_seq_len': self.max_seq_len,
            'learning_rate': self.learning_rate,
            'embedding': cp.asnumpy(self.embedding) if self.gpu_available else self.embedding,
            'output_weights': cp.asnumpy(self.output_weights) if self.gpu_available else self.output_weights,
            'output_bias': cp.asnumpy(self.output_bias) if self.gpu_available else self.output_bias,
            'encoder_layers': [],
            'decoder_layers': []
        }
        
        for layer in self.encoder_layers:
            layer_data = {}
            for key, param in layer.items():
                layer_data[key] = cp.asnumpy(param) if self.gpu_available else param
            model_data['encoder_layers'].append(layer_data)
        
        for layer in self.decoder_layers:
            layer_data = {}
            for key, param in layer.items():
                layer_data[key] = cp.asnumpy(param) if self.gpu_available else param
            model_data['decoder_layers'].append(layer_data)
        
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"✅ Transformer model saved to {filepath}")
    
    @staticmethod
    def load(filepath):
        import pickle
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        model = TransformerChatbot(
            vocab_size=model_data['vocab_size'],
            embed_dim=model_data['embed_dim'],
            num_heads=model_data['num_heads'],
            num_layers=model_data['num_layers'],
            ff_dim=model_data['ff_dim'],
            max_seq_len=model_data['max_seq_len'],
            learning_rate=model_data['learning_rate']
        )
        
        xp = model.xp
        model.embedding = xp.array(model_data['embedding'])
        model.output_weights = xp.array(model_data['output_weights'])
        model.output_bias = xp.array(model_data['output_bias'])
        
        for i, layer_data in enumerate(model_data['encoder_layers']):
            for key, param in layer_data.items():
                model.encoder_layers[i][key] = xp.array(param)
        
        for i, layer_data in enumerate(model_data['decoder_layers']):
            for key, param in layer_data.items():
                model.decoder_layers[i][key] = xp.array(param)
        
        print(f"✅ Transformer model loaded from {filepath}")
        return model
