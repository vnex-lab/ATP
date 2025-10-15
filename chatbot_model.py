import json
import pickle
from typing import List, Dict, Tuple, Optional

# GPU Support - Try CuPy first, fallback to NumPy
try:
    import cupy as cp
    # Test if GPU is actually usable
    try:
        _ = cp.array([1, 2, 3])
        np = cp
        GPU_AVAILABLE = True
        DEVICE = "GPU"
    except Exception:
        # CuPy installed but GPU not usable (driver issues, etc.)
        import numpy as np
        GPU_AVAILABLE = False
        DEVICE = "CPU"
except ImportError:
    # CuPy not installed
    import numpy as np
    GPU_AVAILABLE = False
    DEVICE = "CPU"

class VnexAIChatbot:
    """
    VnexAI Chatbot - A simple sequence-to-sequence chatbot built from scratch
    """
    
    def __init__(self, vocab_size: int, embedding_dim: int = 128, hidden_dim: int = 256, 
                 max_length: int = 50, learning_rate: float = 0.01):
        """
        Initialize the VnexAI Chatbot
        
        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Dimension of word embeddings
            hidden_dim: Dimension of hidden state
            max_length: Maximum sequence length
            learning_rate: Learning rate for training
        """
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.max_length = max_length
        self.learning_rate = learning_rate
        self.device = DEVICE
        self.gpu_available = GPU_AVAILABLE
        
        # Initialize weights
        self._initialize_weights()
        
        # Training history
        self.training_history = {
            'loss': [],
            'perplexity': []
        }
    
    def get_device_info(self) -> str:
        """Get information about the device being used"""
        if self.gpu_available:
            return f"🚀 GPU Acceleration ACTIVE ({np.__name__})"
        else:
            return f"💻 CPU Mode ({np.__name__})"
    
    def _initialize_weights(self):
        """Initialize all model weights"""
        global np, GPU_AVAILABLE, DEVICE
        
        try:
            np.random.seed(42)
            
            # Embedding layer
            self.embedding = np.random.randn(self.vocab_size, self.embedding_dim) * 0.01
            
            # Encoder RNN weights
            self.Wxh_enc = np.random.randn(self.embedding_dim, self.hidden_dim) * 0.01
            self.Whh_enc = np.random.randn(self.hidden_dim, self.hidden_dim) * 0.01
            self.bh_enc = np.zeros((1, self.hidden_dim))
            
            # Decoder RNN weights
            self.Wxh_dec = np.random.randn(self.embedding_dim, self.hidden_dim) * 0.01
            self.Whh_dec = np.random.randn(self.hidden_dim, self.hidden_dim) * 0.01
            self.bh_dec = np.zeros((1, self.hidden_dim))
            
            # Output layer
            self.Why = np.random.randn(self.hidden_dim, self.vocab_size) * 0.01
            self.by = np.zeros((1, self.vocab_size))
            
        except Exception as e:
            # GPU failed during initialization, fall back to CPU
            if GPU_AVAILABLE:
                import numpy as numpy_fallback
                np = numpy_fallback
                GPU_AVAILABLE = False
                DEVICE = "CPU"
                self.gpu_available = False
                self.device = "CPU"
                
                # Retry initialization with NumPy
                np.random.seed(42)
                self.embedding = np.random.randn(self.vocab_size, self.embedding_dim) * 0.01
                self.Wxh_enc = np.random.randn(self.embedding_dim, self.hidden_dim) * 0.01
                self.Whh_enc = np.random.randn(self.hidden_dim, self.hidden_dim) * 0.01
                self.bh_enc = np.zeros((1, self.hidden_dim))
                self.Wxh_dec = np.random.randn(self.embedding_dim, self.hidden_dim) * 0.01
                self.Whh_dec = np.random.randn(self.hidden_dim, self.hidden_dim) * 0.01
                self.bh_dec = np.zeros((1, self.hidden_dim))
                self.Why = np.random.randn(self.hidden_dim, self.vocab_size) * 0.01
                self.by = np.zeros((1, self.vocab_size))
            else:
                raise e
    
    def _sigmoid(self, x):
        """Sigmoid activation"""
        return 1 / (1 + np.exp(-np.clip(x, -250, 250)))
    
    def _tanh(self, x):
        """Tanh activation"""
        return np.tanh(x)
    
    def _softmax(self, x):
        """Softmax activation"""
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    def encode(self, input_seq: np.ndarray) -> np.ndarray:
        """
        Encode input sequence
        
        Args:
            input_seq: Input token indices [seq_len]
        
        Returns:
            Final hidden state [hidden_dim]
        """
        h = np.zeros((1, self.hidden_dim))
        
        for t in range(len(input_seq)):
            if input_seq[t] < self.vocab_size:
                x_t = self.embedding[input_seq[t]].reshape(1, -1)
                h = self._tanh(np.dot(x_t, self.Wxh_enc) + np.dot(h, self.Whh_enc) + self.bh_enc)
        
        return h
    
    def decode(self, hidden_state: np.ndarray, target_seq: Optional[np.ndarray] = None, 
               generate: bool = False) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Decode from hidden state
        
        Args:
            hidden_state: Initial hidden state from encoder
            target_seq: Target sequence for training (optional)
            generate: Whether to generate or use teacher forcing
        
        Returns:
            outputs: Output probabilities for each timestep
            hidden_states: Hidden states for each timestep
        """
        h = hidden_state
        outputs = []
        hidden_states = [h]
        
        # Start token (0) or first target token
        if target_seq is not None and not generate:
            seq_len = len(target_seq)
            for t in range(seq_len):
                x_t = self.embedding[target_seq[t]].reshape(1, -1)
                h = self._tanh(np.dot(x_t, self.Wxh_dec) + np.dot(h, self.Whh_dec) + self.bh_dec)
                y = np.dot(h, self.Why) + self.by
                output = self._softmax(y)
                outputs.append(output)
                hidden_states.append(h)
        else:
            # Generation mode
            x_t = self.embedding[0].reshape(1, -1)  # Start with <START> token
            for t in range(self.max_length):
                h = self._tanh(np.dot(x_t, self.Wxh_dec) + np.dot(h, self.Whh_dec) + self.bh_dec)
                y = np.dot(h, self.Why) + self.by
                output = self._softmax(y)
                outputs.append(output)
                hidden_states.append(h)
                
                # Sample next token
                next_token = np.argmax(output)
                if next_token == 1:  # <END> token
                    break
                x_t = self.embedding[next_token].reshape(1, -1)
        
        return np.array(outputs), hidden_states
    
    def forward(self, input_seq: np.ndarray, target_seq: np.ndarray) -> Tuple[np.ndarray, np.ndarray, List]:
        """
        Forward pass
        
        Args:
            input_seq: Input sequence [seq_len]
            target_seq: Target sequence [seq_len]
        
        Returns:
            outputs: Predicted probabilities
            encoder_hidden: Encoder hidden state
            decoder_hidden_states: Decoder hidden states
        """
        encoder_hidden = self.encode(input_seq)
        outputs, decoder_hidden_states = self.decode(encoder_hidden, target_seq, generate=False)
        return outputs, encoder_hidden, decoder_hidden_states
    
    def compute_loss(self, outputs: np.ndarray, targets: np.ndarray) -> float:
        """
        Compute cross-entropy loss
        
        Args:
            outputs: Predicted probabilities [seq_len, vocab_size]
            targets: Target token indices [seq_len]
        
        Returns:
            loss: Average cross-entropy loss
        """
        loss = 0
        for t in range(len(targets)):
            if t < len(outputs):
                loss += -np.log(outputs[t, 0, targets[t]] + 1e-10)
        return loss / len(targets)
    
    def backward(self, input_seq: np.ndarray, target_seq: np.ndarray, 
                outputs: np.ndarray, encoder_hidden: np.ndarray, 
                decoder_hidden_states: List) -> Dict[str, np.ndarray]:
        """
        Backward pass using simplified gradient computation
        
        Returns:
            gradients: Dictionary of gradients
        """
        # Initialize gradients
        dWxh_enc = np.zeros_like(self.Wxh_enc)
        dWhh_enc = np.zeros_like(self.Whh_enc)
        dbh_enc = np.zeros_like(self.bh_enc)
        
        dWxh_dec = np.zeros_like(self.Wxh_dec)
        dWhh_dec = np.zeros_like(self.Whh_dec)
        dbh_dec = np.zeros_like(self.bh_dec)
        
        dWhy = np.zeros_like(self.Why)
        dby = np.zeros_like(self.by)
        dembedding = np.zeros_like(self.embedding)
        
        # Backward through decoder
        dh_next = np.zeros((1, self.hidden_dim))
        
        for t in reversed(range(len(target_seq))):
            if t < len(outputs):
                # Output layer gradient
                dy = outputs[t].copy()
                dy[0, target_seq[t]] -= 1  # Softmax + cross-entropy gradient
                
                dWhy += np.dot(decoder_hidden_states[t].T, dy)
                dby += dy
                
                # Hidden state gradient
                dh = np.dot(dy, self.Why.T) + dh_next
                dh_raw = dh * (1 - decoder_hidden_states[t] ** 2)  # tanh derivative
                
                # Decoder RNN gradients
                x_t = self.embedding[target_seq[t]].reshape(1, -1)
                dWxh_dec += np.dot(x_t.T, dh_raw)
                if t > 0:
                    dWhh_dec += np.dot(decoder_hidden_states[t-1].T, dh_raw)
                dbh_dec += dh_raw
                
                # Embedding gradient
                dembedding[target_seq[t]] += np.dot(dh_raw, self.Wxh_dec.T).flatten()
                
                dh_next = np.dot(dh_raw, self.Whh_dec.T)
        
        # Simplified encoder gradient (only propagate from final state)
        dh_enc = dh_next
        encoder_states = []
        h = np.zeros((1, self.hidden_dim))
        for t in range(len(input_seq)):
            if input_seq[t] < self.vocab_size:
                x_t = self.embedding[input_seq[t]].reshape(1, -1)
                h = self._tanh(np.dot(x_t, self.Wxh_enc) + np.dot(h, self.Whh_enc) + self.bh_enc)
                encoder_states.append(h)
        
        for t in reversed(range(len(input_seq))):
            if input_seq[t] < self.vocab_size and t < len(encoder_states):
                dh_raw = dh_enc * (1 - encoder_states[t] ** 2)
                x_t = self.embedding[input_seq[t]].reshape(1, -1)
                dWxh_enc += np.dot(x_t.T, dh_raw)
                if t > 0:
                    dWhh_enc += np.dot(encoder_states[t-1].T, dh_raw)
                dbh_enc += dh_raw
                
                dembedding[input_seq[t]] += np.dot(dh_raw, self.Wxh_enc.T).flatten()
                dh_enc = np.dot(dh_raw, self.Whh_enc.T)
        
        # Clip gradients
        for grad in [dWxh_enc, dWhh_enc, dWxh_dec, dWhh_dec, dWhy, dembedding]:
            np.clip(grad, -5, 5, out=grad)
        
        return {
            'dWxh_enc': dWxh_enc,
            'dWhh_enc': dWhh_enc,
            'dbh_enc': dbh_enc,
            'dWxh_dec': dWxh_dec,
            'dWhh_dec': dWhh_dec,
            'dbh_dec': dbh_dec,
            'dWhy': dWhy,
            'dby': dby,
            'dembedding': dembedding
        }
    
    def update_weights(self, gradients: Dict[str, np.ndarray]):
        """Update weights using gradients"""
        self.Wxh_enc -= self.learning_rate * gradients['dWxh_enc']
        self.Whh_enc -= self.learning_rate * gradients['dWhh_enc']
        self.bh_enc -= self.learning_rate * gradients['dbh_enc']
        
        self.Wxh_dec -= self.learning_rate * gradients['dWxh_dec']
        self.Whh_dec -= self.learning_rate * gradients['dWhh_dec']
        self.bh_dec -= self.learning_rate * gradients['dbh_dec']
        
        self.Why -= self.learning_rate * gradients['dWhy']
        self.by -= self.learning_rate * gradients['dby']
        
        self.embedding -= self.learning_rate * gradients['dembedding']
    
    def train_step(self, input_seq: np.ndarray, target_seq: np.ndarray) -> float:
        """Single training step"""
        outputs, encoder_hidden, decoder_hidden_states = self.forward(input_seq, target_seq)
        loss = self.compute_loss(outputs, target_seq)
        gradients = self.backward(input_seq, target_seq, outputs, encoder_hidden, decoder_hidden_states)
        self.update_weights(gradients)
        return loss
    
    def generate_response(self, input_seq: np.ndarray) -> np.ndarray:
        """Generate a response for given input"""
        encoder_hidden = self.encode(input_seq)
        outputs, _ = self.decode(encoder_hidden, generate=True)
        
        # Get tokens from outputs
        response_tokens = []
        for output in outputs:
            token = np.argmax(output)
            if token == 1:  # <END> token
                break
            response_tokens.append(token)
        
        return np.array(response_tokens)
    
    def _to_cpu(self, array):
        """Convert array to CPU (NumPy) for saving"""
        if GPU_AVAILABLE and hasattr(array, 'get'):
            return array.get()  # CuPy to NumPy
        return array
    
    def _to_device(self, array):
        """Convert array to current device"""
        if GPU_AVAILABLE:
            import cupy as cp
            if not isinstance(array, cp.ndarray):
                return cp.array(array)
        return array
    
    def save_model(self, filepath: str):
        """Save model to binary file"""
        model_data = {
            'vocab_size': self.vocab_size,
            'embedding_dim': self.embedding_dim,
            'hidden_dim': self.hidden_dim,
            'max_length': self.max_length,
            'learning_rate': self.learning_rate,
            'embedding': self._to_cpu(self.embedding),
            'Wxh_enc': self._to_cpu(self.Wxh_enc),
            'Whh_enc': self._to_cpu(self.Whh_enc),
            'bh_enc': self._to_cpu(self.bh_enc),
            'Wxh_dec': self._to_cpu(self.Wxh_dec),
            'Whh_dec': self._to_cpu(self.Whh_dec),
            'bh_dec': self._to_cpu(self.bh_dec),
            'Why': self._to_cpu(self.Why),
            'by': self._to_cpu(self.by),
            'training_history': self.training_history
        }
        
        # Save as binary file
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_model(self, filepath: str):
        """Load model from binary file"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.vocab_size = model_data['vocab_size']
        self.embedding_dim = model_data['embedding_dim']
        self.hidden_dim = model_data['hidden_dim']
        self.max_length = model_data['max_length']
        self.learning_rate = model_data['learning_rate']
        
        # Convert to appropriate device (GPU if available)
        self.embedding = self._to_device(model_data['embedding'])
        self.Wxh_enc = self._to_device(model_data['Wxh_enc'])
        self.Whh_enc = self._to_device(model_data['Whh_enc'])
        self.bh_enc = self._to_device(model_data['bh_enc'])
        self.Wxh_dec = self._to_device(model_data['Wxh_dec'])
        self.Whh_dec = self._to_device(model_data['Whh_dec'])
        self.bh_dec = self._to_device(model_data['bh_dec'])
        self.Why = self._to_device(model_data['Why'])
        self.by = self._to_device(model_data['by'])
        self.training_history = model_data['training_history']
