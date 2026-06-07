#import required libraries


class LLM_Model:
    def __init__(self):
        """
        Initialize LLM Model
        """
        pass

    def _check_gpu_availability(self):
        """
        Check GPU availability
        """
        pass


    def _load_model(self, device):
        """
        Load Model using Hugging Transformers Library to GPU
        """
        pass

    def _quantize_model(self, model):
        """
        Quantize Model to either 4 or 8 Bits
        """
        pass
    
    def generate_response(self):
        """
        Generate Clean Model Response
        """
        pass

    def get_model(self):
        """
        Get loaded model and all related properties
        """
        pass