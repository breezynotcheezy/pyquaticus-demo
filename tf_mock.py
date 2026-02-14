"""
Mock TensorFlow module to handle compatibility issues with Ray RLlib.
This provides the minimal interface that Ray RLlib expects.
"""

class MockKeras:
    class layers:
        class Layer:
            def __init__(self, **kwargs):
                pass
            
            def __call__(self, inputs):
                return inputs

# Create mock module structure
keras = MockKeras()

# Mock basic dtypes that Ray RLlib expects
bool = 'bool'
uint8 = 'uint8'
int8 = 'int8'
int16 = 'int16'
int32 = 'int32'
int64 = 'int64'
float16 = 'float16'
float32 = 'float32'
float64 = 'float64'
complex64 = 'complex64'
complex128 = 'complex128'

# Mock other commonly used attributes
class MockTensor:
    def __init__(self, value):
        self.value = value
    
    def numpy(self):
        return self.value

def constant(value):
    return MockTensor(value)

def variable(value):
    return MockTensor(value)

# Add any other attributes that might be needed
__version__ = "2.10.0"
