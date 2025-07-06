# local imports
from llm_interface import LLMInterface
from metacognition import MetacognitionModule


# notes:
# user the role "user" as the "internal prompt-writer" and "assistant" as the "internal answerer"

class CognitiveEngine:
    """
    This is a custom cognitive engine that wraps over LLMs. In theory, it should enhance 
    the ability of LLMs and make them incredibly more flexible.
    """
    def __init__(self) -> None:
        self.llm = LLMInterface(
            model='gpt-4o-mini',
            max_tokens=5000,
            temperature=0.4)
        
        self.state = {}
        
        # instantiate cognitive modules
        self.metacognition = MetacognitionModule(llm=self.llm)
    
    def run(self):
        print('What can I help you with?')
        user_query = input()

        self.metacognition.run(user_query)
    


"""
Test:
We're going to build a fully autonomous quantitative researcher.
"""
ce = CognitiveEngine()
ce.run()