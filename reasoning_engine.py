class BaseReasoningEngine:
    """
    Base engine follows these steps:
    1) Make a plan
    2) Check if steps require function calls
    3) Perform the action
    4) Verify completion
    """
    def __init__(self, llm) -> None:
        self.llm = llm
        self.plan_prompt_builder = DeepSeekPromptBuilder
        self.function_call_prompt_builder = DeepSeekPromptBuilder
        self.verify_prompt_builder = DeepSeekPromptBuilder
    
    def generate_plan(self, thoughts):
        prompt = self.plan_prompt_builder(
            system_message="You are a planning agent in a cognitive engine. Your task is to generate a step-by-step plan to solve a task. Each step must be atomic, in the sense that the step can be solved with one action. You must also "
        )
        prompt.add_user(message=thoughts)
        prompt.add_assistant(use_think=True)
        return self.llm.query(prompt.build())
