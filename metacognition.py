import json

# You must be able to explain what is being asked of you, what are any constraints, and what is the expected output.
class TaskInterpreter:
    def __init__(self, llm):
        self.LLM = llm

        self.TASK_INTERPRETER_PROMPT = """
        I am the metacognition part of my brain. My job is to accurately interpret a task that has been given to me.
        I should ask clarifying questions to help me interpret the task and resolve any ambiquities.
        I should ask one question at a time and avoid repeating myself. I will not ask about dealines.
        I respond in a concise manner.
        """
        # Respond using the following JSON schema:
        # {
        #     "nextMessage: "Your internal thoughts and/or reply to the user.",
        #     "interpretationComplete: True or False
        # }

        self.VERIFICATION_THOUGHTS = """
        You are the metacognition part of a brain focused on interpreting tasks.
        To do this, you must think about the task deeply. You will be given a conversation you
        had with the user where you attempted to refine your understanding of the task. Your job 
        is to determine if you
        """

        self.INTERPRETATION_VERIFICATION = """
        You are the metacognition part of a brain. You will be given the summary of a conversation you have had with a user
        where you attempted to clarify your interpretation of what the user has asked of you. Your current job is to verify
        that your interpretation is sufficient and that no refinements are left to be made.
        """

    def interpretation_of_user_query(self, user_query):
        """
        Aim to "understand" the problem.
        1) Converse with user to refine the problem
        2) Summarize conversation
        """
        # 1)
        json_schema = {
            "name": "initial_interpretation_step",
            "schema": {
                "type": "object",
                "properties": {
                    "nextMessage": {
                        "type": "string",
                        "description": "The LLM's next message."
                    },
                    "interpretationComplete": {
                        "type": "boolean",
                        "description": "True if all information has been gathered."
                    }
                },
                "required": ["nextMessage", "interpretationComplete"],
                "additionalProperties": False
            },
            "strict": True
        }

        messages = [
            {'role': 'system', 'content': self.TASK_INTERPRETER_PROMPT},
            {'role': 'user', 'content': user_query}
        ]

        while True:
            response = self.LLM.query(messages, json_schema=json_schema)
            print('\nLLM:')
            
            messages.append({ 'role': 'assistant', 'content': response['nextMessage'] })

            if response['interpretationComplete']:
                break
            
            print('-'*10)
            print(response['nextMessage'])
            print('\nUser Input:')
            print('-'*10)
            user_input = input()

            messages.append({ 'role': 'user', 'content': user_input })
        
       
        summarization_prompt = (
            "I am the metacognition part of my brain. "
            "I am given a conversation I had with a user where I aimed at interpreting their request. "
            "My task is to summarize my interpretation of the task given my conversation with the user."
        )
        
        conversation = "".join([f"{row['role']}:\n{row['content']}\n" if row['role'] != 'system' else "" for row in messages])

        messages = [
            { 'role': 'system', 'content': summarization_prompt},
            {'role': 'user', 'content': f'Conversation: \n {conversation}'}
        ]

        summary = self.LLM.query(messages)
        
        return summary
        
 

class Planning:
    def __init__(self, llm, max_refinement_iter):
        self.LLM = llm
        self.base_prompt_path = "./prompts/planning"
        self.MAX_REFINMENT_ITER = max_refinement_iter
        self.load_system_prompts()
        
    def load_system_prompts(self):
        with open(f'{self.base_prompt_path}/initial_plan_generator.txt', 'r') as f:
            # need to test the plan with an example
            self.higherarchical_planner_system_prompt = f.read()
        
        with open(f'{self.base_prompt_path}/plan_refinement.txt', 'r') as f:
            self.plan_refinement_system_prompt = f.read()
    
    def plan_refinement_loop(self, plan, refinement_iter_count=0):
        if refinement_iter_count >= self.MAX_REFINMENT_ITER:
            # RECURSION MAX DEPTH HIT
            print("HIT MAX REFINEMENT ITERATION")
            return plan

        plan_as_str = json.dumps(plan, indent=2)

        json_schema = {
            "name": "hierarchical_planning_step",
            "schema": {
                "type": "object",
                "properties": {
                    "plan": {
                        "type": "array",
                        "description": "A list of high-level tasks, each with associated atomic subtasks.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "task": {
                                    "type": "string",
                                    "description": "A high-level step in the plan."
                                },
                                "subtasks": {
                                    "type": "array",
                                    "description": "A sequential list of atomic steps required to complete the task.",
                                    "items": {
                                        "type": "string"
                                    },
                                    "minItems": 1
                                }
                            },
                            "required": ["task", "subtasks"],
                            "additionalProperties": False
                        },
                        "minItems": 1
                    },
                    "refinementCompletion": {
                        "type": "boolean",
                        "description": "True if the plan does NOT need any further refinements."
                    }
                },
                "required": ["plan", "refinementCompletion"],
                "additionalProperties": False
            },
            "strict": True
        }

        messages = [
            {'role': 'system', 'content': self.plan_refinement_system_prompt},
            {'role': 'assistant', 'content': f'Here is the plan I generated:\n```json\n{plan_as_str}'},
            {'role': 'user', 'content': f'{"This is my last refinement step so let me think carefully. " if refinement_iter_count == self.MAX_REFINMENT_ITER - 1 else ""}Are there any other refinements I should make to the plan I generated?' }
        ]


        response = self.LLM.query(messages, json_schema)
        if response['refinementCompletion']:
            return response['plan']
        
        # clean dictionary -> recurse
        del response['refinementCompletion']
        return self.plan_refinement_loop(response, refinement_iter_count + 1)


    def generate_plan(self, task_interpretation):
        """
        This generates the initial plan which will then be revised.
        """
        json_schema = {
            "name": "hierarchical_planning_step",
            "schema": {
                "type": "object",
                "properties": {
                    "plan": {
                        "type": "array",
                        "description": "A list of high-level tasks, each with associated atomic subtasks.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "task": {
                                    "type": "string",
                                    "description": "A high-level step in the plan."
                                },
                                "subtasks": {
                                    "type": "array",
                                    "description": "A sequential list of atomic steps required to complete the task.",
                                    "items": {
                                        "type": "string"
                                    },
                                    "minItems": 1
                                }
                            },
                            "required": ["task", "subtasks"],
                            "additionalProperties": False
                        },
                        "minItems": 1
                    }
                },
                "required": ["plan"],
                "additionalProperties": False
            },
            "strict": True
        }

        messages = [
            {'role': 'system', 'content': self.higherarchical_planner_system_prompt},
            {'role': 'assistant', 'content': f"Interpreted task after conversing with the user: {task_interpretation}"},
            {'role': 'user', 'content': 'Generate a plan.'}
        ]

        initial_plan = self.LLM.query(messages, json_schema)
        print("\nInitial Plan")
        print(initial_plan)

        refined_plan = self.plan_refinement_loop(initial_plan)
        print('\nRefined Plan')
        print(refined_plan)



class MetacognitionModule:
    """
    Metacognition module aims to do basic hypothesis generation and introduces
    flexibility to the Cognitive Engine by allowing it to have a 'Playbook' of
    sorts that it draws inspiration from when thinking.
    """
    def __init__(self, llm) -> None:
        self.LLM = llm
        self.task_interpreter = TaskInterpreter(self.LLM)
        self.planner = Planning(self.LLM, max_refinement_iter=5)

    def set_playbook(self):
        self.playbook = (
            "Start Playbook:\n"
            "- If your current angle is higher than your target, you need to decrease the angle.\n"
            "- If your current angle is lower than your target, than you need to increase the angle.\n"
            "End Playbook"
        )
    
    def run(self, user_query):
        task_interpretation = self.task_interpreter.interpretation_of_user_query(user_query)
        plan = self.planner.generate_plan(task_interpretation)
