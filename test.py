from llm_interface import LLMInterface

query = "Give me a python function that multiplies two numbers."


llm = LLMInterface(
        model='gpt-4o-mini',
        max_tokens=5000,
        temperature=0.4)

messages = [
    {
        'role': 'system',
        'content': 'I am a senior software architect.'
    },
    {
        'role': 'assistant',
        'content': f'A user has provided me the following request: {query}'
    },
    {
        'role': 'user',
        'content': """Break this request down into a comprehensive implementation plan and output it in valid YAML format. The YAML should include the following top-level keys:

            high_level_overview:
            description: |
                A concise summary of the problem to solve and the core features required.

            architecture_module_layout:
            description: |
                A description of the main components or modules and how they interact
                (e.g. data flow, class relationships, external services).

            function_breakdown:
            modules:
                - name: <module_name>
                functions:
                    - name: <function_name>
                    signature: <parameters_and_return_types>
                    description: |
                        What it does, side effects, error handling.

            Use YAML-like sturcture for your output, correctly indented, and uses appropriate YAML constructs (scalars, sequences, mappings)."""
    }
]


print(llm.query(messages=[{'role':'user', 'content': query}]))

# general_plan = llm.query(messages)
# print('---------------------PLAN---------------------')
# print(general_plan)
# print('----------------------------------------------')


# messages = [
#     {
#         "role": "system",
#         "content": "You are a senior Python software architect and code generator."
#     },
#     {
#         "role": "assistant",
#         "content": f"Here is my implementation plan in YAML format for code I am going to generate:\n {general_plan}"
#     },
#     {
#         "role": "user",
#         "content": """Generate an executable Python code base skeleton that implements this plan.  
#             - Output only valid Python code (no markdown fences or explanatory text).  
#             - Include necessary imports for any external libraries listed.  
#             - Create modules or classes as indicated by the YAML.  
#             - For each function in `function_breakdown`, define a `def` with the correct name and signature, include a docstring summarizing its description, and use `pass` or `# TODO` as a placeholder.  
#             - Wire up the functions coherently under a `if __name__ == "__main__":` block or within a `main()` function so the script runs without errors.  
#             - Do not implement business logic—only stubs and the structure needed for execution.""" 
#     }
# ]

# skeleton_code = llm.query(messages)
# print('---------------------CODE---------------------')
# print(skeleton_code)
# print('----------------------------------------------')