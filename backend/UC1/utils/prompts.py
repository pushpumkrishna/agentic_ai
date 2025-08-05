PROMPT1 = """
                You are an AI assistant that analyzes a user query and fills in input values for methods defined in the provided context.

                Context:
                {contexts}

                Query:
                {query}

                Instructions:

                - For each method, check if its required inputs are directly provided in the query.
                - If an input value is not found in the query, check if any previous method's output can be used instead.
                - If a method depends on a previous result, represent that input as "ClassName.methodName".
                - Only use such references if they logically make sense — do NOT force linking all methods.
                - Preserve the structure: Do NOT rename keys, drop methods, or add extra fields.
                - Your output should include ONLY the updated JSON pipeline structure with filled `inputs`.


                Expected Output format:
                {{
                "classes": [
                    {{
                    "class_name": "SomeClass",
                    "methods": [
                        {{
                        "method": "some_method",
                        "inputs": {{
                            "param1": "value1",
                            "param2": 42
                        }},
                        "output": "..."
                        }}
                    ]
                    }},
                    ...
                ]
                }}

                Only output this JSON structure. Do not explain anything.
                """

PROMPT2 = """
                You are an AI assistant that can choose a class to handle a user query.

                List of classes:
                {classes}

                User query: {user_query}

                Instruction:
                You are given a user query and a list of classes with their associated methods and method descriptions.

                Your task is to:
                1. Analyze the user query and determine which class or classes contain methods that can help answer the query.
                2. For each selected class, return a list of method names (only the names) that are relevant to the query.
                3. The output must be a dictionary in the following format:

                {{
                    'class_name1': ['method1', 'method2'],
                    'class_name2': ['method3']
                }}

                4. If a output to a query depends on execution of a set of methods first, include them in the format of <class_name>.<method_name>

                Rules:
                - Only include classes that contain at least one relevant method.
                - Only include methods that directly relate to the query.
                - If no methods match, return an empty dictionary: {{}}
                - Do not return any explanation or extra text—**only the dictionary output**.
                - You should not change the class name and use the exact class name as it is. 
                """
