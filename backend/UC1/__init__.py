# import ast
# import inspect
# import re
# from langchain.schema import HumanMessage
# from langchain_core.prompts import PromptTemplate
# from backend.UC1.config.azure_models import AzureOpenAIModels
#
#
# class Agent:
#     """
#     - Description: A utility class to extract metadata from classes and methods using introspection.
#     """
#     def __init__(self):
#         """
#         - Description: Initializes the ClassMetadataExtractor instance.
#         - params: None
#         - return: None
#         """
#         self.llm = AzureOpenAIModels().get_azure_model_4()
#         self.context = {}
#         self.registered_class = {}
#         self.method_docs = {}
#         self.pipeline = None
#
#     def register_class(self, instance, alias=None):
#         class_name = alias or instance.__class__.__name__
#         self.registered_class[class_name] = instance
#
#         # Get class-level docstring
#         class_doc = inspect.getdoc(instance.__class__)
#
#         # Build structured class metadata
#         class_meta = {
#             "class_name": class_name,
#             "class_description": class_doc or "",
#             "methods": []
#         }
#
#         # Extract method-level metadata
#         for name, func in inspect.getmembers(instance, predicate=inspect.isfunction):
#             if name.startswith("_"):
#                 continue
#
#             # Get the raw object from the class to preserve decorators
#             attr = getattr(instance.__class__, name)
#
#             # For @static methods or @class methods, get the actual function
#             if isinstance(attr, (staticmethod, classmethod)):
#                 func = attr.__func__
#             doc = inspect.getdoc(func)
#             if doc:
#                 parsed = self.parse_docstring(doc)
#                 method_data = {
#                     "method": name,
#                     "method_description": parsed["description"],
#                     "inputs": parsed["inputs"],
#                     "output": parsed["output"],
#                     "raw_doc": doc
#                 }
#                 class_meta["methods"].append(method_data)
#                 self.method_docs[f"{class_name}.{name}"] = method_data
#
#         # Store in global context
#         self.context.setdefault("classes", []).append(class_meta)
#         print("self.context: ", self.context)
#
#     @staticmethod
#     def parse_docstring(docstring):
#         """
#         Parses a structured docstring with:
#         - Description
#         - Parameters (with types)
#         - Return or rtype
#
#         Returns:
#         {
#             description: str,
#             inputs: dict,
#             output: str
#         }
#         """
#         parsed = {
#             "description": "",
#             "inputs": {},
#             "output": ""
#         }
#
#         lines = docstring.strip().splitlines()
#
#         for line in lines:
#             line = line.strip()
#             if line.lower().startswith("- description:"):
#                 parsed["description"] = line.split(":", 1)[1].strip()
#
#             elif line.startswith("- param"):
#                 match = re.match(r"- param (\w+): .*?:type:\s*(.*)", line)
#                 if match:
#                     param, param_type = match.groups()
#                     parsed["inputs"][param] = param_type
#
#             elif line.startswith(":rtype:"):
#                 parsed["output"] = line.split(":", 1)[1].strip()
#             elif line.startswith(":return:") and not parsed["output"]:
#                 parsed["output"] = line.split(":", 1)[1].strip()
#
#         return parsed
#
#     def list_methods(self):
#         """
#         Returns list of all methods, flattened and with class name included.
#         """
#         methods = []
#         for cls in self.context.get("classes", []):
#             class_name = cls["class_name"]
#             for m in cls["methods"]:
#                 methods.append({
#                     **m,
#                     "class": class_name
#                 })
#         return methods
#
#     def list_classes(self):
#         """
#         Returns list of all registered classes with their descriptions.
#         """
#         return [
#             {
#                 "class_name": cls["class_name"],
#                 "class_description": cls["class_description"]
#             }
#             for cls in self.context.get("classes", [])
#         ]
#
#     def get_current_pipeline(self):
#         """
#         Just returns the current pipeline in use. Will change per new query from User.
#         """
#         return self.pipeline
#
#     def get_context(self):
#         """
#         Returns the context which contains all registered classes and their methods.
#         """
#         return self.context
#
#     def llm_determine_input_parameters(self, query):
#         context = "\n".join(
#             [
#                 f"""Class: {cls['class_name']}
#         Method: {met['method']}
#         Inputs: {', '.join([f"{k} ({v})" for k, v in met.get('inputs', {}).items()])}
#         Output: {met.get('output', 'N/A')}"""
#                 for cls in self.pipeline.get("classes", [])
#                 for met in cls.get("methods", [])
#             ]
#         )
#         prompt = PromptTemplate(
#             input_variables=["contexts", "query"],
#             template="""
#                 You are an AI assistant that analyzes a user query and fills in input values for methods defined in the provided context.
#
#                 Context:
#                 {contexts}
#
#                 Query:
#                 {query}
#
#                 Instructions:
#
#                 - For each method, check if its required inputs are directly provided in the query.
#                 - If an input value is not found in the query, check if any previous method's output can be used instead.
#                 - If a method depends on a previous result, represent that input as "ClassName.methodName".
#                 - Only use such references if they logically make sense — do NOT force linking all methods.
#                 - Preserve the structure: Do NOT rename keys, drop methods, or add extra fields.
#                 - Your output should include ONLY the updated JSON pipeline structure with filled `inputs`.
#
#
#                 Expected Output format:
#                 {{
#                 "classes": [
#                     {{
#                     "class_name": "SomeClass",
#                     "methods": [
#                         {{
#                         "method": "some_method",
#                         "inputs": {{
#                             "param1": "value1",
#                             "param2": 42
#                         }},
#                         "output": "..."
#                         }}
#                     ]
#                     }},
#                     ...
#                 ]
#                 }}
#
#                 Only output this JSON structure. Do not explain anything.
#                 """
#         )
#
#         # Format and send prompt to LLM
#         formatted_prompt = prompt.format(contexts=context, query=query)
#         message = HumanMessage(content=formatted_prompt)
#         response = self.llm.invoke([message]).content.strip()
#
#         try:
#             parsed = ast.literal_eval(response)
#             self.pipeline = parsed
#             print("\nPipeline with inputs:\n")
#             return parsed
#         except Exception as e:
#             print("Error parsing LLM response:", e)
#             print("Raw response:\n", response)
#             return {}
#
#     def llm_fetch_class_methods(self, query):
#         prompt = PromptTemplate(
#             input_variables=["classes", "user_query"],
#             template="""
#                 You are an AI assistant that can choose a class to handle a user query.
#
#                 List of classes:
#                 {classes}
#
#                 User query: {user_query}
#
#                 Instruction:
#                 You are given a user query and a list of classes with their associated methods and method descriptions.
#
#                 Your task is to:
#                 1. Analyze the user query and determine which class or classes contain methods that can help answer the query.
#                 2. For each selected class, return a list of method names (only the names) that are relevant to the query.
#                 3. The output must be a dictionary in the following format:
#
#                 {{
#                     'class_name1': ['method1', 'method2'],
#                     'class_name2': ['method3']
#                 }}
#
#                 4. If a output to a query depends on execution of a set of methods first, include them in the format of <class_name>.<method_name>
#
#                 Rules:
#                 - Only include classes that contain at least one relevant method.
#                 - Only include methods that directly relate to the query.
#                 - If no methods match, return an empty dictionary: {{}}
#                 - Do not return any explanation or extra text—**only the dictionary output**.
#                 - You should not change the class name and use the exact class name as it is.
#                 """)
#         classes_desc = "\n".join(
#             [
#                 f"Class: {cls.get('class_name', 'Unknown')} — Method: {met.get('method', 'Unnamed')} — Description: {met.get('method_description', '')}"
#                 for cls in self.context.get("classes", [])
#                 for met in cls.get("methods", [])
#             ]
#         )
#
#         message = HumanMessage(
#             content=prompt.format(classes=classes_desc, user_query=query)
#         )
#
#         resp = self.llm.invoke([message]).content.strip()
#
#         try:
#             resp = ast.literal_eval(resp)
#         except Exception as err:
#             print("Error converting to dictionary. LLM didn't provide dictionary formatted output: {err}")
#         self.pipeline = self.get_method_context_subset(resp)
#         return resp
#
#     def get_method_context_subset(self, selected_dict=None):
#         """
#         Prunes self. context['classes'] based on selected class/methods dictionary.
#         Returns a filtered structure with only relevant class/method pairs.
#         """
#         pruned_context = {"classes": []}
#         selected_dict = selected_dict or self.pipeline
#         for cls in self.context.get("classes", []):
#             class_name = cls["class_name"]
#             if class_name not in selected_dict:
#                 continue
#
#             # Filter methods
#             selected_methods = []
#             for method in cls["methods"]:
#                 if method["method"] in selected_dict[class_name]:
#                     method.pop('method_description')
#                     method.pop('raw_doc')
#                     selected_methods.append(method)
#
#             if selected_methods:
#                 pruned_context["classes"].append({
#                     "class_name": class_name,
#                     # "class_description": cls["class_description"],
#                     "methods": selected_methods
#                 })
#         return pruned_context
#
#     def run_pipeline_with_dependencies(self, pipeline=None, max_passes=5):
#         """
#         Executes a structured pipeline that may include method dependencies.
#         Respects dependency ordering using "Class.method" notation in inputs.
#         """
#         pipeline = pipeline or self.pipeline
#         results = {}
#         context = self.context
#         unresolved_methods = []
#         executed = set()
#
#         # Flatten all method steps
#         steps = []
#         remaining = []
#         for cls in pipeline.get("classes", []):
#             class_name = cls["class_name"]
#             instance = self.registered_class.get(class_name)
#             if not instance:
#                 print(f"❌ Class '{class_name}' is not registered.")
#                 continue
#             for method in cls["methods"]:
#                 steps.append({
#                     "class": class_name,
#                     "instance": instance,
#                     "method_name": method["method"],
#                     "inputs": method["inputs"]
#                 })
#
#         # Resolve dependencies in multiple passes
#         for _ in range(max_passes):
#             progress = False
#
#             for step in steps:
#                 key = f"{step['class']}.{step['method_name']}"
#                 if key in executed:
#                     continue
#
#                 resolved_inputs = {}
#                 inputs = step["inputs"]
#                 can_execute = True
#
#                 for param, val in inputs.items():
#                     if isinstance(val, (int, float, str)) and not isinstance(val, str) or "." not in str(val):
#                         resolved_inputs[param] = val
#                     elif isinstance(val, str) and val in results:
#                         resolved_inputs[param] = results[val]
#                     elif isinstance(val, str) and val in context:
#                         resolved_inputs[param] = context[val]
#                     else:
#                         can_execute = False
#                         break
#
#                 if can_execute:
#                     try:
#                         method_fn = getattr(step["instance"], step["method_name"])
#                         print(f"Executing {key} with inputs: {resolved_inputs}")
#                         output = method_fn(**resolved_inputs)
#
#                         results[key] = output
#                         context[key] = output
#                         if isinstance(output, dict):
#                             context.update(output)
#
#                         executed.add(key)
#                         progress = True
#                     except Exception as e:
#                         print(f"Error executing {key}: {e}")
#                 else:
#                     remaining.append(step)
#
#             if not progress:
#                 break
#             steps = remaining
#
#         # Final report
#         if remaining:
#             print("\nUnresolved methods due to missing inputs or errors:")
#             for r in remaining:
#                 print(f" - {r['class']}.{r['method_name']} (inputs: {r['inputs']})")
#
#         return results
