import ast
import inspect
import re
from langchain.schema import HumanMessage
from langchain_core.prompts import PromptTemplate
from backend.config.azure_models import AzureOpenAIModels
from typing import Optional, Dict, Any, List
from backend.config.logging_lib import logger
from backend.utils.prompts import PROMPT1, PROMPT2


class PythonCodeReaderAgent:
    """
    - Description: A utility class to extract metadata from classes and methods using introspection.
    """

    def __init__(self):
        """
        - Description: Initializes the ClassMetadataExtractor instance.
        - params: None
        - return: None
        """
        self.llm = AzureOpenAIModels().get_azure_model_4()
        self.context = {}
        self.registered_class = {}
        self.method_docs = {}
        self.pipeline = None

    def register_class(self, instance, alias: Optional[str] = None):
        """
        - Description:
            Registers a class instance, extracts its docstrings and methods metadata,
            and stores structured metadata in a global context for later reference.

        - Params:
            instance (Any): The instance of the class to register.
            alias (Optional[str]): An optional alias name for the class.

        - Return:
            None

        - Exceptions:
            TypeError: If the instance is not an object or if docstring parsing fails.
            AttributeError: If attributes or methods are not found as expected.
            Exception: Catches unexpected errors during class registration.
        """
        logger.info("Starting class registration process.")
        try:
            if not hasattr(self, "registered_class"):
                raise AttributeError("Missing attribute 'registered_class' in self.")

            if not hasattr(self, "context"):
                raise AttributeError("Missing attribute 'context' in self.")

            if not hasattr(self, "method_docs"):
                raise AttributeError("Missing attribute 'method_docs' in self.")

            if not isinstance(instance, object):
                raise TypeError("Parameter 'instance' must be an object.")

            class_name = alias or instance.__class__.__name__
            self.registered_class[class_name] = instance

            # Get class-level docstring
            class_doc = inspect.getdoc(instance.__class__)

            # Build structured class metadata
            class_meta = {
                "class_name": class_name,
                "class_description": class_doc or "",
                "methods": [],
            }

            # Extract method-level metadata
            for name, func in inspect.getmembers(
                instance, predicate=inspect.isfunction
            ):
                if name.startswith("_"):
                    continue

                # Get the raw object from the class to preserve decorators
                attr = getattr(instance.__class__, name)

                # For @static methods or @class methods, get the actual function
                if isinstance(attr, (staticmethod, classmethod)):
                    func = attr.__func__
                doc = inspect.getdoc(func)
                if doc:
                    parsed = self.parse_docstring(doc)
                    method_data = {
                        "method": name,
                        "method_description": parsed["description"],
                        "inputs": parsed["inputs"],
                        "output": parsed["output"],
                        "raw_doc": doc,
                    }
                    class_meta["methods"].append(method_data)
                    self.method_docs[f"{class_name}.{name}"] = method_data

            # Store in global context
            self.context.setdefault("classes", []).append(class_meta)
            print("self.context: ", self.context)
            logger.info(f"Successfully registered class '{class_name}'.")

        except Exception as e:
            logger.exception(
                f"Error occurred during registration of class '{alias or instance.__class__.__name__}': {e}"
            )
            raise

    @staticmethod
    def parse_docstring(docstring: str) -> Dict[str, Any]:
        """
        Parses a structured docstring and extracts:
        - Description
        - Parameters (with types)
        - Return type or output

        - Params:
            docstring (str): The docstring to parse. It should be in a structured format.

        - Returns:
            Dict[str, Any]: A dictionary containing:
                - 'description': (str) The function description
                - 'inputs': (dict) A mapping of parameter names to their types
                - 'output': (str) The return type or return description

        - Possible Exceptions:
            TypeError: If `docstring` is not a string.
            Exception: For any unexpected errors during parsing.
        """
        logger.info("Started parsing the docstring...")

        if not isinstance(docstring, str):
            logger.error(
                "Invalid input type. Expected 'str', got '%s'", type(docstring).__name__
            )
            raise TypeError("Input 'docstring' must be a string.")

        parsed = {"description": "", "inputs": {}, "output": ""}

        try:
            lines = docstring.strip().splitlines()

            for line in lines:
                line = line.strip()
                if line.lower().startswith("- description:"):
                    parsed["description"] = line.split(":", 1)[1].strip()

                elif line.startswith("- param"):
                    match = re.match(r"- param (\w+): .*?:type:\s*(.*)", line)
                    if match:
                        param, param_type = match.groups()
                        parsed["inputs"][param] = param_type

                elif line.startswith(":rtype:"):
                    parsed["output"] = line.split(":", 1)[1].strip()
                elif line.startswith(":return:") and not parsed["output"]:
                    parsed["output"] = line.split(":", 1)[1].strip()

            logger.info("Parsing the docstring completed successfully.")
        except Exception as e:
            logger.exception(
                "An error occurred while parsing the docstring: %s", str(e)
            )
            raise

        return parsed

    def list_methods(self) -> List[Dict[str, Any]]:
        """
        Returns list of all methods, flattened and with class name included.

        - Params:
            self: Object instance having a `context` attribute with expected structure.

        - Returns:
            List[Dict[str, Any]]: A list of dictionaries, each representing a method with its class name.

        - Possible Exceptions:
            AttributeError: If `self.context` does not exist or is not a dict.
            KeyError: If expected keys like 'class_name' or 'methods' are missing from any class dict.
            Exception: For any unexpected error during list construction.
        """
        logger.info("Started listing methods from context.")

        if not hasattr(self, "context") or not isinstance(self.context, dict):
            logger.error("Invalid or missing 'context' attribute in the object.")
            raise AttributeError(
                "The object must have a 'context' attribute of type dict."
            )

        methods = []
        try:
            for cls in self.context.get("classes", []):
                class_name = cls.get("class_name")
                for m in cls.get("methods", []):
                    methods.append({**m, "class": class_name})
            logger.info("Completed listing methods successfully.")
        except Exception as e:
            logger.exception("An error occurred while listing methods: %s", str(e))
            raise

        return methods

    def list_classes(self) -> List[Dict[str, str]]:
        """
        Returns list of all registered classes with their descriptions.

        - Params:
            self: Object instance having a `context` attribute with expected structure.

        - Returns:
            List[Dict[str, str]]: List of class metadata with class name and description.

        - Possible Exceptions:
            AttributeError: If `self.context` is missing or not a dictionary.
            KeyError: If class dict is missing expected keys.
            Exception: For any unexpected error.
        """
        logger.info("Started listing classes from context.")

        if not hasattr(self, "context") or not isinstance(self.context, dict):
            logger.error("Invalid or missing 'context' attribute in the object.")
            raise AttributeError(
                "The object must have a 'context' attribute of type dict."
            )

        try:
            return [
                {
                    "class_name": cls["class_name"],
                    "class_description": cls["class_description"],
                }
                for cls in self.context.get("classes", [])
            ]
        except Exception as e:
            logger.exception("An error occurred while listing classes: %s", str(e))
            raise
        finally:
            logger.info("Completed listing classes.")

    def get_current_pipeline(self) -> Any:
        """
        Just returns the current pipeline in use. Will change per new query from User.

        - Params:
            self: Object instance expected to have a `pipeline` attribute.

        - Returns:
            Any: The current pipeline object in use.

        - Possible Exceptions:
            AttributeError: If `pipeline` attribute does not exist.
        """
        logger.info("Fetching current pipeline.")

        if not hasattr(self, "pipeline"):
            logger.error("'pipeline' attribute not found in the object.")
            raise AttributeError("The object must have a 'pipeline' attribute.")

        result = self.pipeline
        logger.info("Fetched current pipeline successfully.")
        return result

    def get_context(self) -> Dict[str, Any]:
        """
        Returns the context which contains all registered classes and their methods.

        - Params:
            self: Object instance expected to have a `context` attribute.

        - Returns:
            Dict[str, Any]: The context dictionary containing all class and method registrations.

        - Possible Exceptions:
            AttributeError: If `context` attribute does not exist.
        """
        logger.info("Fetching context.")

        if not hasattr(self, "context"):
            logger.error("'context' attribute not found in the object.")
            raise AttributeError("The object must have a 'context' attribute.")

        logger.info("Context fetched successfully.")
        return self.context

    def llm_determine_input_parameters(self, query: str) -> Dict[str, Any]:
        """
        Determines the required input parameters for each method using an LLM based on the user query.

        - Params:
            query (str): The user query to be interpreted by the LLM.

        - Returns:
            Dict[str, Any]: Parsed pipeline dictionary with identified inputs and structure.

        - Possible Exceptions:
            AttributeError: If `self.pipeline` or `self.llm` is not defined.
            Exception: If LLM response cannot be parsed or invocation fails.
        """
        logger.info("Started determining input parameters via LLM for the given query.")

        if not hasattr(self, "pipeline") or not isinstance(self.pipeline, dict):
            logger.error("'pipeline' attribute missing or not a dictionary.")
            raise AttributeError(
                "Expected 'pipeline' attribute of type dict in the object."
            )

        if not hasattr(self, "llm"):
            logger.error("'llm' attribute not found in the object.")
            raise AttributeError(
                "Expected 'llm' attribute to be present for invoking the model."
            )
        response = " "

        try:
            context = "\n".join(
                [
                    f"""Class: {cls["class_name"]}
            Method: {met["method"]}
            Inputs: {", ".join([f"{k} ({v})" for k, v in met.get("inputs", {}).items()])}
            Output: {met.get("output", "N/A")}"""
                    for cls in self.pipeline.get("classes", [])
                    for met in cls.get("methods", [])
                ]
            )
            prompt = PromptTemplate(
                input_variables=["contexts", "query"], template=PROMPT1
            )

            logger.debug("Sending formatted prompt to LLM.")
            formatted_prompt = prompt.format(contexts=context, query=query)
            message = HumanMessage(content=formatted_prompt)
            response = self.llm.invoke([message]).content.strip()

            parsed = ast.literal_eval(response)
            self.pipeline = parsed
            logger.info("LLM successfully returned and parsed input parameters.")
            return parsed
        except Exception as e:
            logger.exception(
                "Error occurred while parsing LLM response or invoking LLM: %s", str(e)
            )
            logger.error("Raw response from LLM:\n%s", response)
            return {}

    def llm_fetch_class_methods(self, query: str) -> Any:
        """
        Fetches class methods relevant to a user query using LLM.

        - Params:
            query (str): The user query based on which relevant class methods are fetched.

        - Returns:
            Any: A dictionary (or appropriate structure) representing relevant methods as returned by the LLM.

        - Possible Exceptions:
            AttributeError: If required attributes like `context`, `llm`, or `pipeline` are missing.
            Exception: If LLM response parsing fails or structure is invalid.
        """
        logger.info("Started fetching class methods from LLM.")

        if not hasattr(self, "context") or not isinstance(self.context, dict):
            logger.error("'context' attribute missing or not a dictionary.")
            raise AttributeError(
                "Expected 'context' attribute of type dict in the object."
            )

        if not hasattr(self, "llm"):
            logger.error("'llm' attribute not found in the object.")
            raise AttributeError(
                "Expected 'llm' attribute for invoking the language model."
            )

        try:
            prompt = PromptTemplate(
                input_variables=["classes", "user_query"], template=PROMPT2
            )
            classes_desc = "\n".join(
                [
                    f"Class: {cls.get('class_name', 'Unknown')} — Method: {met.get('method', 'Unnamed')} "
                    f"— Description: {met.get('method_description', '')}"
                    for cls in self.context.get("classes", [])
                    for met in cls.get("methods", [])
                ]
            )

            message = HumanMessage(
                content=prompt.format(classes=classes_desc, user_query=query)
            )

            logger.debug("Sending prompt to LLM for class-method matching.")
            resp = self.llm.invoke([message]).content.strip()

            try:
                resp = ast.literal_eval(resp)
                logger.info("Successfully parsed LLM response.")
            except Exception as err:
                logger.exception(
                    "Error converting LLM output to dictionary: %s", str(err)
                )
                logger.error("Raw LLM output: %s", resp)
                return {}

            self.pipeline = self.get_method_context_subset(resp)
            logger.info("Updated pipeline with matched methods successfully.")
            return resp

        except Exception as e:
            logger.exception(
                "Unexpected error during LLM class-method fetch: %s", str(e)
            )
            return {}

    def get_method_context_subset(
        self, selected_dict: Optional[Dict[str, List[str]]] = None
    ) -> Dict[str, Any]:
        """
        Prunes self. context['classes'] based on a selected class/methods dictionary.
        Returns a filtered structure with only relevant class/method pairs.

        - Params:
            selected_dict (Optional[Dict[str, List[str]]]): Mapping of class names to list of method names.
                If not provided, falls back to self.pipeline.

        - Returns:
            Dict[str, Any]: Filtered context dictionary containing only the selected classes and methods.

        - Possible Exceptions:
            AttributeError: If `context` or `pipeline` is not defined.
            KeyError: If expected keys are missing from any class or method.
        """
        logger.info("Pruning method context based on selected classes and methods.")

        if not hasattr(self, "context"):
            logger.error("'context' attribute missing in object.")
            raise AttributeError("The object must have a 'context' attribute.")

        if selected_dict is None:
            if not hasattr(self, "pipeline"):
                logger.error("'pipeline' not provided and 'self.pipeline' missing.")
                raise AttributeError("The object must have a 'pipeline' attribute.")
            selected_dict = self.pipeline

        pruned_context = {"classes": []}

        try:
            for cls in self.context.get("classes", []):
                class_name = cls["class_name"]
                if class_name not in selected_dict:
                    continue

                selected_methods = []
                for method in cls["methods"]:
                    if method["method"] in selected_dict[class_name]:
                        method.pop("method_description", None)
                        method.pop("raw_doc", None)
                        selected_methods.append(method)

                if selected_methods:
                    pruned_context["classes"].append(
                        {
                            "class_name": class_name,
                            # "class_description": cls["class_description"],
                            "methods": selected_methods,
                        }
                    )

            logger.info("Pruned method context successfully.")
            return pruned_context
        except Exception as e:
            logger.exception("Error pruning method context: %s", str(e))
            raise

    def run_pipeline_with_dependencies(
        self, pipeline: Optional[Dict[str, Any]] = None, max_passes: int = 5
    ) -> Dict[str, Any]:
        """
        Executes a structured pipeline that may include method dependencies.
        Handles multi-pass resolution using "Class.method" references in inputs.

        - Params:
            pipeline (Optional[Dict[str, Any]]): The pipeline dictionary to execute. Defaults to self.pipeline.
            max_passes (int): Number of attempts to resolve method dependencies.

        - Returns:
            Dict[str, Any]: A dictionary containing results of executed methods.

        - Possible Exceptions:
            AttributeError: If required attributes like 'context' or 'registered_class' are missing.
            Exception: If a method execution fails unexpectedly.
        """
        logger.info("Starting pipeline execution with dependency resolution.")

        pipeline = pipeline or self.pipeline

        if not hasattr(self, "registered_class"):
            logger.error("'registered_class' attribute is missing.")
            raise AttributeError("The object must have a 'registered_class' attribute.")

        if not hasattr(self, "context"):
            logger.error("'context' attribute is missing.")
            raise AttributeError("The object must have a 'context' attribute.")

        results = {}
        context = self.context
        executed = set()

        steps = []
        remaining = []
        for cls in pipeline.get("classes", []):
            class_name = cls["class_name"]
            instance = self.registered_class.get(class_name)
            if not instance:
                logger.warning(f"Class '{class_name}' is not registered.")
                continue
            for method in cls["methods"]:
                steps.append(
                    {
                        "class": class_name,
                        "instance": instance,
                        "method_name": method["method"],
                        "inputs": method["inputs"],
                    }
                )

        for _ in range(max_passes):
            progress = False
            for step in steps:
                key = f"{step['class']}.{step['method_name']}"
                if key in executed:
                    continue

                resolved_inputs = {}
                inputs = step["inputs"]
                can_execute = True

                for param, val in inputs.items():
                    if isinstance(val, (int, float)) or (
                        isinstance(val, str) and "." not in val
                    ):
                        resolved_inputs[param] = val
                    elif isinstance(val, str) and val in results:
                        resolved_inputs[param] = results[val]
                    elif isinstance(val, str) and val in context:
                        resolved_inputs[param] = context[val]
                    else:
                        can_execute = False
                        break

                if can_execute:
                    try:
                        method_fn = getattr(step["instance"], step["method_name"])
                        logger.info(f"Executing {key} with inputs: {resolved_inputs}")
                        output = method_fn(**resolved_inputs)

                        results[key] = output
                        context[key] = output
                        if isinstance(output, dict):
                            context.update(output)

                        executed.add(key)
                        progress = True
                    except Exception as e:
                        logger.exception(f"Error executing {key}: {e}")
                else:
                    remaining.append(step)

            if not progress:
                logger.warning("No progress made in this pass, breaking.")
                break
            steps = remaining
            remaining = []

        if remaining:
            logger.warning("Unresolved methods due to missing inputs or errors:")
            for r in remaining:
                logger.warning(
                    f" - {r['class']}.{r['method_name']} (inputs: {r['inputs']})"
                )

        logger.info("Pipeline execution completed.")
        return results
