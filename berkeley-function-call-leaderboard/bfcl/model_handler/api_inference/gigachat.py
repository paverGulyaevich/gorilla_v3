import os
import time
from typing import Any

from gigachat import GigaChat
from gigachat.exceptions import ResponseError
from tenacity import stop_after_attempt

from bfcl.model_handler.base_handler import BaseHandler
from bfcl.model_handler.constant import GORILLA_TO_OPENAPI
from bfcl.model_handler.model_style import ModelStyle
from bfcl.model_handler.utils import (
    convert_to_function_call,
    convert_to_tool,
    func_doc_language_specific_pre_processing,
    retry_with_backoff,
)


def _sanitize_gigachat_schema(obj):
    if isinstance(obj, dict):
        if "enum" in obj and isinstance(obj["enum"], list):
            if any(not isinstance(x, str) for x in obj["enum"]):
                obj["enum"] = [str(x) for x in obj["enum"]]
                obj["type"] = "string"
            else:
                obj["enum"] = [str(x) for x in obj["enum"]]
                if obj.get("type") not in (None, "string"):
                    obj["type"] = "string"

        if "items" in obj and isinstance(obj["items"], dict):
            if "default" in obj["items"] and not isinstance(
                obj["items"]["default"], str
            ):
                default_val = obj["items"]["default"]
                desc = obj["items"].get("description", "")
                extra = f" Default is: {default_val}."
                obj["items"]["description"] = (
                    f"{desc}{extra}" if desc else f"Default is: {default_val}."
                )
                del obj["items"]["default"]

        if "default" in obj and not isinstance(
            obj["default"], (str, int, float, bool, type(None))
        ):
            default_val = obj["default"]
            desc = obj.get("description", "")
            extra = f" Default is: {default_val}."
            obj["description"] = (
                f"{desc}{extra}" if desc else f"Default is: {default_val}."
            )
            del obj["default"]

        for value in obj.values():
            if isinstance(value, (dict, list)):
                _sanitize_gigachat_schema(value)
    elif isinstance(obj, list):
        for item in obj:
            _sanitize_gigachat_schema(item)


class GigaChatHandler(BaseHandler):
    def __init__(self, model_name, temperature):
        super().__init__(model_name, temperature)
        self.model_style = ModelStyle.GigaChat
        self.is_fc_model = True
        self.client = GigaChat(**self._build_client_kwargs())
        print(self.client.chat("Hi"))

    def _build_client_kwargs(self):
        """Collect GigaChat client keyword arguments"""


        kwargs = {
            "profanity_check": False,
            "verify_ssl_certs": False,
            "scope": "GIGACHAT_API_CORP",
            "model": self.model_name
        }

        if os.getenv("GIGACHAT_ENV") == "prod":
            prod_url = os.getenv("GIGACHAT_PROD_URL")
            credentials = os.getenv("GIGACHAT_CREDENTIALS")
            kwargs.update(base_url=prod_url, credentials=credentials)
        else:
            token = os.getenv("GIGACHAT_API_KEY")
            ift_url = os.getenv("GIGACHAT_IFT_URL")
            kwargs.update(base_url=ift_url, access_token=token)

        return kwargs    

    def decode_ast(self, result, language="Python"):
        """
        Convert the raw model response to the format [{func1:{param1:val1,...}},{func2:{param2:val2,...}}];
        i.e., a list of dictionaries, each representing a function call with the function name as the key and the parameters as the value.
        This is the format that the evaluation pipeline expects.
        """
        if self.is_fc_model:
            decoded_output = []
            for invoked_function in result:
                name = list(invoked_function.keys())[0]
                params = invoked_function[name]
                decoded_output.append({name: params})
            return decoded_output
        raise ValueError("Gigachat model can work only in FC mode.")  

    def decode_execute(self, result):
        """
        Convert the raw model response to the format ["func1(param1=val1)", "func2(param2=val2)"]; i.e.,
        a list of strings, each representing an executable function call.
        """
        if self.is_fc_model:
            return convert_to_function_call(result)
        raise ValueError("Gigachat model can work only in FC mode.")

    @retry_with_backoff(error_type=ResponseError, stop=stop_after_attempt(5))
    def generate_with_backoff(self, kwargs):

        start_time = time.time()
        api_response = self.client.chat(kwargs)
        end_time = time.time()

        return api_response, end_time - start_time

    #### FC methods ####
    def _query_FC(self, inference_data: dict):
        """
        Call the model API in FC mode to get the response.
        Return the response object that can be used to feed into the decode method.
        """
        message: list[dict] = inference_data["message"]
        tools = inference_data["tools"]
        inference_data["inference_input_log"] = {
            "message": repr(message),
            "tools": tools,
        }

        reasoning_effort = "medium" if "Reasoning" in self.model_name else None

        kwargs = {
            "messages": message,
            "temperature": self.temperature, # 0.001 by default
            "function_call": "auto",
            "reasoning_effort": reasoning_effort,
        }

        tools = tools if len(tools) > 0 else []
        kwargs["functions"] = tools

        return self.generate_with_backoff(kwargs)

    def _pre_query_processing_FC(self, inference_data: dict, test_entry: dict) -> dict:
        """
        Preprocess the testset entry before sending it to the model.
        This includes transforming the input user message into the format expected by the model, and any other necessary preprocessing steps.
        The inference_data dict is updated in place and returned.
        """
        inference_data["message"] = []
        return inference_data

    def _compile_tools(self, inference_data: dict, test_entry: dict) -> dict:
        """
        Compile the tools from the test entry and add them to the inference data.
        This method is used to prepare the tools for the model query in FC mode.
        The inference_data dict is updated in place and returned.
        """

        functions: list = test_entry["function"]
        test_category: str = test_entry["id"].rsplit("_", 1)[0]

        functions = func_doc_language_specific_pre_processing(functions, test_category)
        tools = convert_to_tool(functions, GORILLA_TO_OPENAPI, self.model_style)
        # _sanitize_gigachat_schema(tools) 
        inference_data["tools"] = tools

        return inference_data

    def _parse_query_response_FC(self, api_response: Any) -> dict:
        """
        Parses the raw response from the model API to extract the result, input token count, and output token count.

        Args:
            api_response (any): The raw response from the model API.

        Returns:
            A dict containing the following elements:
                - model_responses (any): The parsed result that can be directly used as input to the decode method.
                - input_token (int): The number of tokens used in the input to the model.
                - output_token (int): The number of tokens generated by the model as output.
                - tool_call_ids (list[str]): The IDs of the tool calls that are generated by the model. Optional.
                - Any other metadata that is specific to the model.
        """
        try:
            model_responses = [
                {func_call.name: func_call.arguments}
                for func_call in [api_response.choices[0].message.function_call]
            ]
            tool_call_ids = [
                func_call.name
                for func_call in [api_response.choices[0].message.function_call]
            ]
        except Exception:
            model_responses = api_response.choices[0].message.content
            tool_call_ids = []

        model_responses_message_for_chat_history = api_response.choices[0].message.dict()

        return {
            "model_responses": model_responses,
            "model_responses_message_for_chat_history": model_responses_message_for_chat_history,
            "tool_call_ids": tool_call_ids,
            "input_token": api_response.usage.prompt_tokens,
            "output_token": api_response.usage.completion_tokens,
        }

    def add_first_turn_message_FC(
        self, inference_data: dict, first_turn_message: list[dict]
    ) -> dict:
        """
        Add the first turn message to the chat history.
        """
        inference_data["message"].extend(first_turn_message)
        return inference_data

    def _add_next_turn_user_message_FC(
        self, inference_data: dict, user_message: list[dict]
    ) -> dict:
        """
        [Only for multi-turn]
        Add next turn user message to the chat history for query.
        user_message is a list of 1 element, which is the user message.
        """
        inference_data["message"].extend(user_message)
        return inference_data

    def _add_assistant_message_FC(
        self, inference_data: dict, model_response_data: dict
    ) -> dict:
        """
        Add assistant message to the chat history.
        """

        inference_data["message"].append(
            model_response_data["model_responses_message_for_chat_history"]
        )
        return inference_data

    def _add_execution_results_FC(
        self,
        inference_data: dict,
        execution_results: list[str],
        model_response_data: dict,
    ) -> dict:
        """
        Add the execution results to the chat history to prepare for the next turn of query.
        Some models may need to add additional information to the chat history, such as tool call IDs.
        """

        # Add the execution results to the current round result, one at a time
        for execution_result, tool_call_id in zip(
            execution_results, model_response_data["tool_call_ids"]
        ):
            tool_message = {
                "role": "function",
                "content": execution_result,
                "name": tool_call_id,
            }
            inference_data["message"].append(tool_message)

        return inference_data
            
        # for execution_result in execution_results:
        #     tool_message = {
        #         "role": "function",
        #         "content": json.dumps({"result": execution_result}),
        #     }
        #     inference_data["message"].append(tool_message)

        # return inference_data

    #### Prompting methods ####

    def _query_prompting(self, inference_data: dict):
        """
        Call the model API in prompting mode to get the response.
        Return the response object that can be used to feed into the decode method.
        """
        raise NotImplementedError

    def _pre_query_processing_prompting(self, test_entry: dict) -> dict:
        """
        Preprocess the testset entry before sending it to the model.
        Returns a dict that contains all the necessary information for the query method.
        `tools` and `message` must be included in the returned dict.
        Things like `system_prompt` and `chat_history` are optional, specific to the model.
        """
        raise NotImplementedError

    def _parse_query_response_prompting(self, api_response: Any) -> dict:
        """
        Parses the raw response from the model API to extract the result, input token count, and output token count.

        Args:
            api_response (any): The raw response from the model API.

        Returns:
            A dict containing the following elements:
                - model_responses (any): The parsed result that can be directly used as input to the decode method.
                - input_token (int): The number of tokens used in the input to the model.
                - output_token (int): The number of tokens generated by the model as output.
                - tool_call_ids (list[str]): The IDs of the tool calls that are generated by the model. Optional.
                - Any other metadata that is specific to the model.
        """
        raise NotImplementedError

    def add_first_turn_message_prompting(
        self, inference_data: dict, first_turn_message: list[dict]
    ) -> dict:
        """
        Add the first turn message to the chat history.
        """
        raise NotImplementedError

    def _add_next_turn_user_message_prompting(
        self, inference_data: dict, user_message: list[dict]
    ) -> dict:
        """
        [Only for multi-turn]
        Add next turn user message to the chat history for query.
        user_message is a list of 1 element, which is the user message.
        """
        raise NotImplementedError

    def _add_assistant_message_prompting(
        self, inference_data: dict, model_response_data: dict
    ) -> dict:
        """
        Add assistant message to the chat history.
        """
        raise NotImplementedError

    def _add_execution_results_prompting(
        self,
        inference_data: dict,
        execution_results: list[str],
        model_response_data: dict,
    ) -> dict:
        """
        Add the execution results to the chat history to prepare for the next turn of query.
        Some models may need to add additional information to the chat history, such as tool call IDs.
        """
        raise NotImplementedError
