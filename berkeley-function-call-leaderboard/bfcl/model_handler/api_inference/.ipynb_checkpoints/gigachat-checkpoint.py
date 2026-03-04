import json
import os
import requests
import time

from bfcl.model_handler.base_handler import BaseHandler
from bfcl.model_handler.constant import GORILLA_TO_OPENAPI
from bfcl.model_handler.model_style import ModelStyle
from bfcl.model_handler.utils import (
    convert_to_function_call,
    convert_to_tool,
    default_decode_ast_prompting,
    default_decode_execute_prompting,
    format_execution_results_prompting,
    func_doc_language_specific_pre_processing,
    system_prompt_pre_processing_chat_model,
    convert_system_prompt_into_user_prompt,
    combine_consecutive_user_prompts,
)

from gigachat import GigaChat
from gigachat.models.chat import Chat
from gigachat.exceptions import ResponseError

from urllib.error import HTTPError

SYSTEM_PROMPT_FC = """You are an expert in composing functions. You are given a question and a set of possible functions. Based on the question, you will need to make one or more function/tool calls to achieve the purpose.
If none of the function can be used, point it out. If the given question lacks the parameters required by the function, also point it out.
You should only return the function calls in your response.
"""

CREDENTIALS = "ZTZkM2ZmODYtNDRmNC00OWQ0LTkyNTUtOTA1NzE1ZGY2ZTFjOmZkM2M5Nzk1LTY4ZjItNDU1Ni1hN2NlLWU5ODIzNzIwZDMwNg=="
URL_PROD = "https://gigachat.devices.sberbank.ru/api/v1"

with open("/home/jovyan/GIGA_TOKEN.json") as f:
    TOKEN = json.load(f)["response"]

# URL = "https://wmapi-ift.saluteai-pd.sberdevices.ru/v1/chat/completions"
URL = "https://gigachat.ift.sberdevices.ru/v1"

HEADERS = {
  'Content-Type': 'application/json',
  'Accept': 'application/json',
  'Authorization': f'Bearer {TOKEN}'
}
MODEL = "GigaChat-Max"


class GigaChatHandler(BaseHandler):
    def __init__(self, model_name, temperature):
        super().__init__(model_name, temperature)
        self.model_style = ModelStyle.GigaChat
        self.is_fc_model = True
        self.client = GigaChat(
            access_token=TOKEN,
            base_url=URL,
            model=MODEL,
            profanity_check=False,
            verify_ssl_certs=False,
            scope="GIGACHAT_API_CORP",
            timeout=60
        )
        print(self.client.chat("Hi"))
        
    # def __init__(self, model_name, temperature):
    #     super().__init__(model_name, temperature)
    #     self.model_style = ModelStyle.GigaChat
    #     self.is_fc_model = True
    #     self.client = GigaChat(
    #         credentials=CREDENTIALS,
    #         base_url=URL_PROD,
    #         model=MODEL,
    #         profanity_check=False,
    #         verify_ssl_certs=False,
    #         scope="GIGACHAT_API_CORP"
    #     )


    def decode_ast(self, result, language="Python"):
        """
        Convert the raw model response to the format [{func1:{param1:val1,...}},{func2:{param2:val2,...}}]; 
        i.e., a list of dictionaries, each representing a function call with the function name as the key and the parameters as the value. 
        This is the format that the evaluation pipeline expects.
        """
        decoded_output = []
        for invoked_function in result:
            name = list(invoked_function.keys())[0]
            params = invoked_function[name]
            decoded_output.append({name: params})
        return decoded_output
        
    def decode_execute(self, result):
        """
        Convert the raw model response to the format ["func1(param1=val1)", "func2(param2=val2)"]; i.e.,
        a list of strings, each representing an executable function call.
        """
        function_call = convert_to_function_call(result)
        # print("Decode execute")
        # print(function_call)
        return function_call

    #### FC methods ####
    
    # Via requests
    # def _query_FC(self, inference_data: dict):
    #     """
    #     Call the model API in FC mode to get the response.
    #     Return the response object that can be used to feed into the decode method.
    #     """
    #     message: list[dict] = inference_data["message"]
    #     tools = inference_data["tools"]
    #     inference_data["inference_input_log"] = {"message": repr(message), "tools": tools}

    #     # if message[0]["role"] == "system":
    #     #     system_message = []
    #     # else:
    #     #     system_message = [
    #     #         {
    #     #             "role": "system",
    #     #             "content": SYSTEM_PROMPT_FC
    #     #         }
    #     #     ]
    #     system_message = []
    #     if len(tools) > 0:
    #         json_data = {
    #             "model": MODEL,
    #             "messages": system_message + message,
    #             # "messages": [{"role": "user", "content": "GO GO"}]
    #             "function_call": "auto",
    #             "functions": tools,
    #             # "temperature":,
    #             # "profanity_check": "false"
    #         }
    #     else:
    #         json_data = {
    #             "model": MODEL,
    #             "messages": system_message + message,
    #             "function_call": "none",
    #             # "temperature": 0.001,
    #             # "profanity_check": "false"
    #         }

    #     payload = json.dumps(json_data)


    #     print("Payload for GC")
    #     print(payload)
        
    #     k = 1
    #     while True:
    #         response = requests.request("POST", URL, headers=HEADERS, data=payload)
    #         if response.status_code != 200:
    #             print(response)
    #             print(f"Sleeping {k} seconds.")
    #             time.sleep(k)
    #             k += 1
    #             if k == 6:
    #                 response.raise_for_status()
    #         else:
    #             break
                
    #     return response.json()

    # Via gigachat lib
    def _query_FC(self, inference_data: dict):
        """
        Call the model API in FC mode to get the response.
        Return the response object that can be used to feed into the decode method.
        """
        message: list[dict] = inference_data["message"]
        tools = inference_data["tools"]
        inference_data["inference_input_log"] = {"message": repr(message), "tools": tools}

        # if message[0]["role"] == "system":
        #     system_message = []
        # else:
        #     system_message = [
        #         {
        #             "role": "system",
        #             "content": SYSTEM_PROMPT_FC
        #         }
        #     ]
        system_message = []
        if len(tools) > 0:
            json_data = {
                # "model": MODEL,
                "messages": system_message + message,
                # "messages": [{"role": "user", "content": "GO GO"}]
                "function_call": "auto",
                "functions": tools,
                "temperature": 0.001,
                # "profanity_check": "false"
            }
        else:
            json_data = {
                # "model": MODEL,
                "messages": system_message + message,
                "function_call": "auto",
                "temperature": 0.001,
                # "profanity_check": "false"
            }

        # print("Payload for GC")
        # print(json_data)
            
        k = 1

        start_time = time.time()
        while True:
            try:
                response = self.client.chat(json_data)
            except ResponseError as e:
                print(e)
                print(f"Sleeping {k} seconds.")
                time.sleep(k)
                k += 1
                if k == 4:
                    raise e

            else:
                break
        end_time = time.time()

        # print("GC Response")
        # print(response)
                
        return response, end_time - start_time
        
        

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

        inference_data["tools"] = tools

        return inference_data

    def _parse_query_response_FC(self, api_response: any) -> dict:
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
        
        # IMPORTANT: So far GC is capable of calling only one function.
        
        # Via requests
        # try:            
        #     model_responses = [
        #         {func_call["name"]: func_call["arguments"]}
        #         for func_call in [api_response["choices"][0]["message"]["function_call"]]
        #     ]
        # except:
        #     model_responses = api_response["choices"][0]["message"]["content"]
        
        # model_responses_message_for_chat_history = api_response["choices"][0]["message"]
            
        # print("CHAT HISTORY")
        # print(model_responses_message_for_chat_history)

        # return {
        #     "model_responses": model_responses,
        #     "model_responses_message_for_chat_history": model_responses_message_for_chat_history,
        #     # "functions_state_id": functions_state_id,
        #     # "function_name": 
        #     # "tool_call_ids": tool_call_ids,
        #     # "input_token": api_response.usage.prompt_tokens,
        #     # "output_token": api_response.usage.completion_tokens,
        #     "input_token": api_response["usage"]["prompt_tokens"],
        #     "output_token": api_response["usage"]["completion_tokens"],
        # }

        
        # Via gigachat lib
        try:
            model_responses = [
                {func_call.name: func_call.arguments}
                for func_call in [api_response.choices[0].message.function_call]
            ]
            functions_state_id = [api_response.choices[0].message.functions_state_id]
        except:
            model_responses = api_response.choices[0].message.content
            functions_state_id = ["none"]

        
        # model_responses_message_for_chat_history = api_response.choices[0].message.dict()
        model_responses_message_for_chat_history = api_response.choices[0].message.dict()
        
        # if "function_call" not in model_responses_message_for_chat_history.keys():
        #     model_responses_message_for_chat_history["function"] = "none"
        # else:
        #     for k, v in model_responses_message_for_chat_history["function_call"]["arguments"].items():
        #         if model_responses_message_for_chat_history["function_call"]["arguments"][k] == True:
        #             model_responses_message_for_chat_history["function_call"]["arguments"][k] = "true"
        #         elif model_responses_message_for_chat_history["function_call"]["arguments"][k] == False:
        #             model_responses_message_for_chat_history["function_call"]["arguments"] = "false"
        #         elif model_responses_message_for_chat_history["function_call"]["arguments"] == None:
        #             model_responses_message_for_chat_history["function_call"]["arguments"] = "null"
            
        # print("CHAT HISTORY")
        # print(model_responses_message_for_chat_history)

        return {
            "model_responses": model_responses,
            "model_responses_message_for_chat_history": model_responses_message_for_chat_history,
            # "functions_state_id": functions_state_id,
            # "function_name": 
            # "tool_call_ids": tool_call_ids,
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
        self, inference_data: dict, execution_results: list[str], model_response_data: dict
    ) -> dict:
        """
        Add the execution results to the chat history to prepare for the next turn of query.
        Some models may need to add additional information to the chat history, such as tool call IDs.
        """
        
        # Add the execution results to the current round result, one at a time
        for execution_result in execution_results:
            tool_message = {
                "role": "function",
                "content": execution_result,
            }
            inference_data["message"].append(tool_message)

        return inference_data
        # for execution_result, tool_call_id in zip(
        #     execution_results, model_response_data["tool_call_ids"]
        # ):
        #     tool_message = {
        #         "role": "function",
        #         "content": execution_result,
        #         "name": tool_call_id,
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

    def _parse_query_response_prompting(self, api_response: any) -> dict:
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
        self, inference_data: dict, execution_results: list[str], model_response_data: dict
    ) -> dict:
        """
        Add the execution results to the chat history to prepare for the next turn of query.
        Some models may need to add additional information to the chat history, such as tool call IDs.
        """
        raise NotImplementedError
        




