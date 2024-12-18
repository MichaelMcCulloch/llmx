from typing import Union, List, Dict
import os
from dataclasses import asdict
import google.generativeai as genai

from .base_textgen import TextGenerator
from ...datamodel import TextGenerationConfig, TextGenerationResponse, Message
from ...utils import cache_request, get_models_maxtoken_dict, num_tokens_from_messages, get_gcp_credentials


class GeminiTextGenerator(TextGenerator):
    def __init__(
        self,
        api_key: str = None,
        gemini_key_file: str = None,
        project_id: str = None,
        project_location: str = "us-central1",
        provider: str = "gemini",
        model: str = None,
        models: Dict = None,
    ):
        super().__init__(provider=provider)
        if api_key is None and gemini_key_file is None:
            raise ValueError(
                "Gemini API key or Gemini service account key file must be set."
            )
        if api_key:
            self.api_key = api_key
            self.credentials = None 
            self.project_id = None
            self.project_location = None
        else:
            self.project_id = project_id
            self.project_location = project_location
            self.api_key = None
            self.credentials = get_gcp_credentials(gemini_key_file) if gemini_key_file else None

        genai.configure(api_key=self.api_key, credentials=self.credentials)
        self.model_max_token_dict = get_models_maxtoken_dict(models)
        self.model_name = model or "gemini-pro"

    def format_messages(self, messages):
        gemini_messages = []
        system_messages = ""
        for message in messages:
            if message["role"] == "system":
                system_messages += message["content"] + "\n"
            else: 
                gemini_messages.append(
                    {"role": message["role"], "parts": [message["content"]]}
                )
        return system_messages, gemini_messages

    def generate(
        self,
        messages: Union[List[Dict], str],
        config: TextGenerationConfig = TextGenerationConfig(),
        **kwargs,
    ) -> TextGenerationResponse:
        use_cache = config.use_cache
        model = config.model or self.model_name
        system_prompt, messages = self.format_messages(messages)
        self.model_name = model

        max_tokens = (
            self.model_max_token_dict[model]
            if model in self.model_max_token_dict
            else 2048
        )

        gemini_config = {
            "generation_config": {
                "temperature": config.temperature,
                "top_p": config.top_p,
                "max_output_tokens": config.max_tokens or max_tokens,
            },
        }

        cache_key_params = {
            "messages": messages,
            "model": model,
            "system_prompt": system_prompt,
            **gemini_config,
        }

        if use_cache:
            response = cache_request(cache=self.cache, params=cache_key_params)
            if response:
                return TextGenerationResponse(**response)

        gen_model = genai.GenerativeModel(model)
        chat = gen_model.start_chat(history=messages, system_prompt=system_prompt)
        gemini_response = chat.send_message(
            "",
            generation_config=gemini_config["generation_config"],
        )

        response_text = [
            Message(role="assistant", content=gemini_response.text)
        ]

        response = TextGenerationResponse(
            text=response_text,
            logprobs=[],
            config=gemini_config,
            usage={
                "total_tokens": num_tokens_from_messages(
                    messages, model=self.model_name
                )
            },
            response=gemini_response,
        )

        cache_request(
            cache=self.cache, params=cache_key_params, values=asdict(response)
        )
        return response

    def count_tokens(self, text) -> int:
        return num_tokens_from_messages(text)
