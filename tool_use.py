import ollama

INITIAL_PROMPT = 'What is 5 + 5 * 20 - 3 + 69 ?'
ANSWER_MODEL = 'llama3-chatqa:8b'
CRITIC_MODEL = 'llama3.1'

TOOLS = [
    {
        'type': 'function',
        'function': {
            'name': 'call_llm',
            'description': 'Chat with another LLM',
            'parameters': {
                'type': 'object',
                'properties': {
                    'model': {
                        'type': 'string',
                        'description': 'The LLM\'s name',
                    },
                    'new_message': {
                        'type': 'string',
                        'description': 'A message to append to the chat',
                    },
                },
                'required': ['model', 'new_message'],
            },
        },
    }
]

TOOL_USER_MESSAGES = [
        {
            'role': 'user',
            'content': f'You are an LLM in conversation with another LLM. Chat with the model "{ANSWER_MODEL}" with'
                       f' the prompt {INITIAL_PROMPT} and analyse and critique its response.'},
]


OLLAMA_CLIENT = ollama.Client()

ITERATIONS = 20


def call_llm(model, new_message):

    response = OLLAMA_CLIENT.chat(model=model, messages=[{
        'role': 'user',
        'content': new_message,
    }])
    print(f"{model}: {response["message"]["content"]}")
    return response["message"]["content"]


available_functions = {
    'call_llm': call_llm,
}


def run(model: str):
    print(f"{model}: {INITIAL_PROMPT}")

    # First API call: Send the query and function description to the model
    response = OLLAMA_CLIENT.chat(
        model=model,
        messages=TOOL_USER_MESSAGES,
        tools=TOOLS,
    )

    # Check if the model decided to use the provided function
    tool_calls = response['message'].get('tool_calls')
    if tool_calls is not None:
        print(f"The model wants to use tools: {tool_calls}")
        tool_call_count = 0

        # Add the model's response to the conversation history
        TOOL_USER_MESSAGES.append(response['message'])

        for tool in tool_calls:
            print(f"Trying tool call: {tool}")
            function_name = tool['function']['name']
            function_to_call = available_functions.get(function_name)
            if not function_name:
                print("No function name found in tool call")
                continue
            # try:
            function_response = function_to_call(**tool['function']['arguments'])
            # except Exception as e:
            #     print(f"Exception raised during tool call: {e}")
            #     continue
            tool_call_count += 1

            # Add function response to the conversation
            TOOL_USER_MESSAGES.append(
                {
                    'role': 'tool',
                    'content': function_response,
                }
            )
        if not tool_call_count:
            print("No successful tool calls happened")
            return
    else:
        print("The model didn't use any function. Its response was:")
        print(response['message']['content'])
        return

    # Second API call: Get final response from the model
    final_response = OLLAMA_CLIENT.chat(model=model, messages=TOOL_USER_MESSAGES)
    print(f"{model}: {final_response['message']['content']}")


run(CRITIC_MODEL)
