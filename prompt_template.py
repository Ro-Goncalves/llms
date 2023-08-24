from langchain import PromptTemplate

print("An example prompt with no input variables")

template = "Tell me a joke."
prompt = PromptTemplate.from_template(template)

print("Prompt variable: " + str(prompt))
print("Input variable: " + str(prompt.input_variables))
print("Template: " + str(prompt.template))

print("\nAn example prompt with one input variables")

template = "tell me a {adjective} joke."
prompt = PromptTemplate.from_template(template)

print("Input variable: " + str(prompt.input_variables))
print("Template: " + str(prompt.template))

formatted_prompt = prompt.format(adjective="funny")
print("Format prompt template: " + str(formatted_prompt))

print("\nAn example prompt with multiple input variables")

template = "Tell me a {adjective} joke about {content}."
prompt = PromptTemplate.from_template(template)

print("Prompt variable: " + str(prompt))
print("Input variable: " + str(prompt.input_variables))
print("Template: " + str(prompt.template))

formatted_prompt = prompt.format(adjective="funny", content="chickens")
print("Format prompt template: " + str(formatted_prompt))
