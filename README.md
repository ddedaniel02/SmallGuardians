# SmallGuardians
A framework for detecting and evading Jailbreak attacks in your architectures using different modules. 
More and more systems and applications are integrating solutions based on Large Language Models (LLMs), which are capable of performing various tasks related to natural language processing. However, these models are not exempt from threats, especially attacks that compromise their performance and security. One of the most relevant attacks is the jailbreak, which seeks to bypass the restrictions imposed on the model in order to obtain responses that would normally be blocked for ethical, legal, or security reasons. In this context, we propose SmallGuardians, a modular architecture dedicated to the detection and mitigation of jailbreak-type attacks. This solution allows users to customize protection measures and adapt them to their needs. SmallGuardians stands out for its modular design, adaptability to new types of attacks, and efficiency in real-time detection.

# How to use
1. Install the requirements list:
   pip install -r requirements.txt
2. Import the module _smallguardians.py_ and the class _SmallGuardians_
<pre> ```from smallguardians import SmallGuardians ``` </pre>
3. Now any of the available detection solutions must be used. The current options are
      - _MLPClassifier_: a MLP model that binary classifies the input
      - _SLMRAG_: an SLM model based on RAG that uses current jailbreak and safe examples for reference and can offer reasoning behind the classification: JAILBREAK or SAFE. _SLMRAG_ not only is able to classify    user inputs, but also evaluates the protected model responses as TOXIC or SAFE.
   Along with these classification methods there is a defense function that currently has three modalities:
      - Blocking: In case the input is classified as JAILBREAK or TOXIC, the system will raise an exception that will finish the current session between user and aplication (of course the user can catch the exception and perform additional steps or whatever the user consider)
      - Flagging: If the input is malicious, additional text will be appended at the beginning of the input, like a warning, telling the model to take the necessary considerations and answer carefully as the input was classified as JAILBREAK (Only available for classification and not evaluation).
      -  Default Response: Useful for the response evaluation step, in case the output is TOXIC, a default response can be returned, avoiding the model to give malicious answers to the user.
   The 
   
