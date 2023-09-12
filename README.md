# Inner-Critic-Chatbot
NOTE: This repo is a work in progress and I am still working on proper documentation. Please contact me directly if you would like to learn more details about the project.
# Overview and Problem Statement
It's no secret that many people struggle with self-doubt and confidence. It's tricky on whether or not to delineate these as mental health struggles - while depression and anxiety are certainly intertwined, tackling these issues can be more complicated than working with a therapist or psychatrist. Working out these distinctions between the "psychology" and "human" element can get even more subjective and confusing. That being said, there are certain themes that are associated with this area - fear of rejection, perfectionism, and discomfort with being vulnerable, to name a few.

With the recent advances in AI and large language models, different solutions have come up to provide more affordable and on-demand support:
- [ChatGPT](https://chat.openai.com/) can offer decent responses on its own, but it isn't really meant for these use cases. If it doesn't feel like it's helping, it may just stop the conversation and refer you to a mental health professional.
- [Inflection.ai](https://inflection.ai/) is a chat application that on its own, already does a very good job of helping a user work through their issues. It's presented as a personal assistant that can help with a variety of issues (brainstorming, journaling, roleplaying), and gives personal and empathic feel to the conversation.
- AI-based mental health platforms: Companies like [Woebot Health](https://woebothealth.com/) and [Elomia Health](https://elomia.com/) offer chat-based solutions for mental health support. Woebot operates more from the clinical side and provides validated, therapy-based techniques. Elomia is for enterprises and also takes into account depression and anxiety, but satisfaction is more based on customer feedback.

While the above tools likely work well for self-confidence struggles, there is no chatbot that is specifically fine-tuned for this use case. This application provides a chat-based solution that well attuned to these use cases.

# Environment Setup
Using scripts within this repository requires setting up a virtual env and installing dependencies. I use `conda`, `pip-tools`, and `pip` to manage dependencies, but you can use whatever tools you prefer.
```
conda activate --name myenv python=3.10
pip install pip-tools
pip-compile requirements.in
pip install -r requirements.txt
```

You also need an OpenAI API key to run some of the data generation scripts:
```
export OPENAI_API_KEY="your key here"
```

# Documentation Sections

Here are explanations / documentation for main steps of the projects. Some steps are still in progress.
- [Data](data/README.md): How initial training dataset was curated and generated.
- Models [In Progress]: Two-step modeling approach to the chat agent.
- Evaluation [In Progress]: Evaluation of Chat Agent.

# Demo
In Progress