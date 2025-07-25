# Seaside Sips ☕ – AI-Powered Coffee Shop Chatbot

Seaside Sips is a full-stack AI-powered coffee shop experience that uses a multi-agent system to interact with customers, take orders, recommend items, and provide detailed menu insights — all in real time.

## 🌟 Features

- 💬 **Natural Conversation** via LLM-powered chatbot
- 📦 **Order Agent**: Handles customer orders with step-by-step tracking
- 🧠 **Recommendation Agent**: Suggests items based on Apriori association rules and popular picks
- 📚 **Detail Agent**: Retrieves item descriptions from a vector store (Pinecone)
- 🚧 **Guard Agent**: Screens out irrelevant or inappropriate queries
- 🧭 **Classification Agent**: Routes user input to the appropriate specialized agent
- 🛡️ **Modular Agent Design**: Each agent is isolated and follows the `AgentProtocol` interface

---
### 🧠 Agent Overview
| Agent                 | Purpose                                   |
| --------------------- | ----------------------------------------- |
| `GuardAgent`          | Filters inappropriate or irrelevant input |
| `ClassificationAgent` | Chooses the correct agent to handle input |
| `OrderTakingAgent`    | Collects and validates user orders        |
| `RecommendationAgent` | Suggests items using Apriori/popularity   |
| `DetailsAgent`        | Pulls item descriptions via Pinecone      |


## 🛠️ Tech Stack

| Layer            | Technology                               |
|------------------|------------------------------------------|
| Language Model   | Meta LLaMA 3 via RunPod                  |
| Vector DB        | Pinecone                                 |
| Database         | Firebase (for storing unstructured data) |
| Backend          | Python (modular agent architecture)      |
| Frontend         | React (in progress)                      |
| ML Techniques    | Apriori Algorithm (for recommendations)  |
| Env Management   | `dotenv`, Conda                          |

Python • OpenAI API (via RunPod) • Pinecone • Firebase • React.js
Apriori Algorithm • dotenv • Vector Search • LLM-Orchestrated Agents

---

## 📂 Project Structure
```bash
latte_chat/
│
├── python_code/
│ ├── api/
│ │ ├── development_code.py # Main run loop
│ │ ├── agents/
│ │ │ ├── order_taking_agent.py
│ │ │ ├── recommendation_agent.py
│ │ │ ├── details_agent.py
│ │ │ ├── guard_agent.py
│ │ │ ├── classification_agent.py
│ │ │ └── utils.py
│ │ └── recommendation_objects/
│ │ ├── apriori_recommendation.json
│ │ └── popularity_recommendation.csv
│
├── frontend/ # React app (in progress)
└── README.md
```
---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/seaside-sips.git
cd seaside-sips
```

### 2. Setup environment
```
conda create -n seaside-sips python=3.10
conda activate seaside-sips
pip install -r requirements.txt
```
### 3. Install required packages:
```
pip install -r requirements.txt
```

### 4. Add an env. file
```
RUNPOD_TOKEN=your_runpod_api_key
RUNPOD_CHATBOT_URL=your_chatbot_url
MODEL_NAME=meta-llama-3-8b-instruct
```

### 5. Run backend code
```
cd python_code
python api/development_code.py
```

### Example output:
```
User: I want to order a latte
Bot: Great choice! You've ordered 1 Latte. Would you like to add anything else?

User: One chocolate croissant please
Bot: You've ordered a Latte and a Chocolate Croissant.
      Total: $8.50

      Here are some recommendations to go with your order:
      • Sugar Free Vanilla Syrup: A sweet and creamy addition
      • Croissant: Flaky and buttery treat
```
### 📝 TODO
 Finalize and deploy the frontend

 Add user authentication

 Improve error handling and fallback responses