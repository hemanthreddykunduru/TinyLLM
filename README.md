# Custom LLM System

This project is a **fully self-contained Large Language Model** (LLM) trainer and inference system built in PyTorch.  
It allows you to:

- Train your own transformer-based language model on custom text data.
- Use either **BPE** or **SentencePiece** tokenization.
- Continuously improve and extend the model.
- Generate responses or text from prompts.
- Interact with the model via a friendly console menu.

---

## 🚀 Features

✅ Transformer-based architecture with configurable depth and width  
✅ BPE and SentencePiece tokenizers  
✅ Continuous training and vocabulary expansion  
✅ Mixed precision training (if GPU supports)  
✅ Interactive menus for training and response generation

---

## 🛠️ Project Structure


---

## 📈 System Flow

Below is a **Mermaid diagram** explaining the main process:

```mermaid
flowchart TD
    A[Start Application] --> B{Main Menu}
    B --> |1: Training Mode| C[Init Trainer]
    B --> |2: Response Mode| G[Init Response Generator]
    B --> |3: System Info| L[Show System Info]
    B --> |4: Exit| M[Terminate]
    
    C --> D{Trainer Menu}
    D --> |1: Train from File| E[Load File & Train]
    D --> |2: Train from Text| F[Input Text & Train]
    D --> |3: Continue Training| K[Load New Data & Train]
    D --> |4: Check Status| J[Show Training Status]
    D --> |5: Back| B

    G --> H{Response Menu}
    H --> |1: Ask Question| I[Generate Response]
    H --> |2: Chat Mode| N[Chat Loop]
    H --> |3: Generate Text| O[Text Generation]
    H --> |4: Model Info| P[Show Model Info]
    H --> |5: Back| B

    %% Explanations
    click B callback "Main menu allows choosing between Training and Response modes."
    click D callback "Training Menu: options to train, continue training, or check status."
    click H callback "Response Menu: ask questions, chat, or generate text."
```

