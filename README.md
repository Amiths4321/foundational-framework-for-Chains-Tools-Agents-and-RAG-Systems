🏗️ Foundational LLM Architecture SandboxA unified engineering playground built with Streamlit to interactively demonstrate, benchmark, and audit the four core patterns of modern Large Language Model (LLM) applications: Chains, Tools, Agents, and RAG Systems.


This repository is optimized to run completely offline by offloading all heavy compute—text generation, function invocation, and embedding vectorization—to an external GPU server hosting Ollama (mistral:latest & nomic-embed-text).

🧩 Architectural Patterns Covered1. 🔗 Linear Sequence ChainsDemonstrates deterministic pipeline execution. Instead of open-ended generation, inputs flow through a fixed pipeline where the structured output of Step A directly feeds the prompt template of Step B.Example: Raw Product Idea $\rightarrow$ Technical Component Extraction $\rightarrow$ JSON Manifest Transformation.

2. 🛠️ Functional Tool UseImplements native function calling capabilities. The application registers explicit parameter schemas with the LLM, enabling the model to intelligently pause text generation and emit an atomic argument payload to execute precise local calculations.Example: Mathematical parsing and localized calculations for complex financial computations.

3. 🤖 Autonomous AgentsAn open-ended optimization framework where the model operates inside a multi-turn cognitive reasoning loop. Given a high-level target objective, the model continuously plans its path, evaluates its own status, and updates its strategy across successive iterations.

5. 📚 Retrieval-Augmented Generation (RAG)Mitigates model hallucinations by dynamically injecting custom corporate contexts into the prompt matrix. The module transforms data chunks into semantic embeddings and applies local vector cosine similarity calculations to isolate the exact reference data required.🛠️ Setup & Installation1. Configure the Remote GPU ServerBy default, Ollama restricts requests to localhost. To receive traffic from your local workflow engine, you must expose port 11434 on your external GPU machine.For Linux (systemd) Hosts:Open the service modifier editor:Bashsudo systemctl edit ollama.service
Insert the following lines under the [Service] block:Ini, TOML[Service]
Environment="OLLAMA_HOST=0.0.0.0"
Environment="OLLAMA_ORIGINS=*"
Save, reload system configurations, and restart the service daemon:Bashsudo systemctl daemon-reload
sudo systemctl restart ollama

2. Download Core Local WeightsOn your remote GPU machine terminal, make sure the model layers are pulled and cached in VRAM:Bashollama pull mistral:latest
ollama pull nomic-embed-text

3. Install Local DependenciesOn your local developer workstation, install the required packages:Bashpip install streamlit ollama numpy
🚀 Running the SandboxOpen app.py and modify the default configuration fallback variable around line 13 to mirror your remote GPU server's exact IP:Pythongpu_ip = st.text_input("External GPU Server IP", value="10.22.X.X")
Launch the Streamlit application interface from your workstation terminal:Bashstreamlit run app.py
Use the integrated 🔌 Test Server Connection button in the sidebar to run a live network handshake before triggering model workloads.
