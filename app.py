import streamlit as st
import numpy as np
from ollama import Client
import json
import time

st.set_page_config(page_title="LLM Architecture Sandbox", layout="wide")

# --- SIDEBAR CONFIGURATION ---
with st.sidebar:
    st.header("🌐 Remote GPU Configuration")
    
    # 💡 FIX 1: Hardcode your actual 10.22.x.x IP here so it defaults correctly on boot!
    gpu_ip = st.text_input("External GPU Server IP", value="10.22.39.192") 
    gpu_port = st.text_input("Ollama Port", value="11434")
    
    OLLAMA_HOST = f"http://{gpu_ip}:{gpu_port}"
    
    # Initialize client safely
    client = Client(host=OLLAMA_HOST)
    
    # 💡 FIX 2: Connection sanity check block
    connect_status = st.empty()
    if st.button("🔌 Test Server Connection"):
        try:
            # Try to list models on the remote server as a handshake
            client.list()
            connect_status.success(f"🟢 Connected to {gpu_ip} successfully!")
        except Exception as e:
            connect_status.error(f"🔴 Cannot reach server. Is OLLAMA_HOST=0.0.0.0 set on the host? Error: {e}")

st.title("🏗️ Foundational LLM Architecture Sandbox")
st.caption("Interactively run and audit Chains, Tools, Agents, and RAG Systems powered by your remote GPU server.")

# Only render the workspace tabs if an actual IP is provided
if gpu_ip == "localhost" or "X.X" in gpu_ip:
    st.warning("⚠️ Please enter your remote GPU server's exact IP address in the sidebar to unlock the sandbox.")
    st.stop() # Stops Streamlit from running further and crashing

tab1, tab2, tab3, tab4 = st.tabs(["🔗 1. Chains", "🛠️ 2. Tools", "🤖 3. Agents", "📚 4. RAG Systems"])

# ==========================================
# TAB 1: STATIC CHAINS
# ==========================================
with tab1:
    st.header("Linear Sequence Chain")
    st.markdown("This pattern executes deterministic data steps sequentially.")
    
    chain_input = st.text_area("Product Feature Idea:", "A rugged, waterproof smart backpack with solar charging panels.")
    
    if st.button("Run Sequence Chain", type="primary"):
        try:
            with st.spinner("Step 1: Extracting Core Technical Components..."):
                p1 = f"Identify and extract the technical features from this description into a bulleted list:\n{chain_input}"
                r1 = client.generate(model="mistral:latest", prompt=p1)
                st.info(f"**Step 1 Output:**\n\n{r1['response']}")
                
            with st.spinner("Step 2: Transforming Structure into standard JSON Manifest..."):
                p2 = f"Convert this feature checklist into a clean JSON structure array with the keys 'component' and 'risk_factor':\n{r1['response']}"
                r2 = client.generate(model="mistral:latest", prompt=p2, format="json")
                st.code(r2['response'], language="json")
        except Exception as e:
            st.error(f"Inference failed. Check your remote network link: {e}")

# ==========================================
# TAB 2: FUNCTION CALLING (TOOLS)
# ==========================================
with tab2:
    st.header("Deterministic Tool Registration")
    
    def calculate_compound_interest(principal, rate, years):
        amount = principal * (1 + rate / 100) ** years
        return f"Final Amount: ${amount:,.2f} (Interest Earned: ${amount - principal:,.2f})"

    tool_definition = [{
        "type": "function",
        "function": {
            "name": "calculate_compound_interest",
            "description": "Calculate interest earnings and balance configurations accurately.",
            "parameters": {
                "type": "object",
                "properties": {
                    "principal": {"type": "number"},
                    "rate": {"type": "number"},
                    "years": {"type": "number"}
                },
                "required": ["principal", "rate", "years"]
            }
        }
    }]
    
    tool_query = st.text_input("Ask a math question:", "What will my balance be if I invest $10,000 at a 6.5% interest rate for 12 years?")
    
    if st.button("Query Tool Engine"):
        try:
            res = client.chat(model="mistral:latest", messages=[{"role": "user", "content": tool_query}], tools=tool_definition)
            if res.message.tool_calls:
                for call in res.message.tool_calls:
                    st.warning(f"🎯 Model requested Tool execution: `{call.function.name}`")
                    args = call.function.arguments
                    if call.function.name == "calculate_compound_interest":
                        runtime_output = calculate_compound_interest(args['principal'], args['rate'], args['years'])
                        st.success(f"📟 Tool Output:\n\n{runtime_output}")
            else:
                st.info(res.message.content)
        except Exception as e:
            st.error(f"Tool run failed: {e}")

# ==========================================
# TAB 3: AUTONOMOUS AGENTS
# ==========================================
with tab3:
    st.header("Stateful Agent Cognitive Loop")
    agent_goal = st.text_input("Define Agent Objective:", "Audit layout operations and generate a deployment matrix outline.")
    
    if st.button("Initialize Agent Loop"):
        try:
            agent_messages = [
                {"role": "system", "content": "You are an autonomous operations agent. Respond concisely."},
                {"role": "user", "content": agent_goal}
            ]
            for iteration in range(1, 3):
                with st.status(f"Evaluating Execution Turn {iteration}...", expanded=True):
                    response = client.chat(model="mistral:latest", messages=agent_messages)
                    st.markdown(response.message.content)
                    agent_messages.append(response.message)
                    agent_messages.append({"role": "user", "content": "Proceed to the subsequent step."})
                    time.sleep(0.5)
        except Exception as e:
            st.error(f"Agent iteration failed: {e}")

# ==========================================
# TAB 4: RETRIEVAL-AUGMENTED GENERATION (RAG)
# ==========================================
with tab4:
    st.header("Retrieval-Augmentation Matrix")
    KNOWLEDGE_BASE = [
        "Company Policy Protocol 991: All off-site server clusters must utilize encrypted WireGuard network interfaces.",
        "Operational Standard 402: Emergency system reboots require explicit validation signoffs from two Level-3 DevOps managers."
    ]
    st.write("Corporate Knowledge Base Chunks:", KNOWLEDGE_BASE)
    rag_query = st.text_input("Ask a question regarding internal guidelines:", "What are the rules for triggering an emergency reboot?")
    
    if st.button("Run Context-Injected RAG"):
        try:
            q_emb = client.embeddings(model="nomic-embed-text", prompt=rag_query)["embedding"]
            best_chunk = KNOWLEDGE_BASE[0]
            best_score = -1
            
            for chunk in KNOWLEDGE_BASE:
                c_emb = client.embeddings(model="nomic-embed-text", prompt=chunk)["embedding"]
                score = np.dot(q_emb, c_emb) / (np.linalg.norm(q_emb) * np.linalg.norm(c_emb))
                if score > best_score:
                    best_score = score
                    best_chunk = chunk
            
            st.caption(f"🎯 Match Located (Score: {best_score:.4f}):")
            st.info(best_chunk)
            
            augmented_prompt = f"Using ONLY the context, answer the query.\nCONTEXT: {best_chunk}\nQUERY: {rag_query}"
            final_generation = client.generate(model="mistral:latest", prompt=augmented_prompt)
            st.markdown(final_generation["response"])
        except Exception as e:
            st.error(f"RAG system mapping failed: {e}")