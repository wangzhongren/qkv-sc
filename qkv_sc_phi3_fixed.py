import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from collections import defaultdict
import re
import json

class QKV_SC_Phi3_Fixed:
    def __init__(self, model_name="models/phi-3-mini-4k-instruct"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        print("Loading Phi-3 Mini...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            attn_implementation="eager"
        )
        self.model.eval()
        self.memory = defaultdict(lambda: defaultdict(list))

        # Fix: Phi-3 uses qkv_proj
        last_layer = self.model.model.layers[-1]
        hidden_size = self.model.config.hidden_size
        qkv_proj = last_layer.self_attn.qkv_proj.weight.data
        self.W_Q = qkv_proj[0:hidden_size, :].clone()
        self.W_K = qkv_proj[hidden_size:2*hidden_size, :].clone()
        self.W_V = qkv_proj[2*hidden_size:3*hidden_size, :].clone()

    def extract_main_entity(self, text):
        words = re.findall(r'\b[A-Z][a-zA-Z]*\b', text)
        skip = {"The", "This", "That", "It", "They", "We", "You", "I", "He", "She"}
        for w in words:
            if w not in skip:
                return w
        return words[0] if words else "unknown"

    def decompose_with_phi3(self, statement):
        # 强化提示词：强制 JSON 格式 + 示例
        prompt = f"""Decompose the statement into semantic roles. Output ONLY valid JSON with keys: "what", "how", "why", "when", "source". Use "" if not mentioned.

Statement: "Apple was founded in 1976."
JSON: {{"what": "Apple was founded.", "how": "", "why": "", "when": "1976", "source": ""}}

Statement: "{statement}"
JSON:"""

        messages = [{"role": "user", "content": prompt}]
        input_ids = self.tokenizer.apply_chat_template(
            messages, return_tensors="pt", add_generation_prompt=True
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=150,
                do_sample=False,  # greedy decoding
                # temperature=0.0,  # remove to avoid warning
                pad_token_id=self.tokenizer.eos_token_id
            )

        response = self.tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
        response = response.strip()

        # 尝试提取 JSON（即使前后有杂文）
        try:
            # 找到第一个 {{ 和最后一个 }}
            start = response.find('{')
            end = response.rfind('}')
            if start != -1 and end != -1:
                json_str = response[start:end+1]
                roles = json.loads(json_str)
                return {k: roles.get(k, "") for k in ["what", "how", "why", "when", "source"]}
        except Exception as e:
            print(f"⚠️ JSON parse failed: '{response}'")
        return {k: "" for k in ["what", "how", "why", "when", "source"]}

    def get_qkv_from_text(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(self.model.device)
        with torch.no_grad():
            hidden = self.model.model(inputs.input_ids, attention_mask=inputs.attention_mask).last_hidden_state[0]
        tokens = self.tokenizer.convert_ids_to_tokens(inputs.input_ids[0])
        Q = torch.matmul(hidden, self.W_Q.T)
        K = torch.matmul(hidden, self.W_K.T)
        V = torch.matmul(hidden, self.W_V.T)
        return Q.cpu().numpy(), K.cpu().numpy(), V.cpu().numpy(), tokens

    def store_from_statement(self, statement):
        print(f"\n📥 Processing: {statement}")
        roles = self.decompose_with_phi3(statement)
        entity = self.extract_main_entity(statement)
        print(f"🏷️ Entity: {entity}")
        print("Roles:", {k: v for k, v in roles.items() if v})

        for role, content in roles.items():
            if content.strip():
                print(f"  → Storing [{role}]: {content}")
                _, K, V, tokens = self.get_qkv_from_text(content)
                self.memory[entity][role].append({
                    "keys": K, "values": V, "tokens": tokens, "text": content
                })

    def classify_role(self, question):
        q = question.lower()
        if "how" in q: return "how"
        if "why" in q: return "why"
        if "when" in q: return "when"
        if "source" in q or "from" in q: return "source"
        return "what"

    def retrieve(self, question):
        entity = self.extract_main_entity(question)
        role = self.classify_role(question)

        if entity not in self.memory or role not in self.memory[entity]:
            print(f"🔍 No memory for entity '{entity}' and role '{role}'")
            return None

        Q_q, _, _, tokens_q = self.get_qkv_from_text(question)
        entity_clean = entity.lower()

        # 找 entity 对应的 token（忽略 ▁ 和大小写）
        query_vec = Q_q[0]
        for i, tok in enumerate(tokens_q):
            clean_tok = tok.replace("▁", "").lower()
            if clean_tok == entity_clean:
                query_vec = Q_q[i]
                break

        best_score, best_entry = -1e9, None
        for entry in self.memory[entity][role]:
            scores = np.dot(entry["keys"], query_vec)
            max_score = np.max(scores)
            if max_score > best_score:
                best_score = float(max_score)
                best_entry = entry

        if best_entry:
            return {"entity": entity, "role": role, "retrieved_text": best_entry["text"], "score": best_score}
        return None


# === 运行示例 ===
if __name__ == "__main__":
    cache = QKV_SC_Phi3_Fixed()

    # 存储
    cache.store_from_statement("Zorbex is good because it uses quantum chips.")

    # 检查内存是否真的存了
    print("\n=== MEMORY CONTENT ===")
    for ent, roles in cache.memory.items():
        for role, entries in roles.items():
            for e in entries:
                print(f"{ent} - {role}: {e['text']}")

    # 检索
    result = cache.retrieve("what is Zorbex?")
    if result:
        print(f"\n✅ Retrieved: {result['retrieved_text']} (score: {result['score']:.2f})")
    else:
        print("\n❌ Retrieval failed.")