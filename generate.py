\
    from typing import Dict, Any, List
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    import os

    def build_control_block(strategy_weights:Dict[str,float], prefs:Dict[str,Any], persona:Dict[str,str], updates:Dict[str,Any]) -> str:
        sw = ";".join([f"{k}:{v:.2f}" for k,v in strategy_weights.items()])
        pf = ";".join([f"{k}={v}" for k,v in prefs.items()])
        pe = ";".join([f"{k}={v}" for k,v in persona.items()])
        up = f"retract={updates.get('retract',[])} add={updates.get('add',[])}"
        return f"<CTRL|strategy={sw}|prefs={pf}|persona={pe}|updates={up}>"

    # Generator wrapper - uses HF model if available, otherwise fallback to stub
    class GeneratorWrapper:
        def __init__(self, model_name="google/flan-t5-small", device="cpu"):
            self.device = device
            self.model_name = model_name
            self._loaded = False
            self._use_stub = False
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
                self._loaded = True
            except Exception as e:
                print("Could not load HF generator model, will use stub. Error:", e)
                self._use_stub = True

        def generate(self, history:List[Dict[str,str]], control:str, feedback:Dict[str,Any], max_length=128):
            if self._use_stub:
                return self.stub_generate(history, control, feedback)
            # build input string
            hist_text = " ".join([h["text"] for h in history[-6:]])
            inp = f"{control} USER: {hist_text} FEEDBACK: {feedback.get('feedback_type','none')}"
            toks = self.tokenizer(inp, return_tensors="pt", truncation=True).to(self.device)
            ids = self.model.generate(**toks, max_length=max_length, num_beams=2)
            out = self.tokenizer.decode(ids[0], skip_special_tokens=True)
            return out

        def stub_generate(self, history:List[Dict[str,str]], control:str, feedback:Dict[str,Any]):
            ack = ""
            ftype = feedback.get("feedback_type","none")
            if ftype == "correction":
                ack = "Thanks for clarifying—I'll update that."
            elif ftype == "preference":
                ack = "Got it—I’ll respect your preference."
            elif ftype == "dissatisfaction":
                ack = "You’re right—that probably felt generic. Let me try again."
            elif ftype == "refinement":
                ack = "Sure—let me be more concrete."
            else:
                ack = "Thanks for sharing."
            if "strategy=" in control:
                strat = control.split("strategy=")[1].split("|")[0]
                top = sorted([kv.split(":") for kv in strat.split(";")], key=lambda x: float(x[1]), reverse=True)[0][0]
            else:
                top = "reflection"
            if top == "question":
                core = "Could you share a bit more about what’s been the heaviest part for you right now?"
            elif top == "reflection":
                core = "It sounds like you’re carrying a lot—especially with everything happening at once."
            elif top == "validation":
                core = "Your feelings make sense given what you’re dealing with."
            elif top == "suggestion":
                core = "One small step tonight could be to write down the top 3 tasks and pick just one."
            elif top == "resource":
                core = "If it helps, I can point you to a short breathing exercise or a simple planning template."
            else:
                core = "Maybe looking at this from a different angle could ease some pressure."
            return f"{ack} {core}"
