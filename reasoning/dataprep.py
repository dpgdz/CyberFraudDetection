"""
prepare dataset for reasoning fine tuning
this script produces reasoning samples from your fraud data
"""

import pandas as pd
import json
from typing import List, Dict


class ReasoningDatasetCreator:
    """create reasoning samples from fraud dataset"""

    def __init__(self, csv_path: str, output_path: str):
        self.df = pd.read_csv(csv_path)
        self.output_path = output_path

        # list of common fraud indicators per domain
        self.fraud_indicators = {
            # dataset domains
            "phishing": [
                "asks for account details",
                "strange or unsafe links",
                "messages trying to cause alarm",
                "sender address mismatch",
                "poor writing in official style messages",
            ],
            "job_scams": [
                "offers with unrealistic salary",
                "asks for payments",
                "guaranteed work from home",
                "unclear job details",
                "no interview process",
            ],
            "sms": [
                "unsolicited ads",
                "short suspicious links",
                "missing opt out",
                "mass sending patterns",
                "irrelevant promo content",
            ],
            "fake_news": [
                "sensational or emotional language",
                "no clear source or evidence",
                "mixes facts with speculation",
                "blames media without proof",
                "mentions conspiracies or hidden agendas",
            ],
            "product_reviews": [
                "extreme positive or negative tone",
                "repeated generic phrases",
                "few concrete product details",
                "similar wording to many other reviews",
                "focuses on brand more than real usage",
            ],
            "political_statements": [
                "strong political bias in wording",
                "claims without verifiable data",
                "attacks person instead of policy",
                "uses fear or anger to persuade",
                "mentions secret plans or plots",
            ],
            "twitter_rumours": [
                "unconfirmed breaking news",
                "no official source linked",
                "encourages sharing before checking",
                "vague phrases like hearing reports",
                "depends on screenshots or reposts",
            ],

            # extra generic scam types (currently not used by this dataset
            "reward_scam": [
                "random prize claims",
                "claims of winning without joining",
                "fees required to receive prize",
                "pressure to act fast",
                "unclear prize information",
            ],
            "tech_support_scam": [
                "unexpected tech support offers",
                "fake security alerts",
                "requests for remote access",
                "pressure to act quickly",
                "non official contact channels",
            ],
            "refund_scam": [
                "unexpected refund message",
                "asks to confirm financial info",
                "untrusted callback numbers",
                "pushes for fast response",
                "not using official communication",
            ],
            "ssn_scam": [
                "threats about ssn status",
                "pretends to be government",
                "legal threats",
                "asks to verify ssn",
                "creates urgency",
            ],
            "popup_scam": [
                "fake security warnings",
                "fake system errors",
                "tells user to take action fast",
                "phone numbers in popup",
                "blocks normal browser use",
            ],
        }

    def create_reasoning_prompt(self, text: str, label: str, domain: str) -> str:
        """generate reasoning text from input, numeric label, and domain"""
        # normalize label so numeric labels like 0 or 1 are handled
        norm_label = self._normalize_label(label)

        # indicators depend on domain, not on label
        indicators = self.fraud_indicators.get(domain, [])

        # find matched indicators
        present_indicators = []
        text_lower = text.lower()
        for ind in indicators:
            words = ind.lower().split()
            if any(w in text_lower for w in words if len(w) > 3):
                present_indicators.append(ind)

        # choose reasoning type
        if norm_label == "legitimate":
            return self._generate_legitimate_reasoning(text)
        else:
            return self._generate_fraud_reasoning(text, domain, present_indicators)

    def _generate_fraud_reasoning(
        self, text: str, domain: str, indicators: List[str]
    ) -> str:
        """build reasoning for fraudulent messages"""

        parts = [
            "classification: fraud",
            "confidence: 95%",
            "",
            f"fraud_type: {domain}",
            "",
            "signals found:",
        ]

        for idx, ind in enumerate(indicators, 1):
            parts.append(f"{idx}. {ind}")

        parts.append("")
        parts.append("analysis:")

        text_lower = text.lower()

        if "urgent" in text_lower or "immediate" in text_lower:
            parts.append("- message uses pressure or urgency")

        if "click" in text_lower or "http" in text_lower:
            parts.append("- contains link or url that may be unsafe")

        if any(w in text_lower for w in ["won", "winner", "prize", "congratulations"]):
            parts.append("- mentions rewards without clear context")

        if any(w in text_lower for w in ["verify", "confirm", "update", "login"]):
            parts.append("- asks for sensitive user information or action")

        parts.append("")
        parts.append("risk: high")
        parts.append("recommended action: report as fraud")

        return "\n".join(parts)

    def _generate_legitimate_reasoning(self, text: str) -> str:
        """build reasoning for safe messages"""

        return """classification: legitimate
confidence: 92%

analysis:
- no clear fraud signals detected
- message context appears normal
- no strong urgency or manipulation patterns
- no suspicious links or data requests
- tone seems consistent with normal communication

risk: low
recommended action: safe to continue"""

    def create_training_data(self, format_type: str = "multi_task") -> List[Dict]:
        """create dataset entries in selected format"""

        data: List[Dict] = []

        for _, row in self.df.iterrows():
            text = row["text"]
            label = row["label"]
            domain = row["domain"]

            reasoning = self.create_reasoning_prompt(text, label, domain)

            if format_type == "multi_task":
                example = {
                    "instruction": "check this message for fraud and explain your reasoning",
                    "input": text,
                    "output": reasoning,
                }
            elif format_type == "cot":
                example = {
                    "instruction": "think step by step and classify this message",
                    "input": text,
                    "output": self._convert_to_cot(reasoning, text, label),
                }
            else:
                example = {
                    "messages": [
                        {
                            "role": "system",
                            "content": "you analyze fraud and explain your reasoning clearly",
                        },
                        {"role": "user", "content": f"analyze this: {text}"},
                        {"role": "assistant", "content": reasoning},
                    ]
                }

            data.append(example)

        return data

    def _convert_to_cot(self, reasoning: str, text: str, label: str) -> str:
        """convert reasoning into simple step by step format"""
        norm_label = self._normalize_label(label)

        return f"""step 1: read message
"{text[:100]}..."

step 2: check urgency
{'urgent tone found' if any(w in text.lower() for w in ['urgent', 'immediate', 'now']) else 'no urgency'}

step 3: check requests
{'asks for sensitive action' if any(w in text.lower() for w in ['click', 'verify', 'confirm', 'login']) else 'no risky request'}

step 4: conclusion
message is classified as {norm_label}

details:
{reasoning}"""

    def _normalize_label(self, label) -> str:
        """map numeric or other label formats to consistent text labels.

        Rules:
        - 0 or '0' -> 'legitimate'
        - 1 or '1' -> 'fraud'
        - other textual labels are returned as string
        """
        try:
            if isinstance(label, (int, float)):
                return "legitimate" if int(label) == 0 else "fraud"

            if isinstance(label, str) and label.isdigit():
                return "legitimate" if int(label) == 0 else "fraud"
        except Exception:
            pass

        return str(label)

    def save_training_data(self, format_type: str = "multi_task"):
        """write training samples to jsonl file"""

        data = self.create_training_data(format_type)
        out = self.output_path.replace(".jsonl", f"_{format_type}.jsonl")

        with open(out, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        print(f"saved {len(data)} samples")
        print(f"path: {out}")

        return out


if __name__ == "__main__":

        splits = [
            ("data/train_clean.csv", "data/fraud_train_clean.jsonl"),
            ("data/validation_clean.csv", "data/fraud_val_clean.jsonl"),
            ("data/test_clean.csv", "data/fraud_test_clean.jsonl"),
        ]

        for csv_path, out_path in splits:
            print(f"processing {csv_path}...")
            creator = ReasoningDatasetCreator(csv_path, out_path)
            creator.save_training_data(format_type="multi_task")
