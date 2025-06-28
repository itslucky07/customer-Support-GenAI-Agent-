from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftModel

import torch

class FinancialAdviceDetectorGuardrail:
    def __init__(self, model_path: str):
        """
        Initialize the guardrails with the model and tokenizer loaded once.
        """
        self.model_path = model_path

        # Load the base DistilBERT model
        num_labels = 2
        model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=num_labels)

        # Apply LoRA adaptation to the model
        self.model = PeftModel.from_pretrained(model, model_path)

        self.model.eval()

        # Load the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)

        self.threshold = 0.75
    

    def predict(self, text: str):
        """
        Perform inference for a single text input.
        """

        inputs = self.tokenizer(text, return_tensors="pt")

        device = self.model.device
        inputs = {key: value.to(device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        
        logits = outputs.logits

        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        predicted_class = torch.argmax(probabilities, dim=-1).item()

        print(f"text - {text}")
        print(f"probabilities - {probabilities}")

        max_prob = probabilities[0][predicted_class]  # Get the probability of the predicted class
        

        if max_prob >= self.threshold:
            # print(f"predicted class: {predicted_class}")
            return predicted_class  # Return the highest probability class
        
        # Fallback: Select the second-highest probability class
        sorted_probs, sorted_indices = torch.sort(probabilities[0], descending=True)
        fallback_class = sorted_indices[1].item() if len(sorted_indices) > 1 else -1  # Second-best or default fallback

        # print(f"predicted class: {fallback_class}")
        return fallback_class  # Return fallback class
    
    def apply(self, response):
        predicted_class = self.predict(response["answer"])

        if(predicted_class == 1):
            response["answer"] = "I cannot answer this querry as the response conatins financial advice."
        
        return response
    

financial_advice_detector_guardrail = FinancialAdviceDetectorGuardrail("models/financial-advice-detector/distelbert-lora-financial-advice/model")
