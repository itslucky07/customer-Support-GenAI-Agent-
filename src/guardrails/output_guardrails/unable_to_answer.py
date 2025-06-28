class UnableToAnswerGuardrail:
    def __init__(self):
        return
    
    def apply(self, response):
        if("I cannot help with financial advice" in response["answer"]):
            response["context"] = []
        elif("I cannot answer this querry as the response conatins financial advice." in response["answer"]):
            response["context"] = []
        
        return response
    

unable_to_answer_guardrail = UnableToAnswerGuardrail()
