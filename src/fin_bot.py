from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.document_loaders import RecursiveUrlLoader
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
import os
import re
from dotenv import load_dotenv
from setup_store import setup_store

load_dotenv()
retriever = setup_store()

def is_financial_query(query):
    """
    Enhanced function to check if the query is related to banking, finance, or insurance
    """
    # Comprehensive financial keywords with better coverage
    financial_keywords = [
        # Banking terms
        'bank', 'banking', 'account', 'deposit', 'withdrawal', 'loan', 'credit', 'debit',
        'mortgage', 'savings', 'checking', 'atm', 'transfer', 'branch', 'teller',
        'overdraft', 'interest rate', 'apy', 'apr', 'cd', 'certificate of deposit',
        'swift', 'ifsc', 'rtgs', 'neft', 'upi', 'netbanking', 'mobile banking',
        
        # Finance and Investment terms
        'finance', 'financial', 'investment', 'invest', 'investor', 'investing',
        'stock', 'stocks', 'share', 'shares', 'equity', 'bond', 'bonds', 'mutual fund',
        'mf', 'sip', 'portfolio', 'dividend', 'capital', 'debt', 'asset', 'liability',
        'budget', 'budgeting', 'expense', 'income', 'revenue', 'profit', 'loss',
        'ipo', 'initial public offering', 'eti', 'etf', 'index fund',
        'nifty', 'sensex', 'bse', 'nse', 'trading', 'trader', 'broker', 'brokerage',
        
        # Tax and Financial Planning
        'tax', 'taxation', 'income tax', 'gst', 'tds', 'return', 'refund', 'deduction',
        'exemption', 'section 80c', 'elss', 'credit score', 'cibil', 'fico',
        'retirement', '401k', 'ira', 'roth', 'pension', 'annuity', 'gratuity',
        'pf', 'provident fund', 'epf', 'nps', 'national pension scheme',
        'ppf', 'public provident fund', 'nsc', 'national savings certificate',
        
        # Market and Economy
        'forex', 'currency', 'exchange rate', 'cryptocurrency', 'bitcoin', 'crypto',
        'market', 'stock market', 'bear market', 'bull market', 'economy', 'economic',
        'inflation', 'deflation', 'recession', 'gdp', 'federal reserve', 'fed',
        'rbi', 'reserve bank', 'monetary policy', 'fiscal policy', 'repo rate',
        'reverse repo', 'crr', 'slr', 'mclr',
        
        # Insurance terms
        'insurance', 'insure', 'policy', 'premium', 'deductible', 'coverage', 'claim',
        'beneficiary', 'nominee', 'underwriting', 'actuarial', 'life insurance',
        'health insurance', 'medical insurance', 'auto insurance', 'car insurance',
        'vehicle insurance', 'home insurance', 'property insurance', 'fire insurance',
        'liability insurance', 'third party insurance', 'comprehensive insurance',
        'disability insurance', 'umbrella policy', 'reinsurance', 'risk assessment',
        'term insurance', 'whole life insurance', 'ulip', 'endowment',
        
        # Financial Products and Services
        'loan', 'personal loan', 'home loan', 'car loan', 'education loan',
        'gold loan', 'business loan', 'working capital', 'overdraft', 'line of credit',
        'credit card', 'debit card', 'prepaid card', 'wallet', 'payment gateway',
        'emi', 'equated monthly installment', 'down payment', 'principal', 'tenure',
        'amortization', 'compound interest', 'simple interest', 'roi', 'return on investment',
        'cagr', 'compound annual growth rate', 'volatility', 'risk', 'beta', 'alpha',
        
        # Financial Ratios and Analysis
        'pe ratio', 'price to earnings', 'pb ratio', 'debt to equity', 'current ratio',
        'quick ratio', 'roe', 'return on equity', 'roa', 'return on assets',
        'ebitda', 'earnings', 'revenue', 'turnover', 'cash flow', 'balance sheet',
        'profit and loss', 'p&l', 'financial statement', 'annual report',
        
        # Digital Financial Services
        'fintech', 'digital banking', 'online banking', 'mobile payment', 'digital wallet',
        'paytm', 'gpay', 'phonepe', 'bhim', 'cryptocurrency exchange',
        'robo advisor', 'algorithmic trading', 'high frequency trading'
    ]
    
    # Convert query to lowercase for case-insensitive matching
    query_lower = query.lower().strip()
    
    # Direct keyword matching with word boundaries
    for keyword in financial_keywords:
        # Use word boundaries to avoid partial matches
        if re.search(r'\b' + re.escape(keyword.lower()) + r'\b', query_lower):
            return True
    
    # Additional financial patterns
    financial_patterns = [
        r'\b(money|cash|dollar|rupee|rs|inr|usd|payment|pay|cost|price|fee|charge|salary|wage)\b',
        r'\b(financial|economic|monetary|fiscal)\b',
        r'\b(banking|finance|insurance|investment)\b',
        r'\b(trading|market|exchange|bourse)\b',
        r'\b(fund|funds|scheme|schemes)\b',
        r'\b(rate|rates|percentage|percent|%)\b.*\b(interest|return|growth|inflation)\b',
        r'\b(buy|sell|purchase)\b.*\b(stock|share|bond|fund)\b',
        r'\b(what|how|when|where|why)\b.*\b(invest|investment|trading|finance|bank|insurance)\b',
        r'\bipo\b',  # IPO is a common abbreviation
        r'\b(nps|ppf|epf|elss|ulip|sip|etf|nsc|ncd)\b',  # Common financial abbreviations
        r'\b(hdfc|icici|sbi|axis|kotak|yes bank|pnb|bob|canara|union bank)\b',  # Bank names
        r'\b(reliance|tcs|infosys|hdfc bank|icici bank|bajaj|adani|tata)\b.*\b(stock|share|ipo)\b'
    ]
    
    for pattern in financial_patterns:
        if re.search(pattern, query_lower):
            return True
    
    return False

def create_chain():
    system_prompt = (
        "You are a specialized financial assistant for question-answering tasks focused ONLY on banking, finance, and insurance topics. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise. If the question is not clear ask follow up questions.\n\n{context}"
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    llm = ChatOpenAI(model=os.getenv("MODEL"), temperature=float(os.getenv("TEMPERATURE", 0)))

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, question_answer_chain)


def create_history_aware_chain():
    system_prompt = (
        "You are a specialized financial assistant for question-answering tasks focused ONLY on banking, finance, and insurance topics. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use five sentences maximum and keep the "
        "answer concise. Try to keep original content as much possible. "
        "If the question is not clear ask follow up questions. "
        "Only provide information related to banking, finance, and insurance sectors. "
        "Provide accurate and detailed financial information when available.\n\n{context}"
    )

    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is. "
        "Focus on financial, banking, and insurance related context."
    )

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    llm = ChatOpenAI(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        openai_api_base=os.getenv("OPENAI_API_BASE"),
        model_name=os.getenv("MODEL"),
        temperature=float(os.getenv("TEMPERATURE", 0))
    )

    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    return create_retrieval_chain(history_aware_retriever, question_answer_chain)


rag_chain = create_history_aware_chain()

def create_context_from_history(history_from_client):
    chat_history = []

    for history_item in history_from_client:
        if history_item["role"] == "user":
            chat_history.append(HumanMessage(content=history_item["content"]))
        elif history_item["role"] == "assistant":
            chat_history.append(AIMessage(content=history_item["content"]))

    return chat_history


def get_response(query, history_from_client):
    """
    Enhanced response function with better domain validation
    """
    # Check if the query is related to financial domain
    if not is_financial_query(query):
        return {
            "answer": "I am only able to provide information related to finance, banking, and insurance. Please ask questions about these topics.",
            "context": []
        }
    
    try:
        context = create_context_from_history(history_from_client)
        response = rag_chain.invoke({"input": query, "chat_history": context})
        return response
    except Exception as e:
        return {
            "answer": "I apologize, but I encountered an error while processing your financial query. Please try rephrasing your question.",
            "context": []
        }


# Optional: Add a function to test the domain validation
def test_query_classification(test_queries):
    """
    Test function to check if queries are correctly classified as financial or not
    """
    for query in test_queries:
        is_financial = is_financial_query(query)
        print(f"Query: '{query}' -> Financial: {is_financial}")


# Example usage for testing
if __name__ == "__main__":
    test_queries = [
        "what is NPS?",
        "tell me about insurance",
        "reliance ipo",
        "difference between stock and bond",
        "how to make maggie",
        "btech admission process",
        "what is PPF vs NPS",
        "HDFC bank loan interest rates",
        "SBI credit card benefits"
    ]
    test_query_classification(test_queries)